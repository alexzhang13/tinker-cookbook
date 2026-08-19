from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from tinker_cookbook.completers import MessageCompleter, TokenCompleter
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import Transition

Binding = Callable[..., Coroutine[Any, Any, Any]]
MakeChild = Callable[[str, str | None], Any]

current_token_completer: ContextVar[TokenCompleter | None] = ContextVar(
    "rlm_token_completer", default=None
)
current_renderer: ContextVar[Renderer | None] = ContextVar("rlm_renderer", default=None)


@dataclass
class SubCallTrace:
    messages: list[Message]
    completion: str
    kind: str = "llm_query"


class SubCallBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def spend(self, n: int = 1) -> None:
        if self.used + n > self.limit:
            raise RuntimeError(f"sub-LLM call budget exhausted ({self.limit})")
        self.used += n


class RLMTools:
    def __init__(
        self,
        *,
        sub_completer: MessageCompleter | None,
        budget: SubCallBudget,
        depth: int,
        max_depth: int,
        make_child: MakeChild,
        max_prompt_chars: int = 100_000,
        on_policy_llm_query: bool = False,
    ):
        self._sub_completer = sub_completer
        self.budget = budget
        self.depth = depth
        self.max_depth = max_depth
        self._make_child = make_child
        self._max_prompt_chars = max_prompt_chars
        self.on_policy_llm_query = on_policy_llm_query
        self.traces: list[SubCallTrace] = []
        self.child_agents: list[Any] = []
        self._inflight = 0
        self.peak_inflight = 0
        self.seconds = 0.0

    async def sample(self, messages: list[Message], *, kind: str = "llm_query") -> str:
        if self._sub_completer is None:
            raise RuntimeError("sub-LLM calls need a completer (pass sub_completer to RLMHarness)")
        text = get_text_content(await self._sub_completer(messages))
        self.traces.append(SubCallTrace(messages=messages, completion=text, kind=kind))
        return text

    async def llm_query(self, prompt: str, model: str | None = None) -> str:
        prompt = str(prompt)
        if len(prompt) > self._max_prompt_chars:
            raise ValueError(
                f"sub-LLM prompt too long ({len(prompt)} chars > {self._max_prompt_chars})"
            )
        self.budget.spend()
        self._inflight += 1
        self.peak_inflight = max(self.peak_inflight, self._inflight)
        started = time.monotonic()
        try:
            if self._use_on_policy_llm_query():
                return await self._on_policy_llm_query(prompt)
            return await self.sample([{"role": "user", "content": prompt}])
        finally:
            self._inflight -= 1
            self.seconds += time.monotonic() - started

    async def llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        return list(await asyncio.gather(*(self.llm_query(p) for p in prompts)))

    async def rlm_query(
        self, prompt: str, context: str | None = None, model: str | None = None
    ) -> str:
        if self.depth >= self.max_depth:
            raise RuntimeError("sub-agent depth limit reached")
        if context is None and self.depth >= self.max_depth - 1:
            return await self.llm_query(prompt)
        prompt = str(prompt)
        if len(prompt) > self._max_prompt_chars:
            raise ValueError(
                f"sub-LLM prompt too long ({len(prompt)} chars > {self._max_prompt_chars})"
            )
        ctx = None if context is None else str(context)
        self.budget.spend()
        self._inflight += 1
        self.peak_inflight = max(self.peak_inflight, self._inflight)
        started = time.monotonic()
        child = self._make_child(prompt, ctx)
        try:
            end = await child.run(self._child_completer(child))
            self.child_agents.append(child)
            self.traces.extend(child.tools.traces)
            return end.final_answer or ""
        finally:
            self._inflight -= 1
            self.seconds += time.monotonic() - started

    async def rlm_query_batched(
        self,
        prompts: list[str],
        contexts: list[str] | None = None,
        model: str | None = None,
    ) -> list[str]:
        if contexts is None:
            return list(await asyncio.gather(*(self.rlm_query(p, model=model) for p in prompts)))
        return list(
            await asyncio.gather(
                *(self.rlm_query(p, c, model=model) for p, c in zip(prompts, contexts, strict=True))
            )
        )

    async def launch_subagent(self, goal: str, context: str = "") -> Any:
        return await self.rlm_query(prompt=goal, context=context)

    def _use_on_policy_llm_query(self) -> bool:
        return (
            self.on_policy_llm_query
            and self.depth < self.max_depth
            and current_token_completer.get() is not None
            and current_renderer.get() is not None
        )

    async def _on_policy_llm_query(self, prompt: str) -> str:
        """Run an `llm_query` as a sub-agent that has hit the depth limit.

        A single-turn agent with no REPL: it is sampled from the training policy, keeps its
        own transition so RAO can train it, and joins `child_agents` so it is graded and
        credited to its parent like any recursive child."""
        policy = current_token_completer.get()
        renderer = current_renderer.get()
        if policy is None or renderer is None:
            return await self.sample([{"role": "user", "content": prompt}])
        child = self._make_child(prompt, "")
        messages: list[Message] = [{"role": "user", "content": prompt}]
        ob = renderer.build_generation_prompt(messages)
        ac = await policy(ob, renderer.get_stop_sequences())
        parsed, _termination = renderer.parse_response(ac.tokens)
        text = str(parsed.get("content", ""))
        child.messages = [*messages, {"role": "assistant", "content": text}]
        child.transitions.append(Transition(ob=ob, ac=ac, reward=0.0, episode_done=False))
        child.repl.set_final_answer(text)
        child.no_repl = True
        # It never executes code, so release its REPL now; `run()` is what normally closes it.
        child.close()
        self.child_agents.append(child)
        self.traces.append(SubCallTrace(messages=messages, completion=text, kind="llm_query"))
        return text

    def _child_completer(
        self, child: Any
    ) -> Callable[[list[Message]], Coroutine[Any, Any, Message]]:
        policy = current_token_completer.get()
        renderer = current_renderer.get()

        async def completer(messages: list[Message]) -> Message:
            if policy is None or renderer is None:
                text = await child.tools.sample(messages, kind="rlm")
                return {"role": "assistant", "content": text}
            ob = renderer.build_generation_prompt(messages)
            ac = await policy(ob, renderer.get_stop_sequences())
            message, _termination = renderer.parse_response(ac.tokens)
            child.transitions.append(Transition(ob=ob, ac=ac, reward=0.0, episode_done=False))
            result: Message = {"role": "assistant", "content": message.get("content", "")}
            if "tool_calls" in message:
                result["tool_calls"] = message["tool_calls"]
            return result

        return completer

    @property
    def bindings(self) -> dict[str, Binding]:
        return {
            "llm_query": self.llm_query,
            "llm_query_batched": self.llm_query_batched,
            "rlm_query": self.rlm_query,
            "rlm_query_batched": self.rlm_query_batched,
            "launch_subagent": self.launch_subagent,
        }

    def metrics(self) -> dict[str, float]:
        return {
            "sub_llm_calls": float(self.budget.used),
            "sub_llm_seconds": self.seconds,
            "sub_llm_peak_inflight": float(self.peak_inflight),
        }
