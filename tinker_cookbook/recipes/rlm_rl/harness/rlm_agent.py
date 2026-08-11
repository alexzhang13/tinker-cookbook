"""Root LM driving a REPL over `context`, with `llm_query` / `rlm_query` as sub-agent calls.

With a `sub_completer` sub-calls go straight to a sampling client and a block fanning out N
prompts issues N concurrent requests. Without one they are queued as HarnessStep(subagent=True)
so the turns land in the trajectory -- what `sub_reward_lambda > 0` needs for credit assignment
-- but the rollout loop drains one sample per step, making them serial.
"""

from __future__ import annotations

import asyncio
import re
import time

from tinker_cookbook.completers import MessageCompleter
from tinker_cookbook.recipes.rlm_rl.harness import base
from tinker_cookbook.recipes.rlm_rl.harness.prompts import system_messages, turn_prompt
from tinker_cookbook.recipes.rlm_rl.harness.repl import (
    DEFAULT_COMPUTE_TIMEOUT_S,
    DEFAULT_MEM_LIMIT_BYTES,
    ExecResult,
    PythonRepl,
)
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message

MAX_REPL_OUTPUT_CHARS = 20_000
_REPL_BLOCK = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)

_SampleRequest = tuple[list[Message], "asyncio.Future[str]"]


class SubCallBudget:
    """Sub-LLM call budget shared by a root agent and every descendant it spawns.

    A per-agent budget would multiply down the tree: depth=2 with a limit of 25 permits
    25 + 25*25 calls, exactly the fan-out the cap exists to prevent.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def spend(self, n: int = 1) -> None:
        if self.used + n > self.limit:
            raise RuntimeError(f"sub-LLM call budget exhausted ({self.limit})")
        self.used += n


def find_repl_blocks(text: str) -> list[str]:
    return [m.strip() for m in _REPL_BLOCK.findall(text)]


def format_repl_outputs(
    outputs: list[ExecResult], max_chars: int = MAX_REPL_OUTPUT_CHARS
) -> Message:
    parts = []
    for i, o in enumerate(outputs):
        body = "\n\n".join(s for s in (o.stdout.rstrip(), o.stderr.rstrip()) if s) or "No output"
        if len(body) > max_chars:
            body = body[:max_chars] + f"... + [{len(body) - max_chars} chars...]"
        header = f"REPL output (block {i + 1}):" if len(outputs) > 1 else "REPL output:"
        parts.append(f"{header}\n{body}")
    return {"role": "user", "content": "\n\n".join(parts)}


class RLMAgent(base.Harness):
    def __init__(
        self,
        *,
        context: str,
        root_prompt: str | None,
        max_iterations: int = 20,
        depth: int = 1,
        child_max_iterations: int = 8,
        max_sub_calls: int = 25,
        max_subcall_chars: int = 60_000,
        repl_mem_limit_bytes: int = DEFAULT_MEM_LIMIT_BYTES,
        repl_compute_timeout_s: float = DEFAULT_COMPUTE_TIMEOUT_S,
        budget: SubCallBudget | None = None,
        sub_completer: MessageCompleter | None = None,
    ):
        self.context = context
        self.root_prompt = root_prompt
        self.max_iterations = max_iterations
        self.depth = depth
        self.child_max_iterations = child_max_iterations
        self.max_sub_calls = max_sub_calls
        self.max_subcall_chars = max_subcall_chars
        self.budget = budget if budget is not None else SubCallBudget(max_sub_calls)
        self.sub_completer = sub_completer
        self.repl_calls = 0
        # Sub-calls bypass the rollout loop and so are invisible to time/policy_sample despite
        # dominating wall-clock. Seconds are summed across concurrent calls, so seconds/wall
        # gives the effective fan-out achieved.
        self.sub_call_seconds = 0.0
        self._sub_inflight = 0
        self.sub_peak_inflight = 0
        self.messages: list[Message] = []
        self._iteration = 0
        self._requests: asyncio.Queue[_SampleRequest] = asyncio.Queue()
        self._pending: asyncio.Future[str] | None = None
        self._exec: asyncio.Task[list[ExecResult]] | None = None
        self.repl = PythonRepl(
            variables={"context": context},
            async_bindings={
                "llm_query": self._llm_query,
                "llm_query_batched": self._llm_query_batched,
                "rlm_query": self._rlm_query,
                "rlm_query_batched": self._rlm_query_batched,
            },
            mem_limit_bytes=repl_mem_limit_bytes,
            compute_timeout_s=repl_compute_timeout_s,
        )
        self._repl_mem_limit_bytes = repl_mem_limit_bytes
        self._repl_compute_timeout_s = repl_compute_timeout_s

    async def _sample(self, messages: list[Message]) -> str:
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        await self._requests.put((messages, future))
        return await future

    async def _sample_sub(self, messages: list[Message]) -> str:
        """Sample a sub-agent turn: concurrently via the completer, or serially via the queue."""
        if self.sub_completer is not None:
            return get_text_content(await self.sub_completer(messages))
        return await self._sample(messages)

    @property
    def sub_calls(self) -> int:
        """Calls spent across this agent's whole recursion tree."""
        return self.budget.used

    def _spend(self, n: int = 1) -> None:
        self.budget.spend(n)

    async def _llm_query(self, prompt: str, model: str | None = None) -> str:
        prompt = str(prompt)
        if len(prompt) > self.max_subcall_chars:
            raise ValueError(
                f"sub-LLM prompt too long ({len(prompt)} chars > {self.max_subcall_chars})"
            )
        self._spend()
        self._sub_inflight += 1
        self.sub_peak_inflight = max(self.sub_peak_inflight, self._sub_inflight)
        started = time.monotonic()
        try:
            return await self._sample_sub([{"role": "user", "content": prompt}])
        finally:
            self._sub_inflight -= 1
            self.sub_call_seconds += time.monotonic() - started

    async def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        return list(await asyncio.gather(*(self._llm_query(p) for p in prompts)))

    async def _rlm_query(self, prompt: str, model: str | None = None) -> str:
        if self.depth <= 1:
            return await self._llm_query(prompt)
        self._spend()
        child = RLMAgent(
            context=str(prompt),
            root_prompt=None,
            max_iterations=self.child_max_iterations,
            depth=self.depth - 1,
            max_sub_calls=self.max_sub_calls,
            max_subcall_chars=self.max_subcall_chars,
            repl_mem_limit_bytes=self._repl_mem_limit_bytes,
            repl_compute_timeout_s=self._repl_compute_timeout_s,
            budget=self.budget,
            sub_completer=self.sub_completer,
        )

        async def completer(messages: list[Message]) -> Message:
            return {"role": "assistant", "content": await self._sample_sub(messages)}

        step = await base.run(child, completer)
        return step.final_answer or ""

    async def _rlm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        return list(await asyncio.gather(*(self._rlm_query(p) for p in prompts)))

    async def _execute_blocks(self, codes: list[str]) -> list[ExecResult]:
        outputs = []
        for code in codes:
            outputs.append(await self.repl.execute(code))
            self.repl_calls += 1
        return outputs

    def _metrics(self) -> dict[str, float]:
        return {
            "turns": float(self._iteration),
            "repl_calls": float(self.repl_calls),
            "sub_llm_calls": float(self.sub_calls),
            "sub_llm_seconds": self.sub_call_seconds,
            "sub_llm_peak_inflight": float(self.sub_peak_inflight),
        }

    async def initial_messages(self) -> list[Message]:
        self.messages = system_messages(
            context_chars=len(self.context),
            context_type=type(self.context).__name__,
            root_prompt=self.root_prompt,
            max_sub_calls=self.budget.limit,
        ) + [turn_prompt(0, self.max_iterations)]
        return self.messages

    async def step(self, assistant_message: Message) -> base.HarnessStep:
        if self._pending is not None:
            pending, self._pending = self._pending, None
            pending.set_result(get_text_content(assistant_message))
        else:
            self.messages.append(assistant_message)
            self._iteration += 1
            codes = find_repl_blocks(get_text_content(assistant_message))
            if codes:
                self._exec = asyncio.create_task(self._execute_blocks(codes))
        return await self._advance()

    async def _advance(self) -> base.HarnessStep:
        if self._exec is not None:
            getter = asyncio.ensure_future(self._requests.get())
            done, _ = await asyncio.wait({getter, self._exec}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                messages, future = getter.result()
                self._pending = future
                return base.HarnessStep(
                    done=False, messages=messages, subagent=True, metrics=self._metrics()
                )
            getter.cancel()
            outputs, self._exec = self._exec.result(), None
            self.messages.append(format_repl_outputs(outputs))

        final = self.repl.final_answer
        if final is not None or self._iteration >= self.max_iterations:
            return base.HarnessStep(
                done=True, messages=self.messages, final_answer=final, metrics=self._metrics()
            )
        self.messages.append(turn_prompt(self._iteration, self.max_iterations))
        return base.HarnessStep(done=False, messages=self.messages, metrics=self._metrics())

    def close(self) -> None:
        if self._pending is not None and not self._pending.done():
            self._pending.set_exception(RuntimeError("harness closed"))
            self._pending = None
        if self._exec is not None:
            # Cancel before shutting the REPL down, or an in-flight block surfaces as
            # "Task exception was never retrieved" noise on every cleanup.
            self._exec.cancel()
            self._exec.add_done_callback(lambda t: t.cancelled() or t.exception())
            self._exec = None
        self.repl.close()
        while not self._requests.empty():
            _, future = self._requests.get_nowait()
            if not future.done():
                future.set_exception(RuntimeError("harness closed"))
