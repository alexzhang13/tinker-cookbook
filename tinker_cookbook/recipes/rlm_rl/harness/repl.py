from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from tinker_cookbook.recipes.rlm_rl.harness.base import End, Harness, Prompt, Turn
from tinker_cookbook.recipes.rlm_rl.rlm.repl import (
    DEFAULT_COMPUTE_TIMEOUT_S,
    DEFAULT_MEM_LIMIT_BYTES,
    ExecResult,
    PythonRepl,
    find_repl_blocks,
    format_repl_outputs,
)
from tinker_cookbook.renderers import get_text_content
from tinker_cookbook.renderers.base import Message

Binding = Callable[..., Coroutine[Any, Any, Any]]


class REPLHarness(Harness):
    def __init__(
        self,
        *,
        context: str,
        max_iterations: int = 20,
        mem_limit_bytes: int = DEFAULT_MEM_LIMIT_BYTES,
        compute_timeout_s: float = DEFAULT_COMPUTE_TIMEOUT_S,
        async_bindings: dict[str, Binding] | None = None,
        variables: dict[str, Any] | None = None,
    ):
        self.context = context
        self.max_iterations = max_iterations
        self.mem_limit_bytes = mem_limit_bytes
        self.compute_timeout_s = compute_timeout_s
        self.messages: list[Message] = []
        self.repl_calls = 0
        self._iteration = 0
        self.repl = PythonRepl(
            variables=dict(variables or {}),
            async_bindings=dict(async_bindings or {}),
            mem_limit_bytes=mem_limit_bytes,
            compute_timeout_s=compute_timeout_s,
        )

    def _initial_messages(self) -> list[Message]:
        return []

    def _turn_message(self, iteration: int) -> Message:
        return {"role": "user", "content": f"Turn {iteration + 1}/{self.max_iterations}:"}

    def _metrics(self) -> dict[str, float]:
        return {
            "turns": float(self._iteration),
            "repl_calls": float(self.repl_calls),
        }

    async def start(self) -> Turn:
        self.repl.set_variable("context", self.context)
        self.messages = [*self._initial_messages(), self._turn_message(0)]
        return Prompt(messages=self.messages)

    async def act(self, assistant_message: Message) -> Turn:
        self.messages.append(assistant_message)
        self._iteration += 1
        await self._maybe_run_repl(assistant_message)
        return self._next_turn()

    def close(self) -> None:
        self.repl.close()

    async def _maybe_run_repl(self, assistant_message: Message) -> None:
        codes = find_repl_blocks(get_text_content(assistant_message))
        if not codes:
            return
        outputs = await self._execute_blocks(codes)
        self.messages.append(format_repl_outputs(outputs))

    def _next_turn(self) -> Turn:
        final = self.repl.final_answer
        if final is not None or self._iteration >= self.max_iterations:
            return End(
                final_answer=final,
                messages=self.messages,
                metrics=self._metrics(),
                stop_reason="end_turn" if final is not None else "max_turns",
            )
        self.messages.append(self._turn_message(self._iteration))
        return Prompt(messages=self.messages)

    async def _execute_blocks(self, codes: list[str]) -> list[ExecResult]:
        outputs = []
        for code in codes:
            outputs.append(await self.repl.execute(code))
            self.repl_calls += 1
        return outputs
