"""Persistent in-process Python REPL with the `answer`-dict completion protocol.

Async bindings are exposed as synchronous functions inside the REPL. Model
code runs on this REPL's own dedicated thread — never the shared default
executor — because it blocks on sub-call futures for arbitrarily long, and
parking such threads in the shared pool starves every other coroutine that
needs `asyncio.to_thread` (e.g. renderers), deadlocking concurrent rollouts.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import io
import traceback
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecResult:
    stdout: str
    stderr: str


class PythonRepl:
    def __init__(
        self,
        variables: dict[str, Any],
        async_bindings: dict[str, Callable[..., Coroutine[Any, Any, Any]]] | None = None,
    ):
        self._repl_ns: dict[str, Any] = {
            "__name__": "__main__",
            "answer": {"content": "", "ready": False},
            **variables,
        }
        self._repl_ns["SHOW_VARS"] = lambda: ", ".join(
            k for k in self._repl_ns if not k.startswith("_")
        )
        for name, fn in (async_bindings or {}).items():
            self._repl_ns[name] = self._to_sync(fn)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _to_sync(self, fn: Callable[..., Coroutine[Any, Any, Any]]) -> Callable[..., Any]:
        def bound(*args: Any, **kwargs: Any) -> Any:
            assert self._loop is not None
            return asyncio.run_coroutine_threadsafe(fn(*args, **kwargs), self._loop).result()

        return bound

    async def execute(self, code: str) -> ExecResult:
        self._loop = asyncio.get_running_loop()
        return await self._loop.run_in_executor(self._executor, self._execute_sync, code)

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def _execute_sync(self, code: str) -> ExecResult:
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(code, "<repl>", "exec"), self._repl_ns)
        except Exception:
            err.write(traceback.format_exc(limit=3))
        return ExecResult(stdout=out.getvalue(), stderr=err.getvalue())

    @property
    def final_answer(self) -> str | None:
        answer = self._repl_ns.get("answer")
        if isinstance(answer, dict) and answer.get("ready"):
            return str(answer.get("content", ""))
        return None
