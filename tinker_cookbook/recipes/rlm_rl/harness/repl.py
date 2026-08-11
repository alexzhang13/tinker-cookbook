"""Persistent Python REPL for model-written code, bounded in memory and wall-clock.

Runs in a child interpreter (``repl_worker.py``) under ``RLIMIT_AS`` and is killed if it
exceeds its compute budget: model code can allocate without bound or loop forever, and
Python cannot kill a thread, so an in-process REPL has no recovery path. The budget charges
only child compute -- sub-LLM latency serviced by the driver is excluded.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import pickle
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_WORKER = Path(__file__).parent / "repl_worker.py"

logger = logging.getLogger(__name__)

DEFAULT_MEM_LIMIT_BYTES = 4 * 1024**3
DEFAULT_COMPUTE_TIMEOUT_S = 60.0


@dataclass
class ExecResult:
    stdout: str
    stderr: str


FRAME_MAGIC = b"RPL1"


class ProtocolError(Exception):
    """The worker pipe lost frame alignment; the REPL must be restarted."""


def _write_all(stream: Any, data: bytes) -> None:
    """Write every byte. The unbuffered pipe accepts short writes and would silently truncate
    any frame over ~64KB, desynchronising the stream permanently."""
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if not written:
            raise BrokenPipeError("REPL worker stopped accepting input")
        view = view[written:]


def _read_exactly(stream: Any, n: int) -> bytes:
    """Read exactly n bytes. The unbuffered pipe returns short reads on frames over ~64KB,
    which a single read would misreport as a dead worker."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("REPL worker closed the pipe mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class PythonRepl:
    def __init__(
        self,
        variables: dict[str, Any],
        async_bindings: dict[str, Callable[..., Coroutine[Any, Any, Any]]] | None = None,
        *,
        mem_limit_bytes: int = DEFAULT_MEM_LIMIT_BYTES,
        compute_timeout_s: float = DEFAULT_COMPUTE_TIMEOUT_S,
    ):
        self._variables = dict(variables)
        self._bindings = dict(async_bindings or {})
        self._mem_limit_bytes = mem_limit_bytes
        self._compute_timeout_s = compute_timeout_s
        self._proc: subprocess.Popen[bytes] | None = None
        # Dedicated thread, never the shared default executor: blocking pipe reads wait on
        # sub-call futures for arbitrarily long and would starve other coroutines.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._final_answer: str | None = None
        self._reset_pending = False
        self._closed = False

    # ---- framing (runs on the dedicated thread) ----

    def _send_sync(self, obj: Any) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise BrokenPipeError("REPL worker is not running")
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        _write_all(proc.stdin, FRAME_MAGIC + struct.pack(">I", len(payload)) + payload)
        proc.stdin.flush()

    def _recv_sync(self) -> Any:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise EOFError("REPL worker is not running")
        header = _read_exactly(proc.stdout, len(FRAME_MAGIC) + 4)
        if header[: len(FRAME_MAGIC)] != FRAME_MAGIC:
            raise ProtocolError(f"desynchronised frame (magic={header[:4]!r})")
        (n,) = struct.unpack(">I", header[len(FRAME_MAGIC) :])
        return pickle.loads(_read_exactly(proc.stdout, n))

    def _start_sync(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, str(_WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._send_sync(("init", self._variables, tuple(self._bindings), self._mem_limit_bytes))
        msg = self._recv_sync()
        if msg is None or msg[0] != "ready":
            raise RuntimeError(f"REPL worker failed to start: {msg!r}")

    # ---- lifecycle ----

    def _teardown(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        with contextlib.suppress(Exception):
            proc.kill()
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        for stream in (proc.stdin, proc.stdout):
            with contextlib.suppress(Exception):
                if stream is not None:
                    stream.close()

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    async def execute(self, code: str) -> ExecResult:
        if self._closed:
            return ExecResult("", "REPLClosed: the REPL is no longer available.")

        note = ""
        if self._reset_pending:
            note = (
                "[note] The REPL namespace was reset after the previous block was terminated. "
                "Variables you defined earlier are gone; `context` has been restored.\n"
            )
            self._reset_pending = False

        try:
            if self._proc is None or self._proc.poll() is not None:
                self._teardown()
                await self._run(self._start_sync)
            await self._run(self._send_sync, ("exec", code))
            result = await self._pump()
        except TimeoutError:
            self._teardown()
            self._reset_pending = True
            return ExecResult(
                "",
                note + f"REPLTimeout: this block used more than {self._compute_timeout_s:.0f}s of "
                "compute and was terminated. Work on a slice of the data instead of the whole "
                "context, and avoid unbounded loops.",
            )
        except BaseException as exc:
            # Worker death, desynchronised frame, unpicklable value: reset this REPL rather
            # than aborting the run. CancelledError still propagates, after cleanup.
            self._teardown()
            self._reset_pending = True
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.warning("REPL reset after %s: %s", type(exc).__name__, exc)
            return ExecResult(
                "",
                note + f"REPLReset: the REPL was restarted after an internal failure "
                f"({type(exc).__name__}); its {self._mem_limit_bytes // 1024**3}GB memory limit "
                "is the usual cause. Process the context in slices rather than materialising it "
                "all at once, and avoid calls that terminate the interpreter.",
            )

        stderr = note + result.stderr if (note or result.stderr) else ""
        return ExecResult(result.stdout, stderr)

    async def _pump(self) -> ExecResult:
        """Service the child until the block finishes, charging only child compute to the budget."""
        spent = 0.0
        while True:
            remaining = self._compute_timeout_s - spent
            if remaining <= 0:
                raise TimeoutError
            started = time.monotonic()
            recv = asyncio.ensure_future(self._run(self._recv_sync))
            try:
                msg = await asyncio.wait_for(asyncio.shield(recv), timeout=remaining)
            except TimeoutError:
                # The read unblocks when the worker is killed; consume its exception so it
                # does not surface as an unretrieved task error.
                recv.add_done_callback(lambda f: f.cancelled() or f.exception())
                raise
            spent += time.monotonic() - started

            kind = msg[0]
            if kind == "exec_done":
                _, stdout, stderr, answer = msg
                self._final_answer = answer
                return ExecResult(stdout, stderr)
            if kind == "call":
                # Sub-LLM latency is not child compute, so it is excluded from `spent`.
                _, name, args, kwargs = msg
                fn = self._bindings.get(name)
                try:
                    if fn is None:
                        raise RuntimeError(f"unknown binding {name!r}")
                    value = await fn(*args, **kwargs)
                    await self._run(self._send_sync, ("call_result", True, value))
                except Exception as exc:
                    await self._run(
                        self._send_sync, ("call_result", False, f"{type(exc).__name__}: {exc}")
                    )

    def close(self) -> None:
        self._closed = True
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(Exception):
                self._send_sync(("shutdown",))
        self._teardown()
        self._executor.shutdown(wait=False)

    @property
    def final_answer(self) -> str | None:
        return self._final_answer
