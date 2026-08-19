"""Bounded REPL worker: length-prefixed pickle frames over stdin/stdout.

Launched as a script path, never imported: the child needs a clean address space to put an
RLIMIT_AS on, and importing ``tinker_cookbook`` would pull torch into every REPL. Stdlib
plus the sibling ``_protocol.py`` only (both stdlib underneath). Sub-LLM bindings appear as
ordinary sync functions that RPC back to the driver and block.
"""

from __future__ import annotations

import contextlib
import io
import os
import resource
import sys
import traceback
from typing import Any

import _protocol

OUTPUT_LIMIT_CHARS = 262_144


class _BoundedWriter(io.TextIOBase):
    """stdout/stderr sink that stops accumulating past a cap.

    Bounding at write time, not afterwards: `print(context)` would otherwise be copied again
    by `getvalue()` and by the pickle, blowing RLIMIT_AS and killing the worker mid-reply.
    """

    def __init__(self, limit: int = OUTPUT_LIMIT_CHARS):
        self._limit = limit
        self._parts: list[str] = []
        self._len = 0
        self.dropped = 0

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        n = len(s)
        room = self._limit - self._len
        if room > 0:
            chunk = s[:room]
            self._parts.append(chunk)
            self._len += len(chunk)
            self.dropped += n - len(chunk)
        else:
            self.dropped += n
        return n

    def getvalue(self) -> str:
        text = "".join(self._parts)
        if self.dropped:
            text += f"\n... [{self.dropped} more characters suppressed]"
        return text


def _make_stub(inp: Any, out: Any, name: str):
    def stub(*args: Any, **kwargs: Any) -> Any:
        _protocol.send(out, ("call", name, args, kwargs))
        resp = _protocol.recv(inp)
        if resp is None or resp[0] != "call_result":
            raise RuntimeError(f"{name}: lost connection to the driver")
        ok, value = resp[1], resp[2]
        if not ok:
            raise RuntimeError(value)
        return value

    stub.__name__ = name
    return stub


def _read_answer(ns: dict[str, Any]) -> str | None:
    answer = ns.get("answer")
    if isinstance(answer, dict) and answer.get("ready"):
        return str(answer.get("content", ""))
    return None


def _die_with_parent() -> None:
    """SIGKILL this worker if the driver goes away. Done post-exec rather than in a Popen
    preexec_fn, which would run between fork and exec in a heavily threaded parent."""
    try:
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def main() -> int:
    # The protocol owns fds 0/1. Point the real stdout/stderr at /dev/null so a stray
    # C-level write from model code cannot corrupt a frame.
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)

    _die_with_parent()

    init = _protocol.recv(proto_in)
    if init is None or init[0] != "init":
        return 1
    _, variables, binding_names, mem_limit_bytes = init

    if mem_limit_bytes:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))

    ns: dict[str, Any] = {"__name__": "__main__", "answer": {"content": "", "ready": False}}
    ns.update(variables)
    ns["SHOW_VARS"] = lambda: ", ".join(k for k in ns if not k.startswith("_"))
    for name in binding_names:
        ns[name] = _make_stub(proto_in, proto_out, name)

    _protocol.send(proto_out, ("ready",))

    while True:
        msg = _protocol.recv(proto_in)
        if msg is None or msg[0] == "shutdown":
            return 0
        if msg[0] != "exec":
            continue
        out, err = _BoundedWriter(), _BoundedWriter()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(msg[1], "<repl>", "exec"), ns)
        except BaseException:
            # MemoryError from the rlimit, RecursionError, and a model calling sys.exit() all
            # have to come back as REPL output rather than kill the worker.
            try:
                err.write(traceback.format_exc(limit=3))
            except BaseException:
                err.write("error while formatting traceback")
        try:
            payload = ("exec_done", out.getvalue(), err.getvalue(), _read_answer(ns))
        except BaseException:
            payload = ("exec_done", "", "REPL output too large to return", None)
        try:
            _protocol.send(proto_out, payload)
        except BaseException:
            return 1


if __name__ == "__main__":
    sys.exit(main())
