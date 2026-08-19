"""Length-prefixed pickle frames over BinaryIO.

Stdlib-only, shared by ``client.py`` and ``worker.py``. Kept next to the worker so that
``import _protocol`` resolves via ``sys.path[0]`` when the worker runs by script path.
"""

from __future__ import annotations

import pickle
import struct
from typing import Any, BinaryIO

FRAME_MAGIC = b"RPL1"
_HEADER_LEN = len(FRAME_MAGIC) + 4


class ProtocolError(Exception):
    """The pipe lost frame alignment; the peer must be restarted."""


def write_all(out: BinaryIO, data: bytes) -> None:
    """Write every byte. Unbuffered pipes accept short writes and would silently truncate
    any frame over ~64KB, desynchronising the stream permanently."""
    view = memoryview(data)
    while view:
        written = out.write(view)
        if not written:
            raise BrokenPipeError("peer stopped accepting input")
        view = view[written:]


def read_exactly(inp: BinaryIO, n: int) -> bytes | None:
    """Read exactly n bytes, or None if the pipe closes at a boundary (short reads are not EOF)."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = inp.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send(out: BinaryIO, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    write_all(out, FRAME_MAGIC + struct.pack(">I", len(payload)) + payload)
    out.flush()


def recv(inp: BinaryIO) -> Any | None:
    """Read one frame. Returns None on clean EOF at a frame boundary; raises ProtocolError
    on desync or EOF mid-frame."""
    header = read_exactly(inp, _HEADER_LEN)
    if header is None:
        return None
    if header[: len(FRAME_MAGIC)] != FRAME_MAGIC:
        raise ProtocolError(f"desynchronised frame (magic={header[: len(FRAME_MAGIC)]!r})")
    (n,) = struct.unpack(">I", header[len(FRAME_MAGIC) :])
    body = read_exactly(inp, n)
    if body is None:
        raise ProtocolError("EOF mid-frame")
    return pickle.loads(body)
