"""Helpers for parsing Python REPL calls in the RLM."""

from __future__ import annotations

import re

from tinker_cookbook.recipes.rlm_rl.rlm.repl.client import ExecResult
from tinker_cookbook.renderers.base import Message

MAX_REPL_OUTPUT_CHARS = 20_000
_REPL_BLOCK = re.compile(r"```repl\s*\n(.*?)\n```", re.DOTALL)


def find_repl_blocks(text: str) -> list[str]:
    return [m.strip() for m in _REPL_BLOCK.findall(text)]


def format_repl_outputs(
    outputs: list[ExecResult], max_chars: int = MAX_REPL_OUTPUT_CHARS
) -> Message:
    """Truncate REPL output to a fixed length so it cannot overflow the RLM's context."""
    parts = []
    for i, o in enumerate(outputs):
        body = "\n\n".join(s for s in (o.stdout.rstrip(), o.stderr.rstrip()) if s) or "No output"
        if len(body) > max_chars:
            body = body[:max_chars] + f"... + [{len(body) - max_chars} chars...]"
        header = f"REPL output (block {i + 1}):" if len(outputs) > 1 else "REPL output:"
        parts.append(f"{header}\n{body}")
    return {"role": "user", "content": "\n\n".join(parts)}
