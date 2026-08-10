from __future__ import annotations

import functools
from pathlib import Path

import jinja2

from tinker_cookbook.renderers.base import Message

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@functools.cache
def rlm_system_prompt(custom_tools_section: str = "") -> str:
    template = jinja2.Template((_PROMPTS_DIR / "rlm_system_prompt.jinja").read_text())
    return template.render(custom_tools_section=custom_tools_section).strip()


def system_messages(
    *,
    context_chars: int,
    context_type: str,
    root_prompt: str | None,
) -> list[Message]:
    system = rlm_system_prompt()
    metadata = (
        f"Your context is a {context_type} of {context_chars} total characters."
        " Each sub-LLM call can handle roughly ~100k tokens at once."
    )
    if root_prompt:
        metadata = f"Answer the following: {root_prompt}\n\n{metadata}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": metadata},
    ]


def turn_prompt(iteration: int, max_iterations: int) -> Message:
    body = f"Turn {iteration + 1}/{max_iterations}:"
    if iteration == 0:
        body = (
            "You have not interacted with the REPL environment or seen your "
            "prompt / context yet. Look at the context first; do not provide "
            "a final answer yet.\n\n" + body
        )
    return {"role": "user", "content": body}
