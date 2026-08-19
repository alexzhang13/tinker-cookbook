from __future__ import annotations

import functools
from pathlib import Path

import jinja2

from tinker_cookbook.renderers.base import Message

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@functools.cache
def nudge_hint_text(max_sub_calls: int = 25) -> str:
    """The "nudge to decompose" addendum from:
    https://www.alphaxiv.org/abs/2607.language-model-harnesses
    """
    template = jinja2.Template((_PROMPTS_DIR / "nudge_hint.jinja").read_text())
    return template.render(max_sub_calls=max_sub_calls).strip()


@functools.cache
def rlm_system_prompt(
    custom_tools_section: str = "",
    max_sub_calls: int = 25,
    nudge_hint: bool = True,
) -> str:
    template = jinja2.Template((_PROMPTS_DIR / "rlm_system_prompt.jinja").read_text())
    prompt = template.render(
        custom_tools_section=custom_tools_section, max_sub_calls=max_sub_calls
    ).strip()
    if nudge_hint:
        prompt = f"{prompt}\n\n{nudge_hint_text(max_sub_calls)}"
    return prompt


def system_messages(
    *,
    context_chars: int,
    context_type: str,
    root_prompt: str | None,
    max_sub_calls: int = 25,
    nudge_hint: bool = True,
) -> list[Message]:
    system = rlm_system_prompt(max_sub_calls=max_sub_calls, nudge_hint=nudge_hint)
    metadata = (
        f"Your context is a {context_type} of {context_chars} total characters."
        " Each sub-LLM call can handle roughly ~100k characters at once."
    )
    if root_prompt:
        metadata = f"Answer the following: {root_prompt}\n\n{metadata}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": metadata},
    ]


@functools.cache
def judge_system_prompt(no_repl: bool = False) -> str:
    """Rubric for grading one sub-agent.

    `no_repl` selects the variant for a single-turn `llm_query` sub-agent: the default
    rubric only marks an agent successful if it read the context with its REPL or
    delegated, neither of which a single-turn agent can do."""
    template = jinja2.Template((_PROMPTS_DIR / "judge_system_prompt.jinja").read_text())
    return template.render(no_repl=no_repl).strip()


def turn_prompt(iteration: int, max_iterations: int) -> Message:
    body = f"Turn {iteration + 1}/{max_iterations}:"
    if iteration == 0:
        body = (
            "You have not interacted with the REPL environment or seen your "
            "prompt / context yet. Look at the context first; do not provide "
            "a final answer yet.\n\n" + body
        )
    return {"role": "user", "content": body}
