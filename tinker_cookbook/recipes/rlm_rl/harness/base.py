"""Generic sub-agent calling harness class.

A Harness owns everything between model samples: conversation structure,
tool execution, and delegation to sub-agents. Sub-agent turns surface as
HarnessStep(messages=..., subagent=True), so the same policy samples root
and sub-agent tokens alike — on-policy by construction.

Drive a harness with a fixed completer (`run`) or with the trained policy (`rao.HarnessEnv`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from tinker_cookbook.renderers.base import Message

MessageSampler = Callable[[list[Message]], Awaitable[Message]]


@dataclass
class HarnessStep:
    done: bool
    messages: list[Message]
    subagent: bool = False
    final_answer: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class Harness(ABC):
    @abstractmethod
    async def initial_messages(self) -> list[Message]: ...

    @abstractmethod
    async def step(self, assistant_message: Message) -> HarnessStep: ...

    def close(self) -> None:
        pass


async def run(harness: Harness, completer: MessageSampler, max_turns: int = 200) -> HarnessStep:
    step = HarnessStep(done=False, messages=await harness.initial_messages())
    try:
        for _ in range(max_turns):
            step = await harness.step(await completer(step.messages))
            if step.done:
                break
    finally:
        harness.close()
    return step
