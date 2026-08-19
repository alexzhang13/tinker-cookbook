"""Generic sub-agent calling harness, ACP-shaped.

A `Harness` is a sample/end state machine: `start()` returns the first `Turn`,
each `act(assistant_message)` returns the next `Turn`. A `Prompt` asks the caller to
sample this conversation; an `End` finishes the session. Names mirror the
[Agent Client Protocol](https://agentclientprotocol.com/) so an ACP transport can wrap
this later; the loop remains in-process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from tinker_cookbook.renderers.base import Message

MessageSampler = Callable[[list[Message]], Awaitable[Message]]


@dataclass
class Prompt:
    """Messages the session wants sampled next."""

    messages: list[Message]


@dataclass
class End:
    """Session finished."""

    final_answer: str | None = None
    messages: list[Message] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    stop_reason: str = "end_turn"


Turn = Prompt | End


class Harness(ABC):
    @abstractmethod
    async def start(self) -> Turn: ...

    @abstractmethod
    async def act(self, assistant_message: Message) -> Turn: ...

    def close(self) -> None:
        pass

    async def run(self, completer: MessageSampler, max_turns: int = 200) -> End:
        turn = await self.start()
        try:
            for _ in range(max_turns):
                if isinstance(turn, End):
                    return turn
                turn = await self.act(await completer(turn.messages))
            return turn if isinstance(turn, End) else End(stop_reason="max_turns")
        finally:
            self.close()
