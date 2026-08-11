"""Recursive Agent Optimization (RAO) glue between a Harness and the RL loop.

`sub_reward_lambda` controls how much reward sub-agent transitions receive: at 0 they are
loss-masked and only the root chain trains. lambda > 0 is not implemented yet.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable, Sequence

from tinker_cookbook.recipes.rlm_rl.harness.base import Harness
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import PARSE_ERROR_MASKED_METRIC_KEY, EnvGroupBuilder, RLDataset

GradeFn = Callable[[str | None], Awaitable[float]]

SUBAGENT_METRIC_KEY = "subagent_turn"


class HarnessEnv(MessageEnv):
    def __init__(self, harness: Harness, grade: GradeFn, sub_reward_lambda: float = 0.0):
        if sub_reward_lambda != 0.0:
            raise NotImplementedError("only sub_reward_lambda=0 (root-only credit) is implemented")
        self.harness = harness
        self.grade = grade
        self._expect_subagent = False

    async def initial_observation(self) -> list[Message]:
        return await self.harness.initial_messages()

    async def step(self, message: Message) -> MessageStepResult:
        metrics: dict[str, float] = {}
        if self._expect_subagent:
            # PARSE_ERROR_MASKED_METRIC_KEY is the loss-mask channel trajectory_to_data
            # honors; here it marks a sub-agent turn.
            metrics = {SUBAGENT_METRIC_KEY: 1.0, PARSE_ERROR_MASKED_METRIC_KEY: 1.0}
        step = await self.harness.step(message)
        self._expect_subagent = step.subagent
        if not step.done:
            return MessageStepResult(
                reward=0.0, episode_done=False, next_messages=step.messages, metrics=metrics
            )
        reward = await self.grade(step.final_answer)
        return MessageStepResult(
            reward=reward,
            episode_done=True,
            next_messages=[],
            metrics={
                "correct": reward,
                "answered": float(step.final_answer is not None),
                **step.metrics,
                **metrics,
            },
            logs={"final_answer": step.final_answer or "<no answer submitted>"},
        )


class CyclingRLDataset(RLDataset):
    """Cycles a small builder pool for n_batches batches, reshuffling every epoch."""

    def __init__(
        self,
        env_group_builders_P: Sequence[EnvGroupBuilder],
        batch_size: int,
        n_batches: int,
        seed: int = 0,
    ):
        self.env_group_builders_P = list(env_group_builders_P)
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.seed = seed

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        n_problems = len(self.env_group_builders_P)
        epoch, offset = divmod(index * self.batch_size, n_problems)
        order_P = list(range(n_problems))
        random.Random(self.seed + epoch).shuffle(order_P)
        return [
            self.env_group_builders_P[order_P[(offset + i) % n_problems]]
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return self.n_batches
