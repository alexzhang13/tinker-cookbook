"""
Defining an environment that implements Recursive Agent Optimization (RAO) over a harness.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import tinker
import torch

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.rlm_rl.harness.base import Harness, Prompt
from tinker_cookbook.recipes.rlm_rl.rlm.tools import current_token_completer
from tinker_cookbook.recipes.rlm_rl.rlm_harness import RLMHarness
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, Trajectory, TrajectoryGroup

GradeFn = Callable[[str | None], Awaitable[float]]
SubGradeFn = Callable[[RLMHarness], Awaitable[float]]

_INSTALLED = False


def _iter_harnesses(harness: RLMHarness) -> list[RLMHarness]:
    out = [harness]
    for child in harness.tools.child_agents:
        out.extend(_iter_harnesses(child))
    return out


class HarnessEnv(MessageEnv):
    def __init__(
        self,
        harness: Harness,
        grade: GradeFn,
        sub_reward_lambda: float = 0.0,
        sub_grade: SubGradeFn | None = None,
    ):
        self.harness = harness
        self.grade = grade
        self.sub_reward_lambda = sub_reward_lambda
        self.sub_grade = sub_grade
        self.extra_trajectories: list[Trajectory] = []

    async def initial_observation(self) -> list[Message]:
        turn = await self.harness.start()
        if not isinstance(turn, Prompt):
            raise RuntimeError("harness ended before sampling any messages")
        return turn.messages

    async def step(self, message: Message) -> MessageStepResult:
        turn = await self.harness.act(message)
        if isinstance(turn, Prompt):
            return MessageStepResult(reward=0.0, episode_done=False, next_messages=turn.messages)
        score = await self.grade(turn.final_answer)
        reward = score
        metrics: dict[str, float] = {
            "correct": score,
            "answered": float(turn.final_answer is not None),
            **turn.metrics,
        }
        if self.sub_reward_lambda != 0.0 and isinstance(self.harness, RLMHarness):
            reward, extras = await self._rao_finish(score)
            self.extra_trajectories = extras
            metrics["rao/is_root"] = 1.0
            metrics["rao/depth"] = float(self.harness.depth)
            metrics["rao/local_reward"] = reward
        return MessageStepResult(
            reward=reward,
            episode_done=True,
            next_messages=[],
            metrics=metrics,
            logs={"final_answer": turn.final_answer or "<no answer submitted>"},
        )

    async def _rao_finish(self, root_score: float) -> tuple[float, list[Trajectory]]:
        assert isinstance(self.harness, RLMHarness)
        nodes = _iter_harnesses(self.harness)
        scores: list[float] = [root_score]

        async def _child_score(harness: RLMHarness) -> float:
            if self.sub_grade is None:
                return 0.0
            return await self.sub_grade(harness)

        if len(nodes) > 1:
            scores.extend(await asyncio.gather(*(_child_score(n) for n in nodes[1:])))
        score_of = dict(zip(nodes, scores, strict=True))
        extras: list[Trajectory] = []
        root_reward = root_score
        for harness in nodes:
            children = harness.tools.child_agents
            bonus = sum(score_of[c] for c in children) / len(children) if children else 0.0
            local = score_of[harness] + self.sub_reward_lambda * bonus
            if harness is self.harness:
                root_reward = local
                continue
            if not harness.transitions:
                continue
            last = harness.transitions[-1]
            last.reward = local
            last.episode_done = True
            last.metrics = {
                **last.metrics,
                "rao/is_root": 0.0,
                "rao/depth": float(harness.depth),
                "rao/local_reward": local,
            }
            extras.append(
                Trajectory(
                    transitions=list(harness.transitions),
                    final_ob=tinker.ModelInput.empty(),
                )
            )
        return root_reward, extras


def expand_rao_trajectories(
    trajectory_group: list[Trajectory], env_group: Sequence[Env]
) -> list[tuple[float, dict[str, float]]]:
    extras: list[Trajectory] = []
    for i, (traj, env) in enumerate(zip(trajectory_group, env_group, strict=True)):
        harness_env = getattr(env, "message_env", None)
        if getattr(harness_env, "sub_reward_lambda", 0.0) == 0.0:
            continue
        if traj.transitions:
            traj.transitions[-1].metrics["rao/tree_idx"] = float(i)
        for child_traj in getattr(harness_env, "extra_trajectories", None) or []:
            if child_traj.transitions:
                child_traj.transitions[-1].metrics["rao/tree_idx"] = float(i)
            extras.append(child_traj)
    trajectory_group.extend(extras)
    return [(0.0, {}) for _ in trajectory_group]


def _is_rao_group(group: TrajectoryGroup) -> bool:
    if not group.trajectories_G:
        return False
    traj = group.trajectories_G[0]
    return bool(traj.transitions) and "rao/tree_idx" in traj.transitions[-1].metrics


def rao_advantages(group: TrajectoryGroup) -> torch.Tensor:
    trajs = group.trajectories_G
    rewards = group.get_total_rewards()
    tree_ids: list[int] = []
    depths: list[int] = []
    is_root: list[bool] = []
    for traj in trajs:
        metrics = traj.transitions[-1].metrics if traj.transitions else {}
        tree_ids.append(int(metrics.get("rao/tree_idx", 0)))
        depths.append(int(metrics.get("rao/depth", 0)))
        is_root.append(float(metrics.get("rao/is_root", 0.0)) >= 0.5)

    root_reward_by_tree: dict[int, float] = {}
    for reward, tree_id, root in zip(rewards, tree_ids, is_root, strict=True):
        if root:
            root_reward_by_tree[tree_id] = reward

    trees = list(root_reward_by_tree)
    n_trees = len(trees)
    total = sum(root_reward_by_tree.values())
    baselines: dict[int, float] = {}
    for tree_id in trees:
        if n_trees > 1:
            baselines[tree_id] = (total - root_reward_by_tree[tree_id]) / (n_trees - 1)
        else:
            baselines[tree_id] = root_reward_by_tree[tree_id]

    advantages = [
        reward - baselines.get(tree_id, 0.0) for reward, tree_id in zip(rewards, tree_ids)
    ]
    n_traj = len(trajs)
    counts = Counter(depths)
    d_used = len(counts) or 1
    alpha = n_traj / d_used
    weights = [alpha / counts[depth] for depth in depths]
    return torch.tensor([a * w for a, w in zip(advantages, weights)], dtype=torch.float32)


def install_rao_training() -> None:
    """Hook the cookbook loop for RAO: expose the on-policy completer to child rollouts, and use root-LOO + depth-weighted advantages instead of GRPO."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from tinker_cookbook.rl import train as rl_train

    orig_call = TinkerTokenCompleter.__call__

    async def patched_call(
        self: TinkerTokenCompleter,
        model_input: tinker.ModelInput,
        stop: Any,
        *,
        max_tokens: int | None = None,
    ) -> Any:
        current_token_completer.set(self)
        return await orig_call(self, model_input, stop, max_tokens=max_tokens)

    TinkerTokenCompleter.__call__ = patched_call

    orig_adv = rl_train.compute_advantages

    def patched_adv(trajectory_groups_P: list[TrajectoryGroup]) -> list[torch.Tensor]:
        return [
            rao_advantages(group) if _is_rao_group(group) else orig_adv([group])[0]
            for group in trajectory_groups_P
        ]

    rl_train.compute_advantages = patched_adv


class RepeatingRLDataset(RLDataset):
    """Repeats a small problem pool for `n_batches` steps, reshuffling each epoch."""

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
