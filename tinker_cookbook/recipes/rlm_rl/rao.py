"""
Recursive Agent Optimization (RAO) over a harness of recursive sub-agents (RLMs).
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import tinker
import torch

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.rlm_rl.harness.base import Harness, Prompt
from tinker_cookbook.recipes.rlm_rl.harness.repl import REPLHarness
from tinker_cookbook.recipes.rlm_rl.rlm.tools import current_token_completer
from tinker_cookbook.recipes.rlm_rl.rlm_harness import RLMHarness
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.message_env import MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, Trajectory, TrajectoryGroup

# `\tilde{s}` for the root (from its final answer) and for a sub-agent (from its whole
# trajectory, since a sub-task usually has no verifier).
RootGrader = Callable[[str | None], Awaitable[float]]
NodeGrader = Callable[[RLMHarness], Awaitable[float]]

# Tree structure travels from rollout to advantage computation as trajectory metrics.
# IS_ROOT is also the marker for "this node was graded": a trajectory without it is not
# part of any tree RAO trains on (see `RAOHarnessEnv.step`).
IS_ROOT = "rao/is_root"
DEPTH = "rao/depth"
LOCAL_REWARD = "rao/local_reward"
TREE_IDX = "rao/tree_idx"


class RAOHarnessEnv(MessageEnv):
    """One RAO rollout tree, presented to the RL loop as a single-agent environment.

    The RL loop samples the root agent's turns; sub-agent turns are sampled underneath,
    inside the harness, from the same policy (see `RLMTools`). When the harness finishes,
    `step` scores every node in the tree with Eq. 1 and hands the sub-agents' trajectories
    to `expand_rao_trajectories` via `extra_trajectories`.

    On discarded trees: `step` only runs while the rollout runner is still stepping this env.
    If the runner ends the rollout itself -- the sampler hit `max_tokens`, the response
    failed to parse, or the conversation outgrew the context window -- it never calls
    `step`, so the tree is never graded and *the whole tree is thrown out*: sub-agent turns
    already sampled are dropped, and the root trajectory carries no `IS_ROOT`, which keeps
    it out of the Eq. 3 baseline and gives it zero advantage. This costs the sub-agent
    samples from those rollouts, and is the intended behaviour: a tree whose root never
    submitted an answer has no `\\tilde{s}(root)` to build Eq. 1 from.

    At `sub_reward_lambda == 0` this is plain root-only training: no node is graded but the
    root, no extra trajectories are produced, and Eq. 3 reduces to a standard leave-one-out
    baseline over the group.
    """

    def __init__(
        self,
        harness: Harness,
        grade: RootGrader,
        sub_reward_lambda: float = 0.0,
        sub_grade: NodeGrader | None = None,
    ):
        self.harness = harness
        self.grade = grade
        self.sub_reward_lambda = sub_reward_lambda
        self.sub_grade = sub_grade
        # Read by `expand_rao_trajectories` after the rollout; empty unless this tree was
        # graded and had sub-agents.
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

        # The root agent finished: grade it, then the rest of its tree.
        root_success = await self.grade(turn.final_answer)
        reward = root_success
        metrics: dict[str, float] = {
            "correct": root_success,
            "answered": float(turn.final_answer is not None),
            **turn.metrics,
        }

        # If lambda is non-zero, begin adding local rewards to sub-agents' trajectories.
        if self._grades_sub_agents():
            reward, self.extra_trajectories = await self._score_tree(root_success)
            metrics |= {IS_ROOT: 1.0, DEPTH: float(self.harness.depth), LOCAL_REWARD: reward}

        return MessageStepResult(
            reward=reward,
            episode_done=True,
            next_messages=[],
            metrics=metrics,
            logs={"final_answer": turn.final_answer or "<no answer submitted>"},
        )

    async def observe_truncated_response(self, message: Message) -> list[Message] | None:
        """Keep a turn the sampler cut off at `max_tokens` so the tree survives it.

        Paired with `terminate_on_length=False` on the adapter. Without both, one over-long
        turn discards the whole tree, and the loss is not evenly spread: a policy that
        delegates more writes longer code blocks, so the discard rate climbs with training
        and quietly censors the metrics (it reached 43% of rollouts in one 42-step run)."""
        if not isinstance(self.harness, REPLHarness):
            return None
        return self.harness.observe_truncated(message)

    def _grades_sub_agents(self) -> bool:
        return self.sub_reward_lambda != 0.0 and isinstance(self.harness, RLMHarness)

    async def _score_tree(self, root_success: float) -> tuple[float, list[Trajectory]]:
        r"""Apply Eq. 1 to every node; return the root's reward and the sub-agents' trajectories.

            R(X) = \tilde{s}(X) + \lambda * mean_{c in C(X)} \tilde{s}(c)

        The bonus uses the children's *mean* success, not their count, so spawning more
        children is not itself rewarded. A node's reward lands on the last transition of
        its own trajectory, which is where `rao_advantages` reads it.
        """
        assert isinstance(self.harness, RLMHarness)
        tree = _walk_tree(self.harness)
        nodes = [node for node, _children in tree]
        successes = [root_success, *await self._grade_sub_agents(nodes[1:])]
        success_of = dict(zip(nodes, successes, strict=True))

        root_reward = root_success
        sub_agent_trajectories: list[Trajectory] = []
        for node, children in tree:
            bonus = sum(success_of[c] for c in children) / len(children) if children else 0.0
            reward = success_of[node] + self.sub_reward_lambda * bonus
            if node is self.harness:
                root_reward = reward
                continue
            if not node.transitions:
                # Nothing was sampled for this node (e.g. its own sub-call raised), so
                # there is no trajectory to train even though it still fed its parent's bonus.
                continue
            last = node.transitions[-1]
            last.reward = reward
            last.episode_done = True
            last.metrics = {
                **last.metrics,
                IS_ROOT: 0.0,
                DEPTH: float(node.depth),
                LOCAL_REWARD: reward,
            }
            sub_agent_trajectories.append(
                Trajectory(transitions=list(node.transitions), final_ob=tinker.ModelInput.empty())
            )
        return root_reward, sub_agent_trajectories

    async def _grade_sub_agents(self, nodes: Sequence[RLMHarness]) -> list[float]:
        if not nodes:
            return []
        grade = self.sub_grade
        if grade is None:
            return [0.0] * len(nodes)
        return list(await asyncio.gather(*(grade(node) for node in nodes)))


def expand_rao_trajectories(
    trajectory_group: list[Trajectory], env_group: Sequence[Env]
) -> list[tuple[float, dict[str, float]]]:
    """Add each tree's sub-agent trajectories to the group, tagged with their tree.

    Call from `EnvGroupBuilder.compute_group_rewards`, which is the one hook that sees a
    whole group at once. `trajectory_group` is extended in place: it arrives holding one
    root trajectory per rollout and leaves holding every trained node in the group. The
    returned rewards are all zero because node rewards already live on the transitions.

    Trees the runner discarded contribute nothing here (see `RAOHarnessEnv`); their root
    trajectory is still tagged with `TREE_IDX` so `rao_advantages` can tell it apart from a
    trajectory that was never part of a tree at all.
    """
    sub_agent_trajectories: list[Trajectory] = []
    for tree_idx, (traj, env) in enumerate(zip(trajectory_group, env_group, strict=True)):
        rao_env = _rao_env(env)
        if rao_env is None:
            continue
        if traj.transitions and "correct" not in traj.transitions[-1].metrics:
            # The runner ended this rollout before the harness could be graded, so `step`
            # never emitted `correct`. Scoring it 0 keeps the denominator fixed: a rollout
            # that never answered is a failure, not an absence. It still has no IS_ROOT, so
            # it stays out of the Eq. 3 baseline and earns no gradient.
            traj.transitions[-1].metrics |= {"correct": 0.0, "answered": 0.0, "discarded": 1.0}
        if rao_env.sub_reward_lambda == 0.0:
            continue
        for node_traj in [traj, *rao_env.extra_trajectories]:
            if node_traj.transitions:
                node_traj.transitions[-1].metrics[TREE_IDX] = float(tree_idx)
        sub_agent_trajectories.extend(rao_env.extra_trajectories)
    trajectory_group.extend(sub_agent_trajectories)
    return [(0.0, {}) for _ in trajectory_group]


def rao_advantages(group: TrajectoryGroup) -> torch.Tensor:
    r"""Advantage per trajectory: a root-only leave-one-out baseline, weighted by depth.

        Eq. 3:  A(\tau^{(g)}) = R(\tau^{(g)}) - b_{-g},
                b_{-g} = 1/(G-1) * \sum_{g' \neq g} R^{(g')}_{root}
        Eq. 4:  w_d = \alpha / N_d,  \alpha = |B| / D

    Every node is centered on the mean reward of the *other* rollouts' roots, then scaled
    by its depth's inverse frequency. Discarded trees (`IS_ROOT` absent) get advantage 0
    and are excluded from both the baseline and the depth counts, so they neither train nor
    shift anyone else's baseline.
    """
    rewards = group.get_total_rewards()
    nodes = [_NodeInfo.from_trajectory(traj) for traj in group.trajectories_G]

    # Eq. 3: leave-one-out over root rewards. With one root left there is nothing to leave
    # out, so the baseline is that root's own reward and the group contributes no gradient.
    root_reward_by_tree = {
        node.tree_idx: reward for node, reward in zip(nodes, rewards, strict=True) if node.is_root
    }
    total = sum(root_reward_by_tree.values())
    n_trees = len(root_reward_by_tree)
    baselines = {
        tree: (total - reward) / (n_trees - 1) if n_trees > 1 else reward
        for tree, reward in root_reward_by_tree.items()
    }

    # Eq. 4: inverse frequency over the depths actually being trained.
    trained_depths = [node.depth for node in nodes if node.trained]
    per_depth = Counter(trained_depths)
    alpha = len(trained_depths) / len(per_depth) if per_depth else 0.0

    def advantage(node: _NodeInfo, reward: float) -> float:
        if not node.trained or node.tree_idx not in baselines:
            return 0.0  # discarded tree: no gradient from it
        return (reward - baselines[node.tree_idx]) * (alpha / per_depth[node.depth])

    return torch.tensor(
        [advantage(node, reward) for node, reward in zip(nodes, rewards, strict=True)],
        dtype=torch.float32,
    )


def install_rao_training() -> None:
    """Hook the cookbook RL loop for RAO. Idempotent, and a no-op for non-RAO groups.

    Two hooks: expose the training policy to the harness so sub-agent turns are sampled
    on-policy (rather than from a frozen sampling client), and swap GRPO's group centering
    for `rao_advantages` on groups that carry a tree.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from tinker_cookbook.rl import train as rl_train

    # The harness reaches the policy through a ContextVar: it runs deep inside `Env.step`,
    # far from where the loop constructs the completer.
    original_call = TinkerTokenCompleter.__call__

    async def call_and_publish_policy(
        self: TinkerTokenCompleter,
        model_input: tinker.ModelInput,
        stop: Any,
        *,
        max_tokens: int | None = None,
    ) -> Any:
        current_token_completer.set(self)
        return await original_call(self, model_input, stop, max_tokens=max_tokens)

    TinkerTokenCompleter.__call__ = call_and_publish_policy

    grpo_advantages = rl_train.compute_advantages

    def advantages_per_group(trajectory_groups_P: list[TrajectoryGroup]) -> list[torch.Tensor]:
        return [
            rao_advantages(group) if _is_rao_group(group) else grpo_advantages([group])[0]
            for group in trajectory_groups_P
        ]

    rl_train.compute_advantages = advantages_per_group


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_INSTALLED = False


@dataclass(frozen=True)
class _NodeInfo:
    """Where one trajectory sits in its rollout tree, read back from its metrics."""

    trained: bool
    """Graded by `_score_tree`. False means the tree was discarded before grading."""
    is_root: bool
    depth: int
    tree_idx: int

    @classmethod
    def from_trajectory(cls, traj: Trajectory) -> _NodeInfo:
        metrics = traj.transitions[-1].metrics if traj.transitions else {}
        return cls(
            trained=IS_ROOT in metrics,
            is_root=float(metrics.get(IS_ROOT, 0.0)) >= 0.5,
            depth=int(metrics.get(DEPTH, 0)),
            tree_idx=int(metrics.get(TREE_IDX, -1)),
        )


def _walk_tree(root: RLMHarness) -> list[tuple[RLMHarness, list[RLMHarness]]]:
    """The tree flattened depth-first, root first, each node paired with the children it
    had at this instant.

    The pairing is the point: the tree can still be growing. A sub-call orphaned by a
    failed `*_batched` gather (one raises, its siblings are never cancelled) keeps running
    and appends itself to its parent's children when it finishes -- possibly while grading
    is awaiting judge verdicts. Scoring against a snapshot keeps every child that a node is
    credited for inside the set of nodes that were graded. Latecomers are left untrained,
    like any other node whose tree was not graded.
    """
    tree = [(root, list(root.children))]
    for child in tree[0][1]:
        tree.extend(_walk_tree(child))
    return tree


def _rao_env(env: Env) -> RAOHarnessEnv | None:
    """Unwrap the `RAOHarnessEnv` inside a token-level env, if there is one."""
    inner = getattr(env, "message_env", env)
    return inner if isinstance(inner, RAOHarnessEnv) else None


def _is_rao_group(group: TrajectoryGroup) -> bool:
    """Whether this group carries rollout trees, and so wants `rao_advantages`."""
    return any(
        traj.transitions and TREE_IDX in traj.transitions[-1].metrics
        for traj in group.trajectories_G
    )
