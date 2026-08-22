"""
OOLONG-Pairs: pairwise-aggregation over a long context, scored by set F1.
"""

from __future__ import annotations

import ast
import functools
import json
import re
from collections.abc import Sequence
from typing import Any

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.recipes.rlm_rl.rao import (
    RAOHarnessEnv,
    RepeatingRLDataset,
    expand_rao_trajectories,
)
from tinker_cookbook.recipes.rlm_rl.rlm.tools import current_renderer
from tinker_cookbook.recipes.rlm_rl.rlm_harness import RLMHarness
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, Trajectory

PAIRS_REPO = "mit-oasys/oolong-pairs"
SYNTH_REPO = "oolongbench/oolong-synth"

PAIRS_INSTRUCTION = (
    "The context contains thousands of general-knowledge questions, one per line, each tagged "
    "with a User ID. Each line implicitly belongs to one of 6 categories: 'numeric value', "
    "'entity', 'location', 'description and abstract concept', 'abbreviation', 'human being'. "
    "The labels are not given -- infer them from each question's semantics. Answer the following "
    "aggregate question about pairs of users."
)

# Kept byte-for-byte compatible with the length-gen reference implementation
# (rlm-minimal-training oolong_pairs_fake/env.py) so scores are comparable across setups:
# parenthesised pairs first, bare `a, b` / `a/b` / `a&b` only as a fallback, <think> blocks
# stripped before parsing, and self-pairs retained so they count against precision.
_PAIR_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_BARE_PAIR_RE = re.compile(r"\b(-?\d+)\s*[,/&]\s*(-?\d+)\b")
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)

Pair = tuple[int, int]


def _parse_pairs(text: str) -> set[Pair]:
    """Pull pairs out of free text, normalised so the lower ID comes first."""
    pairs: set[Pair] = set()
    matches = _PAIR_RE.findall(text or "") or _BARE_PAIR_RE.findall(text or "")
    for a, b in matches:
        ia, ib = int(a), int(b)
        pairs.add((ia, ib) if ia < ib else (ib, ia))
    return pairs


def pairs_f1(gold: set[Pair], output: str | None) -> float:
    """Set F1 between the gold pairs and the pairs found in the model's answer."""
    cleaned = _THINK_RE.sub("", output or "")
    pred = _parse_pairs(cleaned)
    if not gold and not pred:
        return 1.0
    if not pred or not gold:
        return 0.0
    correct = len(pred & gold)
    if correct == 0:
        return 0.0
    precision = correct / len(pred)
    recall = correct / len(gold)
    return 2 * precision * recall / (precision + recall)


@functools.cache
def load_pairs_context(context_len: int) -> str:
    """The shared trec_coarse context window for a context length (unlabelled)."""
    import datasets

    stream = datasets.load_dataset(SYNTH_REPO, split="validation", streaming=True)
    for ex in stream:
        if ex["dataset"] == "trec_coarse" and int(ex["context_len"]) == context_len:
            return ex["context_window_text"]
    raise ValueError(f"no trec_coarse context for context_len={context_len}")


def load_pairs(*, context_len: int, num_examples: int, seed: int = 0) -> list[dict[str, Any]]:
    """Questions + gold pair sets for one context length, paired with their shared context."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=PAIRS_REPO,
        filename=f"data/oolong-pairs-{context_len}.json",
        repo_type="dataset",
    )
    with open(path) as f:
        questions = json.load(f)
    context = load_pairs_context(context_len)

    rows: list[dict[str, Any]] = []
    for q in questions[:num_examples]:
        answer = q["answer"]
        if isinstance(answer, str):
            answer = ast.literal_eval(answer)
        gold = _parse_pairs(" ".join(str(p) for p in answer))
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "gold": gold,
                "context_window_text": context,
            }
        )
    if not rows:
        raise ValueError(f"no pairs questions for context_len={context_len}")
    return rows


class PairsEnvGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        *,
        row: dict[str, Any],
        model_name: str,
        renderer_name: str | None,
        group_size: int,
        depth: int,
        max_iterations: int,
        max_trajectory_tokens: int,
        sub_reward_lambda: float = 0.0,
        max_sub_calls: int = 50,
        child_max_iterations: int = 8,
        sub_completer: MessageCompleter | None = None,
        repl_mem_limit_bytes: int = 2 * 1024**3,
        repl_compute_timeout_s: float = 60.0,
        nudge_hint: bool = True,
    ):
        self.row = row
        self.model_name = model_name
        self.renderer_name = renderer_name
        self.group_size = group_size
        self.depth = depth
        self.max_iterations = max_iterations
        self.max_trajectory_tokens = max_trajectory_tokens
        self.sub_reward_lambda = sub_reward_lambda
        self.max_sub_calls = max_sub_calls
        self.child_max_iterations = child_max_iterations
        self.sub_completer = sub_completer
        self.repl_mem_limit_bytes = repl_mem_limit_bytes
        self.repl_compute_timeout_s = repl_compute_timeout_s
        self.nudge_hint = nudge_hint
        self._harnesses_G: list[RLMHarness] = []

    async def make_envs(self) -> Sequence[Env]:
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer_utils.get_tokenizer(self.model_name))
        current_renderer.set(renderer)
        gold: set[Pair] = self.row["gold"]

        async def grade(final_answer: str | None) -> float:
            return pairs_f1(gold, final_answer)

        self._harnesses_G = [
            RLMHarness(
                context=self.row["context_window_text"],
                root_prompt=f"{PAIRS_INSTRUCTION}\n\nQuestion: {self.row['question']}",
                max_iterations=self.max_iterations,
                max_depth=self.depth,
                child_max_iterations=self.child_max_iterations,
                max_sub_calls=self.max_sub_calls,
                sub_completer=self.sub_completer,
                mem_limit_bytes=self.repl_mem_limit_bytes,
                compute_timeout_s=self.repl_compute_timeout_s,
                nudge_hint=self.nudge_hint,
            )
            for _ in range(self.group_size)
        ]
        return [
            EnvFromMessageEnv(
                renderer=renderer,
                message_env=RAOHarnessEnv(h, grade, sub_reward_lambda=self.sub_reward_lambda),
                failed_parse_reward=0.0,
                context_overflow_reward=0.0,
                max_trajectory_tokens=self.max_trajectory_tokens,
                # Keep a turn the sampler clipped rather than discarding the whole tree.
                terminate_on_length=False,
            )
            for h in self._harnesses_G
        ]

    async def cleanup(self) -> None:
        for h in self._harnesses_G:
            h.close()
        self._harnesses_G = []

    def logging_tags(self) -> list[str]:
        return ["oolong_pairs"]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, dict[str, float]]]:
        return expand_rao_trajectories(trajectory_group, env_group)


@chz.chz
class OolongPairsDatasetBuilder(RLDatasetBuilder):
    """Train on one OOLONG-Pairs context-length bucket, eval on a longer one."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    depth: int = 1
    sub_reward_lambda: float = 0.0
    train_context_len: int = 8192
    eval_context_len: int = 32768
    num_train_examples: int = 20
    num_eval_examples: int = 20
    n_batches: int = 50
    max_iterations: int = 15
    child_max_iterations: int = 8
    max_trajectory_tokens: int = 32768
    max_sub_calls: int = 50
    eval_max_iterations: int | None = None
    eval_max_sub_calls: int | None = None
    sub_max_tokens: int = 8192
    sub_renderer_name: str | None = None
    disable_thinking: bool = True
    sub_temperature: float = 1.0
    repl_mem_limit_gb: int = 2
    repl_compute_timeout_s: float = 60.0
    nudge_hint: bool = True
    seed: int = 42

    def policy_renderer_name(self) -> str:
        if self.renderer_name is not None:
            return self.renderer_name
        candidates = model_info.get_recommended_renderer_names(self.model_name_for_tokenizer)
        if self.disable_thinking:
            for name in candidates:
                if "disable_thinking" in name or "no_thinking" in name:
                    return name
        return candidates[0] if candidates else "role_colon"

    def _sub_renderer_name(self) -> str:
        return self.sub_renderer_name or self.policy_renderer_name()

    def _sub_completer(self) -> MessageCompleter:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name_for_tokenizer)
        service_client = tinker.ServiceClient()
        return TinkerMessageCompleter(
            sampling_client=service_client.create_sampling_client(
                base_model=self.model_name_for_tokenizer
            ),
            renderer=get_renderer(self._sub_renderer_name(), tokenizer),
            max_tokens=self.sub_max_tokens,
            temperature=self.sub_temperature,
        )

    def _group_builder(
        self,
        row: dict[str, Any],
        group_size: int,
        sub_completer: MessageCompleter | None,
        *,
        is_eval: bool = False,
    ) -> PairsEnvGroupBuilder:
        max_iterations = self.max_iterations
        max_sub_calls = self.max_sub_calls
        if is_eval:
            max_iterations = self.eval_max_iterations or max_iterations
            max_sub_calls = self.eval_max_sub_calls or max_sub_calls
        return PairsEnvGroupBuilder(
            row=row,
            model_name=self.model_name_for_tokenizer,
            renderer_name=self.policy_renderer_name(),
            group_size=group_size,
            depth=self.depth,
            max_iterations=max_iterations,
            max_trajectory_tokens=self.max_trajectory_tokens,
            sub_reward_lambda=self.sub_reward_lambda,
            max_sub_calls=max_sub_calls,
            child_max_iterations=self.child_max_iterations,
            sub_completer=sub_completer,
            repl_mem_limit_bytes=self.repl_mem_limit_gb * 1024**3,
            repl_compute_timeout_s=self.repl_compute_timeout_s,
            nudge_hint=self.nudge_hint,
        )

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        sub_completer = self._sub_completer()
        train_rows = load_pairs(
            context_len=self.train_context_len,
            num_examples=self.num_train_examples,
            seed=self.seed,
        )
        train = RepeatingRLDataset(
            [self._group_builder(r, self.group_size, sub_completer) for r in train_rows],
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            seed=self.seed,
        )
        if self.num_eval_examples <= 0:
            return train, None
        eval_rows = load_pairs(
            context_len=self.eval_context_len,
            num_examples=self.num_eval_examples,
            seed=self.seed,
        )
        test = RepeatingRLDataset(
            [self._group_builder(r, 1, sub_completer, is_eval=True) for r in eval_rows],
            batch_size=len(eval_rows),
            n_batches=1,
        )
        return train, test
