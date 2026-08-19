"""
OOLONG-Real environment for Recursive Agent Optimization.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from typing import Any

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.recipes.rlm_rl.rao import (
    HarnessEnv,
    RepeatingRLDataset,
    expand_rao_trajectories,
)
from tinker_cookbook.recipes.rlm_rl.rlm.prompts import judge_system_prompt
from tinker_cookbook.recipes.rlm_rl.rlm.tools import current_renderer
from tinker_cookbook.recipes.rlm_rl.rlm_harness import RLMHarness
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, Trajectory

REAL_REPO = "oolongbench/oolong-real"
REAL_CONFIG = "dnd"
JUDGE_MODEL = "gpt-5-mini"


def dnd_parse_answer(answer: str) -> int | str | list[str]:
    try:
        return int(answer)
    except ValueError:
        pass
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]
    return answer


def dnd_parse_response(answer: str) -> tuple[int | str | list[str], str]:
    answer = answer.strip()
    match = re.search(r"\\boxed\{\\text\{([^}]*)\}\}", answer)
    if not match:
        match = re.search(r"\\boxed\{([^}]*)\}", answer)
    if match:
        return dnd_parse_answer(match.group(1)), "high"
    if not answer:
        return answer, "low"
    return dnd_parse_answer(answer), "med"


def dnd_score(datapoint: dict[str, Any], output: str | None) -> float:
    gold = dnd_parse_answer(str(datapoint["answer"]))
    trimmed_output, _parse_confidence = dnd_parse_response(output or "")
    if isinstance(gold, int) and isinstance(trimmed_output, int):
        return float(0.75 ** abs(gold - trimmed_output))
    if isinstance(gold, str) and isinstance(trimmed_output, str):
        return float(gold.strip().lower() == trimmed_output.strip().lower())
    if isinstance(gold, list) and isinstance(trimmed_output, list):
        overlap = set(gold) & set(trimmed_output)
        return float(len(overlap) / len(gold)) if gold else 0.0
    return 0.0


def _parse_judge_response(response: str) -> dict[str, Any]:
    json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        json_str = code_match.group(1).strip() if code_match else response.strip()
    parsed = json.loads(json_str)
    if not isinstance(parsed, dict):
        raise ValueError("Response must be a JSON object")
    for field in ("reason", "success"):
        if field not in parsed:
            raise ValueError(f"Missing required field: {field}")
    return parsed


def parse_judge_score(response: str) -> float:
    rubric = _parse_judge_response(response)
    success_flag = rubric["success"]
    if isinstance(success_flag, bool):
        return 1.0 if success_flag else 0.0
    return 0.0


def _action_history(harness: RLMHarness) -> str:
    parts: list[str] = []
    for message in harness.messages:
        if message.get("role") == "system":
            continue
        parts.append(f"{message.get('role', '')}: {get_text_content(message)}")
    return "\n".join(parts)


async def judge_subagent(harness: RLMHarness, *, model: str = JUDGE_MODEL) -> float:
    from openai import AsyncOpenAI

    goal = harness.root_prompt or ""
    if harness.context:
        prompt_start = f"# Task:\n{goal}\n\n# Context:\n{harness.context}"
    else:
        prompt_start = f"# Task:\n{goal}"
    final_message = harness.repl.final_answer
    user_prompt = (
        f"{prompt_start}\n\n# Agent Trajectory Info\n## Action History\n{_action_history(harness)}"
        f"\n\n## Agent Output\n{final_message if final_message is not None else 'No output provided'}"
        "\n\n## Error Message\nNo error message."
    )
    try:
        completion = await AsyncOpenAI().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": judge_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1,
        )
        text = completion.choices[0].message.content or ""
        return parse_judge_score(text)
    except Exception:
        return 0.0


def load_real(
    *,
    split: str,
    n_episodes: int,
    max_chars: int | None = None,
    num_examples: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(REAL_REPO, REAL_CONFIG, split=split)
    rows: list[dict[str, Any]] = []
    for ex in dataset:
        episodes = ex.get("episodes") or []
        if len(episodes) != n_episodes:
            continue
        text = ex["context_window_text"]
        if max_chars is not None and len(text) > max_chars:
            continue
        rows.append(dict(ex))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_examples is not None:
        rows = rows[:num_examples]
    if not rows:
        raise ValueError(
            f"no oolong-real examples for split={split} n_episodes={n_episodes} max_chars={max_chars}"
        )
    return rows


class RealEnvGroupBuilder(EnvGroupBuilder):
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
        sub_reward_lambda: float = 0.4,
        max_sub_calls: int = 50,
        child_max_iterations: int = 15,
        sub_completer: MessageCompleter | None = None,
        judge_model: str = JUDGE_MODEL,
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
        self.judge_model = judge_model
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
        row = self.row

        async def grade(final_answer: str | None) -> float:
            return dnd_score(row, final_answer)

        async def sub_grade(harness: RLMHarness) -> float:
            return await judge_subagent(harness, model=self.judge_model)

        self._harnesses_G = [
            RLMHarness(
                context=row["context_window_text"],
                root_prompt=row["question"],
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
                message_env=HarnessEnv(
                    h,
                    grade,
                    sub_reward_lambda=self.sub_reward_lambda,
                    sub_grade=sub_grade,
                ),
                failed_parse_reward=0.0,
                context_overflow_reward=0.0,
                max_trajectory_tokens=self.max_trajectory_tokens,
            )
            for h in self._harnesses_G
        ]

    async def cleanup(self) -> None:
        for h in self._harnesses_G:
            h.close()
        self._harnesses_G = []

    def logging_tags(self) -> list[str]:
        return ["oolong_real"]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, dict[str, float]]]:
        return expand_rao_trajectories(trajectory_group, env_group)


@chz.chz
class OolongRealDatasetBuilder(RLDatasetBuilder):
    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    depth: int = 2
    sub_reward_lambda: float = 0.4
    train_n_episodes: int = 1
    eval_n_episodes: int = 2
    train_max_chars: int = 240000
    num_train_examples: int | None = None
    num_eval_examples: int | None = 20
    n_batches: int = 50
    max_iterations: int = 15
    child_max_iterations: int = 15
    max_trajectory_tokens: int = 32768
    max_sub_calls: int = 50
    eval_max_iterations: int | None = None
    eval_max_sub_calls: int | None = None
    sub_max_tokens: int = 8192
    sub_renderer_name: str | None = None
    disable_thinking: bool = True
    sub_temperature: float = 1.0
    judge_model: str = JUDGE_MODEL
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
    ) -> RealEnvGroupBuilder:
        max_iterations = self.max_iterations
        max_sub_calls = self.max_sub_calls
        if is_eval:
            max_iterations = self.eval_max_iterations or max_iterations
            max_sub_calls = self.eval_max_sub_calls or max_sub_calls
        return RealEnvGroupBuilder(
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
            judge_model=self.judge_model,
            repl_mem_limit_bytes=self.repl_mem_limit_gb * 1024**3,
            repl_compute_timeout_s=self.repl_compute_timeout_s,
            nudge_hint=self.nudge_hint,
        )

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        sub_completer = self._sub_completer()
        train_rows = load_real(
            split="validation",
            n_episodes=self.train_n_episodes,
            max_chars=self.train_max_chars,
            num_examples=self.num_train_examples,
            seed=self.seed,
        )
        train = RepeatingRLDataset(
            [self._group_builder(r, self.group_size, sub_completer) for r in train_rows],
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            seed=self.seed,
        )
        if not self.num_eval_examples:
            return train, None
        eval_rows = load_real(
            split="test",
            n_episodes=self.eval_n_episodes,
            num_examples=self.num_eval_examples,
            seed=self.seed,
        )
        test = RepeatingRLDataset(
            [self._group_builder(r, 1, sub_completer, is_eval=True) for r in eval_rows],
            batch_size=len(eval_rows),
            n_batches=1,
        )
        return train, test
