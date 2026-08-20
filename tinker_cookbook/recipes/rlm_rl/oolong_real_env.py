"""
OOLONG-Real environment for Recursive Agent Optimization.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import random
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.exceptions import ConfigurationError
from tinker_cookbook.recipes.rlm_rl.rao import (
    RAOHarnessEnv,
    RepeatingRLDataset,
    expand_rao_trajectories,
)
from tinker_cookbook.recipes.rlm_rl.rlm.prompts import judge_system_prompt
from tinker_cookbook.recipes.rlm_rl.rlm.tools import current_renderer
from tinker_cookbook.recipes.rlm_rl.rlm_harness import RLMHarness
from tinker_cookbook.renderers import get_renderer, get_text_content
from tinker_cookbook.renderers.base import Message
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, Trajectory

if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

OOLONG_REPO = "oolongbench/oolong-real"
OOLONG_CONFIG = "dnd"
# Namespaced names ("thinkingmachines/...") are Tinker models, bare names ("gpt-5-mini")
# are OpenAI models. The default judge runs on Tinker, so grading needs no second provider
# or API key; the paper's judge is one flag away (judge_model=gpt-5-mini).
TINKER_JUDGE_MODEL = "thinkingmachines/Inkling-Small"
OPENAI_JUDGE_MODEL = "gpt-5-mini"
JUDGE_MODEL = TINKER_JUDGE_MODEL
JUDGE_MAX_TOKENS = 2048
# Inkling is post-trained with an explicit thinking-effort message. Grading needs a little
# reasoning to check an answer against a transcript, but the verdict itself is one flag.
JUDGE_EFFORT = 0.7
OOLONG_INSTRUCTION = (
    "The transcript is the REPL variable `context`, not this message. "
    "For long context, chunk it (~32K characters) and call "
    "`launch_subagent(goal, chunk)` / `rlm_query(prompt, context=chunk)`; "
    "do not put the chunk in the goal, and do not use `llm_query` to read the transcript. "
    'Put only the answer value in `answer["content"]`: an integer, a short string, or a '
    "comma-separated list. Do not write a sentence (submit `10`, not "
    "'The count of Nat20s is 10.'). `\\boxed{...}` is also accepted."
)


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
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Last resort: an unfenced verdict wrapped in prose. Measured at roughly 5% of
        # verdicts from a Tinker judge, and each failure silently zeroes a node's reward.
        braces = re.search(r"\{.*\}", json_str, re.DOTALL)
        if braces is None:
            raise
        parsed = json.loads(braces.group(0))
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
    # A quoted boolean is common enough that treating it as a failure would silently
    # zero every sub-agent reward for a whole run.
    if isinstance(success_flag, str) and success_flag.strip().lower() in ("true", "false"):
        return 1.0 if success_flag.strip().lower() == "true" else 0.0
    return 0.0


def _action_history(harness: RLMHarness) -> str:
    parts: list[str] = []
    for message in harness.messages:
        if message.get("role") == "system":
            continue
        parts.append(f"{message.get('role', '')}: {get_text_content(message)}")
    return "\n".join(parts)


def _is_tinker_judge(model: str) -> bool:
    return "/" in model


class TinkerJudge:
    """Judge served by Tinker, using the credentials training already needs.

    Sampled from the frozen base model, never the policy being trained, so no sub-agent is
    graded by its own updated weights. Not a `TinkerMessageCompleter`: that renders without
    an explicit thinking effort, and an Inkling judge is post-trained with an effort message
    the renderer inserts, so leaving it implicit puts the judge off-distribution.
    """

    def __init__(
        self, model: str, effort: float = JUDGE_EFFORT, max_tokens: int = JUDGE_MAX_TOKENS
    ):
        renderer_names = model_info.get_recommended_renderer_names(model)
        renderer_name = next(
            (name for name in renderer_names if "disable_thinking" in name), renderer_names[0]
        )
        self.renderer = get_renderer(renderer_name, tokenizer_utils.get_tokenizer(model))
        self.sampling_client = tinker.ServiceClient().create_sampling_client(base_model=model)
        self.effort = effort
        self.max_tokens = max_tokens
        self._takes_effort = (
            "effort" in inspect.signature(self.renderer.build_generation_prompt).parameters
        )

    async def __call__(self, messages: list[Message]) -> str:
        effort_kwarg = {"effort": self.effort} if self._takes_effort else {}
        prompt = self.renderer.build_generation_prompt(messages, **effort_kwarg)
        response = await self.sampling_client.sample_async(
            prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=1.0,
                max_tokens=self.max_tokens,
                stop=self.renderer.get_stop_sequences(),
            ),
        )
        message, termination = self.renderer.parse_response(response.sequences[0].tokens)
        if not termination.is_clean:
            raise ValueError("judge response did not parse; it was probably cut at max_tokens")
        return get_text_content(message)


@functools.cache
def tinker_judge(model: str) -> TinkerJudge:
    return TinkerJudge(model)


@functools.cache
def judge_client() -> AsyncOpenAI:
    """The shared judge client, or a clear error if the optional dependency/key is missing.

    Cached: sub-agent grading fans out one call per node, and a fresh ``AsyncOpenAI``
    per call would leak a connection pool each time.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ConfigurationError(
            "Grading sub-agents (sub_reward_lambda > 0) needs the openai package: "
            "uv pip install 'tinker-cookbook[rlm-rl]'"
        ) from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise ConfigurationError(
            "An OpenAI judge model was selected for sub-agent grading, so OPENAI_API_KEY "
            f"must be set. The default judge ({TINKER_JUDGE_MODEL}) runs on Tinker and "
            "needs no extra key; sub_reward_lambda=0.0 trains the root agent only."
        )
    return AsyncOpenAI()


async def judge_subagent(harness: RLMHarness, *, model: str = JUDGE_MODEL) -> float:
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
        # Restated after the trajectory: with tens of thousands of tokens between the system
        # rubric and here, the judge otherwise drifts into answering the agent's task itself
        # (measured at 25-50% of verdicts once the prompt passes ~30K tokens).
        "\n\n# Your verdict\nYou are grading the agent above, not solving its task. Do not "
        "answer the task yourself. Reply with only the JSON object described in your "
        "instructions, with the `reason` and `success` fields."
    )
    messages: list[Message] = [
        {"role": "system", "content": judge_system_prompt(no_repl=harness.no_repl)},
        {"role": "user", "content": user_prompt},
    ]
    try:
        if _is_tinker_judge(model):
            return parse_judge_score(await tinker_judge(model)(messages))
        completion = await judge_client().chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=1,
        )
        return parse_judge_score(completion.choices[0].message.content or "")
    except ConfigurationError:
        # A missing dependency or key is a misconfigured run, not a bad sub-agent.
        raise
    except Exception:
        # An unreachable judge or an unparsable rubric scores 0, but must be visible:
        # silently returning 0 for every node looks exactly like a policy that never
        # delegates well.
        logger.warning("sub-agent judge failed; scoring this node 0.0", exc_info=True)
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

    dataset = load_dataset(OOLONG_REPO, OOLONG_CONFIG, split=split)
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
                root_prompt=f"{OOLONG_INSTRUCTION}\n\nQuestion: {row['question']}",
                max_iterations=self.max_iterations,
                max_depth=self.depth,
                child_max_iterations=self.child_max_iterations,
                max_sub_calls=self.max_sub_calls,
                sub_completer=self.sub_completer,
                mem_limit_bytes=self.repl_mem_limit_bytes,
                compute_timeout_s=self.repl_compute_timeout_s,
                nudge_hint=self.nudge_hint,
                on_policy_llm_query=self.sub_reward_lambda != 0.0,
            )
            for _ in range(self.group_size)
        ]
        return [
            EnvFromMessageEnv(
                renderer=renderer,
                message_env=RAOHarnessEnv(
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
        if self.sub_reward_lambda != 0.0 and not _is_tinker_judge(self.judge_model):
            # Fail at startup rather than on the first graded sub-agent, several
            # minutes into step 0. A Tinker judge needs no key beyond TINKER_API_KEY.
            judge_client()
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
