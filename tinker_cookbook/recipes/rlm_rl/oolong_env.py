"""OOLONG-synth dataset: trec-coarse split."""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import chz
import tinker

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.completers import MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.recipes.rlm_rl.harness.rlm_agent import RLMAgent
from tinker_cookbook.recipes.rlm_rl.rao import CyclingRLDataset, HarnessEnv
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.message_env import EnvFromMessageEnv
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder

TREC_INSTRUCTION = (
    "The context contains thousands of general-knowledge questions, one per "
    "line. Each line has a User ID and a question, and each question's answer "
    "falls into one of 6 categories: 'numeric value', 'entity', 'location', "
    "'description and abstract concept', 'abbreviation', 'human being'. "
    "Answer the following aggregate question."
)

COMPARISON_PHRASES = ("more common than", "less common than", "same frequency as")
_ANSWER_MARKER = re.compile(r"(?:final\s+answer|answer|label)\s*:", re.IGNORECASE)


def _find_comparison_phrase(output: str) -> str | None:
    out_low = output.lower()
    hits = [(out_low.rfind(p), p) for p in COMPARISON_PHRASES if p in out_low]
    return max(hits)[1] if hits else None


def _extract_final_answer(output: str) -> str:
    text = (output or "").strip()
    if not text:
        return ""
    hits = list(_ANSWER_MARKER.finditer(text))
    if hits:
        tail = text[hits[-1].end() :].strip()
        cand = tail.splitlines()[0].strip() if tail else ""
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cand = lines[-1] if lines else text
    return cand.replace("*", "").replace("[", "").replace("]", "").strip()


def synth_score(datapoint: dict[str, Any], output: str) -> float:
    answer = str(datapoint.get("answer", ""))
    try:
        if "datetime" in answer:
            gold: Any = datetime.strptime(answer, "[datetime.date(%Y, %m, %d)]")
        else:
            gold = ast.literal_eval(answer)[0]
    except Exception:
        gold = answer

    gold_s = str(gold)
    cand = _extract_final_answer(output)
    if not cand:
        return 0.0
    if cand == gold_s or cand.lower() == gold_s.lower():
        return 1.0

    atype = datapoint.get("answer_type", "")
    if atype == "ANSWER_TYPE.NUMERIC":
        nums = re.findall(r"-?\d+", cand)
        try:
            return 1.0 if nums and int(nums[0]) == int(gold) else 0.0
        except Exception:
            return 0.0
    if atype == "ANSWER_TYPE.DATE":
        import dateutil.parser

        try:
            return 1.0 if dateutil.parser.parse(cand) == gold else 0.0
        except Exception:
            return 0.0
    if gold_s.lower() in [p.lower() for p in COMPARISON_PHRASES]:
        return 1.0 if _find_comparison_phrase(cand) == gold_s.lower() else 0.0
    return 1.0 if gold_s.lower() in cand.lower() else 0.0


def load_oolong(
    *,
    context_len: int,
    num_examples: int,
    seed: int,
    split: str = "validation",
    dataset: str = "trec_coarse",
) -> list[dict[str, Any]]:
    import datasets

    stream = datasets.load_dataset("oolongbench/oolong-synth", split=split, streaming=True)
    # Shuffling precedes the filter below, so the buffer retains rows of every context length,
    # including 256k ones at ~1MB apiece. Keep it small.
    stream = stream.shuffle(seed=seed, buffer_size=1_000)
    keep = ("id", "question", "answer", "answer_type", "context_window_text")
    rows_P: list[dict[str, Any]] = []
    for ex in stream:
        if ex["dataset"] != dataset or int(ex["context_len"]) != context_len:
            continue
        rows_P.append({k: ex[k] for k in keep})
        if len(rows_P) >= num_examples:
            break
    if not rows_P:
        raise ValueError(f"no {dataset} rows with context_len={context_len}")
    return rows_P


class OolongEnvGroupBuilder(EnvGroupBuilder):
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
        max_sub_calls: int = 25,
        sub_completer: MessageCompleter | None = None,
        repl_mem_limit_bytes: int = 2 * 1024**3,
        repl_compute_timeout_s: float = 60.0,
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
        self.sub_completer = sub_completer
        self.repl_mem_limit_bytes = repl_mem_limit_bytes
        self.repl_compute_timeout_s = repl_compute_timeout_s
        self._harnesses_G: list[RLMAgent] = []

    async def make_envs(self) -> Sequence[Env]:
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer_utils.get_tokenizer(self.model_name))
        row = self.row

        async def grade(final_answer: str | None) -> float:
            return synth_score(row, final_answer or "")

        self._harnesses_G = [
            RLMAgent(
                context=row["context_window_text"],
                root_prompt=f"{TREC_INSTRUCTION}\n\nQuestion: {row['question']}",
                max_iterations=self.max_iterations,
                depth=self.depth,
                max_sub_calls=self.max_sub_calls,
                sub_completer=self.sub_completer,
                repl_mem_limit_bytes=self.repl_mem_limit_bytes,
                repl_compute_timeout_s=self.repl_compute_timeout_s,
            )
            for _ in range(self.group_size)
        ]
        return [
            EnvFromMessageEnv(
                renderer=renderer,
                message_env=HarnessEnv(harness, grade, sub_reward_lambda=self.sub_reward_lambda),
                failed_parse_reward=0.0,
                context_overflow_reward=0.0,
                max_trajectory_tokens=self.max_trajectory_tokens,
            )
            for harness in self._harnesses_G
        ]

    async def cleanup(self) -> None:
        for harness in self._harnesses_G:
            harness.close()
        self._harnesses_G = []

    def logging_tags(self) -> list[str]:
        return ["oolong", "trec_coarse"]


@chz.chz
class OolongDatasetBuilder(RLDatasetBuilder):
    """Train on one trec_coarse context-length bucket, eval on a longer one."""

    model_name_for_tokenizer: str
    batch_size: int
    group_size: int
    renderer_name: str | None = None
    depth: int = 2
    sub_reward_lambda: float = 0.0
    train_context_len: int = 32768
    eval_context_len: int = 262144
    num_train_examples: int = 50
    num_eval_examples: int = 50
    n_batches: int = 150
    max_iterations: int = 20
    max_trajectory_tokens: int = 16384
    max_sub_calls: int = 25
    sub_max_tokens: int = 16384
    # Sub-calls are tool-style extraction, not reasoning turns: on a thinking model the default
    # renderer burns the whole token budget on a trace and is truncated before it answers.
    sub_renderer_name: str | None = None
    sub_temperature: float = 1.0
    # Worst-case REPL memory is (groups_per_batch * group_size) * repl_mem_limit_gb: every
    # episode holds one REPL process concurrently.
    repl_mem_limit_gb: int = 2
    repl_compute_timeout_s: float = 60.0
    seed: int = 42

    def _sub_completer(self) -> MessageCompleter | None:
        """A dedicated sampling client for sub-agent turns, enabling concurrent fan-out.

        Only valid at ``sub_reward_lambda == 0``, where sub-agent turns are loss-masked and the
        rollout queue would buy nothing but serial latency. Above 0 those turns must land in
        the trajectory to receive credit, so the queue path is used instead.
        """
        if self.sub_reward_lambda != 0.0:
            return None
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

    def _sub_renderer_name(self) -> str:
        """Renderer for sub-agent calls, preferring a non-thinking variant when one exists."""
        if self.sub_renderer_name is not None:
            return self.sub_renderer_name
        candidates = model_info.get_recommended_renderer_names(self.model_name_for_tokenizer)
        for name in candidates:
            if "disable_thinking" in name or "no_thinking" in name:
                return name
        return candidates[0] if candidates else "role_colon"

    def _group_builder(
        self,
        row: dict[str, Any],
        group_size: int,
        sub_completer: MessageCompleter | None,
    ) -> OolongEnvGroupBuilder:
        return OolongEnvGroupBuilder(
            row=row,
            model_name=self.model_name_for_tokenizer,
            renderer_name=self.renderer_name,
            group_size=group_size,
            depth=self.depth,
            max_iterations=self.max_iterations,
            max_trajectory_tokens=self.max_trajectory_tokens,
            sub_reward_lambda=self.sub_reward_lambda,
            max_sub_calls=self.max_sub_calls,
            sub_completer=sub_completer,
            repl_mem_limit_bytes=self.repl_mem_limit_gb * 1024**3,
            repl_compute_timeout_s=self.repl_compute_timeout_s,
        )

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        sub_completer = self._sub_completer()
        train_rows_P = load_oolong(
            context_len=self.train_context_len, num_examples=self.num_train_examples, seed=self.seed
        )
        train = CyclingRLDataset(
            [self._group_builder(row, self.group_size, sub_completer) for row in train_rows_P],
            batch_size=self.batch_size,
            n_batches=self.n_batches,
            seed=self.seed,
        )
        if self.num_eval_examples <= 0:
            return train, None
        eval_rows_P = load_oolong(
            context_len=self.eval_context_len, num_examples=self.num_eval_examples, seed=self.seed
        )
        test = CyclingRLDataset(
            [self._group_builder(row, 1, sub_completer) for row in eval_rows_P],
            batch_size=len(eval_rows_P),
            n_batches=1,
        )
        return train, test
