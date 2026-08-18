# Training Recursive Agents on OOLONG-Pairs

Harnesses where a root LLM spawns sub-LLMs are increasingly used for long-context work.
[Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601) are one such design: a
persistent Python REPL holds the context as a variable, and recursive sub-agents are exposed as
functions the model can call. [Recursive Agent Optimization
(RAO)](https://arxiv.org/abs/2605.06639) is an RL strategy for training them.

This recipe trains an RLM with RL on [`mit-oasys/oolong-pairs`](https://huggingface.co/datasets/mit-oasys/oolong-pairs),
a long-context aggregation benchmark. Each question asks for **every pair of user IDs** satisfying
a condition (e.g. *"list all pairs of users who both have at least one numeric-value or location
instance"*), so the answer is a set of up to hundreds of `(id1, id2)` pairs. Training runs on a
short split and evaluation on a much longer one, which tests whether the REPL-plus-sub-call
strategy generalizes across context length.

Only the root agent is trained (RAO with lambda=0).

## Installation

```bash
uv pip install 'tinker-cookbook[rlm-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

## The dataset

OOLONG-Pairs ships questions and gold answers but **no context**. The context is pulled from
`oolongbench/oolong-synth` (`validation`, `dataset == "trec_coarse"`) at the matching
`context_len`; every example of a given length shares one context window. Gold answers were
computed against the *labelled* context, and the model is shown the *unlabelled* one, so it has
to infer each line's TREC category itself.

- Context lengths: 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576
- **20 questions per context length** (so the training pool is 20, all over one context)
- Gold set sizes at 8k range from 6 to 496 pairs; they grow combinatorially with length

## Reward: set F1

Exact match is unusable here -- a 496-element set is never reproduced exactly, so every rollout
would score 0.0 and RL would see no gradient. Reward is instead **F1 between the gold pair set and
the pairs parsed out of the model's answer** (`pairs_f1` in `oolongp_env.py`), which is dense and
credits partial progress. Pairs are matched order-insensitively (`(b, a)` == `(a, b)`).

Train reward is logged as `"env/all/correct"`, eval as `"test/env/all/correct"`; both are mean F1.

## Train (8k) and evaluate (32k)

The defaults are the setting that produced the results below: 50 steps at 8k, evaluated at 32k
every 10 steps.

```bash
python -m tinker_cookbook.recipes.rlm_rl.train \
    model_name="Qwen/Qwen3.5-9B" \
    log_path=/tmp/tinker-examples/rlm_rl/pairs-8k-to-32k
```

Spelled out, that is `group_size=4 groups_per_batch=8` (32 episodes/step), `learning_rate=5e-5
lora_rank=32`, `train_context_len=8192 eval_context_len=32768`, `max_iterations=15
max_sub_calls=200 sub_max_tokens=8192`, `nudge_hint=true`, and `eval_every=10
num_eval_examples=20` with `eval_max_iterations=25 eval_max_sub_calls=400`.

`nudge_hint=true` (the default) appends a block telling the model to act as an orchestrator: plan
a decomposition, push long-context work into `llm_query` / `llm_query_batched`, and prefer a few
dense prompts over many tiny ones (`prompts/nudge_hint.jinja`). `nudge_hint=false` leaves the
neutral prompt, which describes the REPL interface and says nothing about strategy. The two behave
very differently under training -- see the results below.

`disable_thinking=true` (the default) picks a non-thinking renderer for both the policy and the
sub-calls when the model has one. Qwen3.5 is a hybrid model whose recommended renderer opens a
`<think>` block, so the policy would otherwise spend its token budget on a reasoning preamble
instead of emitting a ```` ```repl ```` block -- worth ~0.3 F1 on this task. Passing
`renderer_name=` explicitly overrides the flag.

The eval split is longer than the training split, so it gets a larger exploration budget via
`eval_max_iterations` / `eval_max_sub_calls` -- with the training caps it would run out of
sub-calls before finishing its chunking.

### Base-model reference

```bash
python -m tinker_cookbook.recipes.rlm_rl.train \
    model_name="Qwen/Qwen3.5-9B" \
    max_steps=1 eval_every=1 \
    log_path=/tmp/tinker-examples/rlm_rl/pairs-eval-base-32k
```

Compare a trained checkpoint by adding
`load_checkpoint_path="tinker://<run-id>:train:0/weights/<step>"`. Keep
`eval_max_iterations` / `eval_max_sub_calls` identical across the two, or the comparison measures
the budget rather than the policy.

### Expected results

RLM(Qwen3.5-9B) on OOLONG-Pairs, trained at 8k and evaluated at 32k (n=20, mean set F1):

| Checkpoint | 32k F1 | sub-calls | turns |
|------------|--------|-----------|-------|
| step 0 (untrained) | 0.302 | 92.8 | 14.5 |
| step 50 | **0.549** | **17.1** | 10.6 |

Training roughly doubles long-context F1 while cutting sub-LLM calls by ~5x: the policy learns to
place a few dense, well-targeted calls instead of many small ones. Intermediate eval points were
0.321 / 0.477 / 0.546 / 0.691 at steps 10 / 20 / 30 / 40, so 0.549 is off a step-40 peak rather
than the top of a monotone climb. 8k train F1 rises from 0.453 to 0.565 over the run.

For scale: emitting all C(56,2)=1540 possible pairs scores F1 0.205 with no task understanding at
all, so read these against ~0.205 rather than against zero.

Running with `nudge_hint=false` gives a *stronger* untrained model (0.439) that then **declines**
under training (0.387 after 50 steps), while its sub-calls climb to 293 per episode. The hint is
what makes the learned strategy transfer to the longer context.

Wall clock: ~12h for 50 steps at 32 episodes/step (~12 min/step), sharing sampling capacity
with a second run.

Each eval point is n=20, so single comparisons are noisy (step 50 vs step 0 alone is z~1.6). The
result rests on the trajectory and on the gap to the `nudge_hint=false` arm.

## Caveats

- **The pool is 20 questions over a single context**, since every OOLONG-Pairs example at a given
  length shares one context window. Memorisation risk is high and the per-step reward is noisy.
- Only the root agent is trained (`sub_reward_lambda=0`); sub-calls run against a fixed sampling
  client. `sub_reward_lambda > 0` is not implemented.
- The model builds its answer inside the REPL, so it can emit hundreds of pairs without needing a
  large `max_tokens` -- but a truncated final answer costs recall directly under F1.
