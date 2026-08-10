# Training Recursive Agents with Tinker

Harnesses with LLM sub-agents, where a root LLM spawns sub-LLMs, are increasingly being used for most tasks. [Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601) [1] are one popular example, which use a persistent REPL with recursive sub-agents are functions, and the context as a variable in the REPL. [Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639) [2] is an RL training strategy around how to reward each LLM for recursive agent strategies like the RLM.

This recipe focuses on two examples of recursive agent training, one where we only assign rewards to the root model (i.e. RAO with $\lambda=0$), and one where we assign partial rewards to sub-agents as well during RL training.

## Installation
TODO: Figure out how `rlm-rl` works.

```bash
uv pip install 'tinker-cookbook[rlm-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

## RL on short tasks, generalizing to long tasks
This recipe first replicates a setup in the [Language model harnesses are compositional generalizers](https://alexzhang13.github.io/blog/2026/harness/) blogpost, which trains a `Qwen3-30B-A3B-Instruct` model as an RLM on only short tasks to see generalization on longer variants. We train only the root model, which can be thought of as applying RAO with $\lambda=0$.

### Train

```bash
python -m tinker_cookbook.recipes.rlm_rl.train \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    group_size=4 groups_per_batch=16 \
    learning_rate=5e-5 lora_rank=32 \
    depth=2 sub_reward_lambda=0.0 \
    train_context_len=32768 \
    max_steps=150
```

Train reward is logged as `"env/all/correct"` in `<log_path>/metrics.jsonl`.

### Eval

```bash
# step 0: base model on the 256k split
python -m tinker_cookbook.recipes.rlm_rl.train \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    eval_context_len=262144 \
    max_steps=1 eval_every=1 \
    log_path=/tmp/tinker-examples/rlm_rl/eval-step0

# step 150: trained checkpoint on the 256k split
python -m tinker_cookbook.recipes.rlm_rl.train \
    model_name="Qwen/Qwen3-30B-A3B-Instruct-2507" \
    load_checkpoint_path="tinker://<run-id>/..." \
    eval_context_len=262144 \
    max_steps=1 eval_every=1 \
    log_path=/tmp/tinker-examples/rlm_rl/eval-step150
```

Eval accuracy is logged as `"test/env/all/correct"`.

### Results: RLM(Qwen3-30B-A3B-Instruct) on OOLONG (trec-coarse, 256k)
| Benchmark | Total | PASS | FAIL | ERROR | Pass Rate |
|-----------|-------|------|------|-------|-----------|
| SWE-Bench Verified 1.0 | 500 | 145 (29.0%) | 52 (10.4%) | 303 (60.6%) | 29.0% |
| Terminal-Bench 2.0 | 89 | 14 (15.7%) | 31 (34.8%) | 44 (49.4%) | 15.7% |

## RL training RLMs on TextCraft-synth
The second recipe


[1] Zhang, A. L., Kraska, T., & Khattab, O. (2026). Recursive language models. arXiv preprint arXiv:2512.24601.

[2] Gandhi, A., Chakraborty, S., Wang, X., Kumar, A., & Neubig, G. (2026). Recursive agent optimization. arXiv preprint arXiv:2605.06639.


# TMP/ DELETE LATER
## How it works

- The policy never sees the long context in its prompt. It gets the question plus a Python REPL
  where the context is bound to a `context` string variable.
- Each turn the model writes a ```` ```python ```` block (executed in a persistent namespace,
  stdout returned, truncated) or answers with `FINAL: <answer>`, graded by exact match.
- Inside the REPL, `llm(prompt)` queries a sub-model instance — recursive delegation over
  slices of `context`.

Standard GRPO-style training: `group_size` rollouts per question, group-centered advantages.

## Run

```bash
python -m tinker_cookbook.recipes.rlm_rl.train
```

## Caveats (deliberately minimal)

- **RAO-lite:** only the root agent is trained; `llm()` hits a fixed-policy sub-model
  (the same base model by default). Full RAO — training the sub-agent trajectories with shared
  weights and credit assignment — is the natural next step.
- REPL code runs `exec` **in-process** with no sandboxing. Fine for this example; don't point it
  at untrusted data.
- Exact-match reward only; no partial credit, no LM judge.
