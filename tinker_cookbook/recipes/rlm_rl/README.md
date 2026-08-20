# Training Recursive Agents with Tinker

Harnesses with LLM sub-agents, where a root LLM spawns sub-LLMs, are increasingly being used for most tasks. [Recursive Language Models (RLM)](https://arxiv.org/abs/2512.24601) [1] are one popular example, which use a persistent REPL with recursive sub-agents are functions, and the context as a variable in the REPL. [Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639) [2] is an RL training strategy around how to reward each LLM for recursive agent strategies like the RLM.

This recipe focuses on two examples of recursive agent training, one where we only assign rewards to the root model (i.e. RAO with $\lambda=0$), and one where we assign partial rewards to sub-agents as well during RL training.

## Installation

```bash
uv pip install 'tinker-cookbook[rlm-rl] @ git+https://github.com/thinking-machines-lab/tinker-cookbook.git@nightly'
```

## RL on short tasks, generalizing to long tasks
This first recipe is similar to an experiment in the [Language model harnesses are compositional generalizers](https://alexzhang13.github.io/blog/2026/harness/) blogpost, which trains a local Qwen model as an RLM on only short tasks to see generalization on longer variants. We train only the root model, which can be thought of as applying RAO with $\lambda=0$.

This recipe trains a Recursive Language Model (RLM) with RL on sequences of input length 8K on [`Oolong-Pairs`](https://huggingface.co/datasets/mit-oasys/oolong-pairs) and evaluates on input sequences of length 32K. The output size grows combinatorially with length because the tasks asks for pairs of inputs satisfying some property, meaning the harness has to generalize to an order of magnitude larger output length.

The defaults are the setting that produced the results below: 50 steps at 8k, evaluated at 32k
every 10 steps.

```bash
python -m tinker_cookbook.recipes.rlm_rl.train \
    model_name="Qwen/Qwen3.5-9B" \
    log_path=/tmp/tinker-examples/rlm_rl/pairs-8k-to-32k \
    group_size=4 \
    groups_per_batch=8 \
    learning_rate=5e-5 \
    lora_rank=32 \
    train_context_len=8192 \
    eval_context_len=32768 \
    max_iterations=15 \
    max_sub_calls=200 \
    sub_max_tokens=8192 \
    nudge_hint=true \
    eval_every=10 \
    num_eval_examples=20 \
    eval_max_iterations=25 \
    eval_max_sub_calls=400
```

### Expected results

We use Qwen3.5-9B without thinking. You should find that it trains for roughly 12 hours total, with the number of sub-calls decreasing over time as the model learns to be more efficient, and F1 scores on both the training length of 8k and the eval length of 32k gradually rising over time.

| Checkpoint | 32k F1 | sub-calls | turns |
|------------|--------|-----------|-------|
| step 0 (untrained) | 0.302 | 92.8 | 14.5 |
| step 50 | **0.549** | **17.1** | 10.6 |


## RL on higher recursion depths
The second recipe trains an RLM with Recursive Agent Optimization ($\lambda > 0$), training the root and the sub-agents at recursion depth 2. We train on a split of (~55k) OOLONG-Real, and evaluate on
held-out(~118k) test examples. Sub-agents are trained on-policy with RAO ($\lambda=0.4$, depth 2).

Following the RAO setup, $\lambda>0$ scores each sub-agent with an LLM judge on OOLONG-Real. We `thinkingmachines/Inkling-Small` as the judge.

```bash
python -m tinker_cookbook.recipes.rlm_rl.train \
    dataset=real \
    model_name="Qwen/Qwen3.5-9B" \
    log_path=/tmp/tinker-examples/rlm_rl/real-55k-to-118k \
    group_size=4 \
    groups_per_batch=4 \
    learning_rate=3e-5 \
    lora_rank=32 \
    depth=2 \
    sub_reward_lambda=0.4 \
    max_iterations=15 \
    child_max_iterations=15 \
    max_sub_calls=50 \
    eval_every=50 \
    num_eval_examples=20 \
    max_steps=50
```

[1] Zhang, A. L., Kraska, T., & Khattab, O. (2026). Recursive language models. arXiv preprint arXiv:2512.24601.

[2] Gandhi, A., Chakraborty, S., Wang, X., Kumar, A., & Neubig, G. (2026). Recursive agent optimization. arXiv preprint arXiv:2605.06639.
