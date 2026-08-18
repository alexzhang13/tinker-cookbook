"""
CLI for RL training of recursive sub-agent systems like RLMs.
"""

import asyncio
from datetime import datetime

import chz

from tinker_cookbook import cli_utils
from tinker_cookbook.recipes.rlm_rl.oolongp_env import OolongPairsDatasetBuilder
from tinker_cookbook.rl.train import Config, main


@chz.chz
class CLIConfig:
    # Required parameters
    model_name: str = "Qwen/Qwen3.5-9B"
    log_path: str | None = None
    load_checkpoint_path: str | None = None

    # Training parameters
    learning_rate: float = 5e-5
    group_size: int = 4
    groups_per_batch: int = 8
    max_steps: int = 50
    max_tokens: int = 4096

    # Model parameters
    lora_rank: int = 32

    # Infrastructure parameters
    base_url: str | None = None

    # Checkpointing and evaluation
    save_every: int = 5
    eval_every: int = 10
    num_eval_examples: int = 20

    # Dataset-specific parameters
    renderer_name: str | None = None
    train_context_len: int = 8192
    eval_context_len: int = 32768
    num_train_examples: int = 20
    seed: int = 42

    # RLM-specific parameters
    sub_renderer_name: str | None = None
    disable_thinking: bool = True
    depth: int = 1
    sub_reward_lambda: float = 0.0
    max_iterations: int = 15
    max_trajectory_tokens: int = 32768
    max_sub_calls: int = 200
    eval_max_iterations: int = 25
    eval_max_sub_calls: int = 400
    sub_max_tokens: int = 8192
    sub_temperature: float = 1.0
    repl_mem_limit_gb: int = 2
    repl_compute_timeout_s: float = 60.0
    nudge_hint: bool = True

    # Logging parameters
    wandb_project: str | None = None
    wandb_name: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    dataset_builder = OolongPairsDatasetBuilder(
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=cli_config.renderer_name,
        sub_renderer_name=cli_config.sub_renderer_name,
        disable_thinking=cli_config.disable_thinking,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        depth=cli_config.depth,
        sub_reward_lambda=cli_config.sub_reward_lambda,
        train_context_len=cli_config.train_context_len,
        eval_context_len=cli_config.eval_context_len,
        num_train_examples=cli_config.num_train_examples,
        n_batches=cli_config.max_steps,
        max_iterations=cli_config.max_iterations,
        max_trajectory_tokens=cli_config.max_trajectory_tokens,
        max_sub_calls=cli_config.max_sub_calls,
        eval_max_iterations=cli_config.eval_max_iterations,
        eval_max_sub_calls=cli_config.eval_max_sub_calls,
        sub_max_tokens=cli_config.sub_max_tokens,
        sub_temperature=cli_config.sub_temperature,
        repl_mem_limit_gb=cli_config.repl_mem_limit_gb,
        repl_compute_timeout_s=cli_config.repl_compute_timeout_s,
        nudge_hint=cli_config.nudge_hint,
        num_eval_examples=cli_config.num_eval_examples if cli_config.eval_every > 0 else 0,
        seed=cli_config.seed,
    )
    renderer_name = dataset_builder.policy_renderer_name()
    date_and_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = (
        f"rlm-oolong-pairs-{cli_config.train_context_len // 1024}k-to-"
        f"{cli_config.eval_context_len // 1024}k-"
        f"{cli_config.model_name.split('/')[-1]}-{date_and_time}"
    )

    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        log_path = f"/tmp/tinker-examples/rlm_rl/{run_name}"

    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = run_name

    config = Config(
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        recipe_name="rlm_rl",
        lora_rank=cli_config.lora_rank,
        learning_rate=cli_config.learning_rate,
        max_tokens=cli_config.max_tokens,
        dataset_builder=dataset_builder,
        max_steps=cli_config.max_steps,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        log_path=log_path,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        base_url=cli_config.base_url,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
