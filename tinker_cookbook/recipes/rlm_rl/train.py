"""Length-generalization RAO (lambda=0) on OOLONG trec_coarse: train at 32k,
eval at 256k. Mirrors rlm-minimal-training's lengthgen-oolong config."""

import asyncio
from datetime import datetime

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.rlm_rl.oolong_env import OolongDatasetBuilder
from tinker_cookbook.rl.train import Config, main


@chz.chz
class CLIConfig:
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None
    lora_rank: int = 32
    learning_rate: float = 5e-5

    group_size: int = 4
    groups_per_batch: int = 16
    max_steps: int = 150
    depth: int = 2
    sub_reward_lambda: float = 0.0

    train_context_len: int = 32768
    eval_context_len: int = 262144
    max_iterations: int = 20
    max_tokens: int = 4096
    max_trajectory_tokens: int = 16384
    max_sub_calls: int = 25
    sub_max_tokens: int = 16384
    sub_renderer_name: str | None = None
    sub_temperature: float = 1.0
    repl_mem_limit_gb: int = 2
    repl_compute_timeout_s: float = 60.0

    eval_every: int = 0
    save_every: int = 20
    log_path: str | None = None
    wandb_project: str | None = None
    seed: int = 42
    base_url: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli: CLIConfig):
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(cli.model_name)
    run_name = (
        f"rlm-oolong-trec-{cli.train_context_len // 1024}k-to-{cli.eval_context_len // 1024}k"
        f"-{cli.model_name.split('/')[-1]}-{datetime.now():%Y-%m-%d-%H-%M}"
    )
    log_path = cli.log_path or f"/tmp/tinker-examples/rlm_rl/{run_name}"

    config = Config(
        model_name=cli.model_name,
        renderer_name=renderer_name,
        recipe_name="rlm_rl",
        lora_rank=cli.lora_rank,
        learning_rate=cli.learning_rate,
        max_tokens=cli.max_tokens,
        dataset_builder=OolongDatasetBuilder(
            model_name_for_tokenizer=cli.model_name,
            renderer_name=renderer_name,
            batch_size=cli.groups_per_batch,
            group_size=cli.group_size,
            depth=cli.depth,
            sub_reward_lambda=cli.sub_reward_lambda,
            train_context_len=cli.train_context_len,
            eval_context_len=cli.eval_context_len,
            n_batches=cli.max_steps,
            max_iterations=cli.max_iterations,
            max_trajectory_tokens=cli.max_trajectory_tokens,
            max_sub_calls=cli.max_sub_calls,
            sub_max_tokens=cli.sub_max_tokens,
            sub_renderer_name=cli.sub_renderer_name,
            sub_temperature=cli.sub_temperature,
            repl_mem_limit_gb=cli.repl_mem_limit_gb,
            repl_compute_timeout_s=cli.repl_compute_timeout_s,
            num_eval_examples=50 if cli.eval_every > 0 else 0,
            seed=cli.seed,
        ),
        max_steps=cli.max_steps,
        load_checkpoint_path=cli.load_checkpoint_path,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        log_path=log_path,
        wandb_project=cli.wandb_project,
        wandb_name=run_name,
        base_url=cli.base_url,
    )
    cli_utils.check_log_dir(log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
