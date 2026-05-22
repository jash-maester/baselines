"""Launcher for Janner's Diffuser (`jannerm/diffuser`).

Three stages: train.py (diffusion) -> train_values.py (value) -> plan_guided.py.
Smoke runs only stage 1 with n_train_steps overridden. The full pipeline is
launched separately in Phase 6.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from _launch_common import (
    REPOS, parse_common_args, env_to_minari, split_env_dataset,
    install_wandb_stub, mlflow_start, alarm_timeout, end_mlflow_run, TimeoutSentinel,
)

DIFFUSER_ROOT = REPOS / "diffuser"


def main() -> int:
    args = parse_common_args(extra_args=[
        ("--stage_script", {"default": "scripts/train.py",
                              "help": "Which Diffuser stage to launch."}),
    ])
    env, dataset = split_env_dataset(args.env_d4rl_name)
    if env == "humanoid":
        print("Diffuser has no Humanoid block in config/locomotion.py (D5). Skipping.")
        return 64

    minari_id = env_to_minari(args.env_d4rl_name)
    install_wandb_stub()

    run = mlflow_start(
        algo="diffuser", env=env, dataset=dataset, seed=args.seed,
        stage=args.stage, repo_url="https://github.com/jannerm/diffuser",
        repo_path=DIFFUSER_ROOT, method_family="trajectory_diffusion",
        minari_id=minari_id, smoke=args.smoke,
        extra_tags={"eval_mode": "stoch_single"},
    )

    prev_cwd = os.getcwd()
    prev_argv = list(sys.argv)
    try:
        os.chdir(str(DIFFUSER_ROOT))
        sys.path.insert(0, str(DIFFUSER_ROOT))

        argv = [args.stage_script,
                "--dataset", args.env_d4rl_name,
                "--config", "config.locomotion"]
        if args.smoke:
            argv += ["--n_train_steps", "200",
                     "--n_steps_per_epoch", "100",
                     "--save_freq", "100",
                     "--sample_freq", "200",
                     "--n_saves", "1",
                     "--batch_size", "8"]
        sys.argv = argv
        with alarm_timeout(args.smoke_seconds if args.smoke else 0):
            try:
                import runpy
                runpy.run_path(str(DIFFUSER_ROOT / args.stage_script),
                               run_name="__main__")
            except (TimeoutSentinel, KeyboardInterrupt) as e:
                print(f"[run_diffuser] smoke timeout: {e}")
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv
        end_mlflow_run("FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
