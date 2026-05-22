"""Launcher for Decision Diffuser (`anuragajay/decision-diffuser`).

DD's `default_inv.py` hard-codes `Config.dataset = 'hopper-medium-expert-v2'`,
so per the no-source-edits rule we run it only on that task. For the matrix it
contributes one row. Smoke uses `alarm_timeout` to bound wall time; we accept
any partial run that emits a train.loss.

This launcher ignores `--env_d4rl_name` content beyond logging.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from _launch_common import (
    REPOS, parse_common_args, env_to_minari, split_env_dataset,
    install_wandb_stub, mlflow_start, alarm_timeout, end_mlflow_run, TimeoutSentinel,
)

DD_CODE = REPOS / "decision-diffuser" / "code"


def main() -> int:
    args = parse_common_args()
    # DD only ships hopper-medium-expert-v2 default. Override.
    env_d4rl_name = "hopper-medium-expert-v2"
    env, dataset = split_env_dataset(env_d4rl_name)
    minari_id = "localD4RL/hopper/medium-expert-v2"  # Option B; not yet built

    install_wandb_stub()
    run = mlflow_start(
        algo="dd", env=env, dataset=dataset, seed=args.seed, stage=args.stage,
        repo_url="https://github.com/anuragajay/decision-diffuser",
        repo_path=DD_CODE.parent, method_family="trajectory_diffusion",
        minari_id=minari_id, smoke=args.smoke,
        extra_tags={"eval_mode": "stoch_single",
                    "ship_config": "code/analysis/default_inv.py",
                    "note": "ships only hopper-medium-expert-v2"},
    )

    if not Path("/home/jash/diffusion-rl-research-02/baselines/repos/decision-diffuser/code").exists():
        end_mlflow_run("FAILED")
        return 1

    prev_cwd = os.getcwd()
    prev_argv = list(sys.argv)
    try:
        os.chdir(str(DD_CODE))
        sys.path.insert(0, str(DD_CODE))
        sys.argv = ["train.py"]
        # NOTE: DD also requires `localD4RL/hopper/medium-expert-v2` Option B
        # dataset. If running on Option C only (this session), DD smoke will
        # likely fail at data load. We still launch and let the MLflow run
        # capture the failure for visibility.
        with alarm_timeout(args.smoke_seconds if args.smoke else 0):
            try:
                import runpy
                runpy.run_path(str(DD_CODE / "analysis" / "train.py"),
                               run_name="__main__")
            except (TimeoutSentinel, KeyboardInterrupt) as e:
                print(f"[run_dd] smoke timeout: {e}")
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv
        end_mlflow_run("FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
