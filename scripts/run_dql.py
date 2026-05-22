"""Launcher for Diffusion-QL (Zhendong-Wang/Diffusion-Policies-for-Offline-RL).

Approach:
  - main.py reads hyperparameters from a hard-coded `hyperparameters` dict keyed
    on D4RL env name (-v2 names only).
  - For Option C (-v0), we add the v0 keys at runtime by copying the closest
    v2 row, then runpy the target.
  - Smoke uses `signal.alarm` to bound wall time; we accept any partial-run
    MLflow log that contains a train.loss-equivalent metric.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from _launch_common import (
    REPOS, parse_common_args, env_to_minari, split_env_dataset,
    install_wandb_stub, mlflow_start, alarm_timeout, end_mlflow_run, TimeoutSentinel,
)

DQL_ROOT = REPOS / "Diffusion-Policies-for-Offline-RL"


def main() -> int:
    args = parse_common_args(extra_args=[
        ("--device", {"default": "0"}),
    ])
    env, dataset = split_env_dataset(args.env_d4rl_name)
    if env == "humanoid":
        print("Diffusion-QL has no Humanoid hyperparameter row (decision D5). Skipping.")
        return 64

    minari_id = env_to_minari(args.env_d4rl_name)

    install_wandb_stub()
    run = mlflow_start(
        algo="dql", env=env, dataset=dataset, seed=args.seed, stage="full",
        repo_url="https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL",
        repo_path=DQL_ROOT, method_family="diffusion_policy",
        minari_id=minari_id, smoke=args.smoke,
        extra_tags={"eval_mode": "stoch_resample_50"},
    )

    # Inject v0-key into DQL's hyperparameter dict at import time.
    # We patch `main.hyperparameters` BEFORE running it via runpy.
    prev_cwd = os.getcwd()
    prev_argv = list(sys.argv)
    try:
        os.chdir(str(DQL_ROOT))
        sys.path.insert(0, str(DQL_ROOT))

        # Pre-import to register v0 hyperparameter rows derived from v2 defaults.
        import importlib
        if "main" in sys.modules:
            del sys.modules["main"]
        main_mod = importlib.import_module("main")
        hp = main_mod.hyperparameters
        # Copy closest v2 row for each v0 env we care about.
        v0_to_v2 = {
            "halfcheetah-medium-v0": "halfcheetah-medium-v2",
            "halfcheetah-expert-v0": "halfcheetah-medium-v2",
            "hopper-medium-v0":      "hopper-medium-v2",
            "hopper-expert-v0":      "hopper-medium-v2",
            "walker2d-medium-v0":    "walker2d-medium-v2",
            "walker2d-expert-v0":    "walker2d-medium-v2",
        }
        for v0, v2 in v0_to_v2.items():
            if v0 not in hp and v2 in hp:
                hp[v0] = dict(hp[v2])

        if args.env_d4rl_name not in hp:
            raise SystemExit(f"DQL has no hyperparameter row for {args.env_d4rl_name!r}")

        if args.smoke:
            hp[args.env_d4rl_name]["num_epochs"] = 1
            hp[args.env_d4rl_name]["eval_freq"] = 1

        # Build argv per README defaults.
        sys.argv = [
            "main.py",
            "--env_name", args.env_d4rl_name,
            "--device", args.device,
            "--ms", "online",
            "--lr_decay",
            "--seed", str(args.seed),
        ]
        with alarm_timeout(args.smoke_seconds if args.smoke else 0):
            try:
                main_mod_entry = main_mod
                # main.py has `if __name__ == "__main__"` guard; emulate by calling main()
                if hasattr(main_mod_entry, "main"):
                    main_mod_entry.main()
                else:
                    import runpy
                    runpy.run_path(str(DQL_ROOT / "main.py"), run_name="__main__")
            except (TimeoutSentinel, KeyboardInterrupt) as e:
                print(f"[run_dql] smoke timeout: {e}")
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv
        end_mlflow_run("FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
