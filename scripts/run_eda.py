"""Launcher for EDA (thu-ml/Efficient-Diffusion-Alignment).

Three stages: train_behavior -> train_critic -> finetune_policy.
EDA hard-codes `n_epochs = 100` inside each train function (no CLI override).
Smoke relies on `alarm_timeout` to bound wall time. The repo's `--beta 0.1`
default is used unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from _launch_common import (
    REPOS, parse_common_args, env_to_minari, split_env_dataset,
    install_wandb_stub, mlflow_start, alarm_timeout, end_mlflow_run, TimeoutSentinel,
)

EDA_ROOT = REPOS / "Efficient-Diffusion-Alignment"

STAGE_TO_SCRIPT = {
    "behavior": "train_behavior.py",
    "critic":   "train_critic.py",
    "finetune": "finetune_policy.py",
}


def main() -> int:
    args = parse_common_args(extra_args=[
        ("--actor_load_path",  {"default": None}),
        ("--critic_load_path", {"default": None}),
        ("--beta", {"default": "0.1"}),
    ])
    if args.stage not in STAGE_TO_SCRIPT:
        print(f"Unknown EDA stage {args.stage!r}; expected {list(STAGE_TO_SCRIPT)}",
              file=sys.stderr)
        return 2
    env, dataset = split_env_dataset(args.env_d4rl_name)
    minari_id = env_to_minari(args.env_d4rl_name)

    expid = f"{args.env_d4rl_name}-baseline-seed{args.seed}"

    install_wandb_stub()
    run = mlflow_start(
        algo="eda", env=env, dataset=dataset, seed=args.seed, stage=args.stage,
        repo_url="https://github.com/thu-ml/Efficient-Diffusion-Alignment",
        repo_path=EDA_ROOT, method_family="diffusion_alignment",
        minari_id=minari_id, smoke=args.smoke,
        extra_tags={"eval_mode": "stoch_resample_4",
                    "beta": args.beta,
                    "tuning": "none_default",
                    "expid": expid},
    )

    prev_cwd = os.getcwd()
    prev_argv = list(sys.argv)
    try:
        os.chdir(str(EDA_ROOT))
        sys.path.insert(0, str(EDA_ROOT))
        script_rel = STAGE_TO_SCRIPT[args.stage]

        argv = [script_rel,
                "--env", args.env_d4rl_name,
                "--seed", str(args.seed),
                "--expid", expid,
                "--device", "cuda"]
        if args.stage == "finetune":
            argv += ["--beta", args.beta]
            # Auto-discover behavior + critic checkpoints from the same expid.
            factory_dir = EDA_ROOT / "EDA_model_factory" / expid
            actor_path = args.actor_load_path
            critic_path = args.critic_load_path
            if not actor_path and factory_dir.is_dir():
                cands = sorted(factory_dir.glob("behavior_ckpt*.pth"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
                if cands:
                    actor_path = str(cands[0])
            if not critic_path and factory_dir.is_dir():
                cands = sorted(factory_dir.glob("critic_ckpt*.pth"),
                               key=lambda p: p.stat().st_mtime, reverse=True)
                if cands:
                    critic_path = str(cands[0])
            if not actor_path or not critic_path:
                print(f"[run_eda:finetune] missing behavior/critic ckpt under {factory_dir}",
                      file=sys.stderr)
                return 65
            argv += ["--actor_load_path", actor_path,
                     "--critic_load_path", critic_path]
        sys.argv = argv

        with alarm_timeout(args.smoke_seconds if args.smoke else 0):
            try:
                import runpy
                runpy.run_path(str(EDA_ROOT / script_rel), run_name="__main__")
            except (TimeoutSentinel, KeyboardInterrupt) as e:
                print(f"[run_eda:{args.stage}] smoke timeout: {e}")
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv
        end_mlflow_run("FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
