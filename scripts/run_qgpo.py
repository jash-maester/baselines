"""Launcher for QGPO/CEP (thu-ml/CEP-energy-guided-diffusion, Offline_RL_2D).

Two stages: train_behavior -> train_critic. Uses README defaults
(diffusion_steps=15, alpha=3, q_alpha=1, method=CEP).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from _launch_common import (
    REPOS, parse_common_args, env_to_minari, split_env_dataset,
    install_wandb_stub, mlflow_start, alarm_timeout, end_mlflow_run, TimeoutSentinel,
)

QGPO_ROOT = REPOS / "CEP-energy-guided-diffusion" / "Offline_RL_2D"

STAGE_TO_SCRIPT = {
    "behavior": "train_behavior.py",
    "critic":   "train_critic.py",
}


def main() -> int:
    args = parse_common_args(extra_args=[
        ("--actor_load_path",  {"default": None}),
        ("--diffusion_steps", {"default": "15"}),
        ("--alpha",   {"default": "3"}),
        ("--q_alpha", {"default": "1"}),
        ("--method",  {"default": "CEP"}),
    ])
    if args.stage not in STAGE_TO_SCRIPT:
        print(f"Unknown QGPO stage {args.stage!r}; expected {list(STAGE_TO_SCRIPT)}",
              file=sys.stderr)
        return 2

    env, dataset = split_env_dataset(args.env_d4rl_name)
    minari_id = env_to_minari(args.env_d4rl_name)
    expid = f"{args.env_d4rl_name}{args.seed}reproduce"

    install_wandb_stub()
    run = mlflow_start(
        algo="qgpo", env=env, dataset=dataset, seed=args.seed, stage=args.stage,
        repo_url="https://github.com/thu-ml/CEP-energy-guided-diffusion",
        repo_path=QGPO_ROOT.parent, method_family="diffusion_policy",
        minari_id=minari_id, smoke=args.smoke,
        extra_tags={"eval_mode": "stoch_cep",
                    "diffusion_steps": args.diffusion_steps,
                    "alpha": args.alpha,
                    "q_alpha": args.q_alpha,
                    "method": args.method,
                    "expid": expid},
    )

    prev_cwd = os.getcwd()
    prev_argv = list(sys.argv)
    try:
        os.chdir(str(QGPO_ROOT))
        sys.path.insert(0, str(QGPO_ROOT))
        script_rel = STAGE_TO_SCRIPT[args.stage]

        argv = [script_rel,
                "--env", args.env_d4rl_name,
                "--seed", str(args.seed),
                "--expid", expid,
                "--device", "cuda"]
        if args.stage == "critic":
            argv += [
                "--diffusion_steps", args.diffusion_steps,
                "--alpha", args.alpha,
                "--q_alpha", args.q_alpha,
                "--method", args.method,
            ]
            # Auto-discover behavior checkpoint from the same expid.
            actor_path = args.actor_load_path
            if not actor_path:
                bdir = QGPO_ROOT / "models_rl" / expid
                if bdir.is_dir():
                    cands = sorted(bdir.glob("behavior_ckpt*.pth"),
                                   key=lambda p: p.stat().st_mtime, reverse=True)
                    if cands:
                        actor_path = str(cands[0])
            if not actor_path:
                print(f"[run_qgpo:critic] no behavior_ckpt*.pth under "
                      f"{QGPO_ROOT}/models_rl/{expid}", file=sys.stderr)
                return 65
            argv += ["--actor_load_path", actor_path]
        sys.argv = argv

        with alarm_timeout(args.smoke_seconds if args.smoke else 0):
            try:
                import runpy
                runpy.run_path(str(QGPO_ROOT / script_rel), run_name="__main__")
            except (TimeoutSentinel, KeyboardInterrupt) as e:
                print(f"[run_qgpo:{args.stage}] smoke timeout: {e}")
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv
        end_mlflow_run("FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
