"""Launcher for LDCQ (ldcq/ldcq).

Four stages. Smoke truncates `--num_epochs`/`--n_epoch` to 1 and uses a small
batch. Each stage is its own MLflow run with stage tag.

Usage:
    python scripts/run_ldcq.py --algo ldcq --env_d4rl_name halfcheetah-medium-v0 \
        --seed 0 --stage skills [--smoke]

Stages: skills, diffusion, q_net, eval
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from _launch_common import (
    REPOS, parse_common_args, env_to_minari, split_env_dataset,
    install_wandb_stub, mlflow_start, alarm_timeout, end_mlflow_run, TimeoutSentinel,
)

LDCQ_ROOT = REPOS / "ldcq"


STAGE_TO_SCRIPT = {
    "skills":    "training/train_skills.py",
    "collect":   "training/collect_q_learning_dataset.py",   # produces *_states.npy etc.
    "diffusion": "training/train_diffusion.py",
    "q_net":     "training/train_q_net.py",
    "eval":      None,  # uses scripts/eval_ldcq_locomotion.py
}


def _find_latest_skill_ckpt(ldcq_root: Path, env_d4rl_name: str) -> Optional[str]:
    """Return basename of the most-recent skills checkpoint for this env.

    LDCQ's train_skills.py writes:
        checkpoints/skill_model_<env>_encoderType(...)_..._best.pth
    The downstream scripts expect just the basename via --skill_model_filename.
    """
    ckpt_dir = ldcq_root / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    pat = f"skill_model_{env_d4rl_name}_*_best.pth"
    candidates = sorted(ckpt_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        # Fall back to non-_best variant (epoch-numbered).
        pat2 = f"skill_model_{env_d4rl_name}_*.pth"
        candidates = sorted(ckpt_dir.glob(pat2), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name if candidates else None


def main() -> int:
    args = parse_common_args(extra_args=[
        ("--skill_model_filename", {"default": None}),
        ("--diffusion_model_filename", {"default": None}),
        ("--q_model_filename", {"default": None}),
    ])
    if args.stage not in STAGE_TO_SCRIPT:
        print(f"Unknown LDCQ stage {args.stage!r}; expected {list(STAGE_TO_SCRIPT)}",
              file=sys.stderr)
        return 2

    env, dataset = split_env_dataset(args.env_d4rl_name)
    minari_id = env_to_minari(args.env_d4rl_name)

    install_wandb_stub()
    run = mlflow_start(
        algo="ldcq", env=env, dataset=dataset, seed=args.seed,
        stage=args.stage, repo_url="https://github.com/ldcq/ldcq",
        repo_path=LDCQ_ROOT, method_family="latent_diffusion",
        minari_id=minari_id, smoke=args.smoke,
        extra_tags={"eval_mode": "stoch_argmax_q"},
    )

    # LDCQ scripts expect `data/<env>.pkl` (pre-built by us via
    # scripts/prepare_data_ldcq.py). The launcher only checks existence;
    # building the pkl is a separate step the orchestrator runs once per env.
    data_pkl = LDCQ_ROOT / "data" / f"{args.env_d4rl_name}.pkl"
    if args.stage in ("skills", "diffusion", "q_net") and not data_pkl.exists():
        print(f"[run_ldcq] WARN: {data_pkl} missing; will fail unless prepare_data_ldcq has been run.")

    prev_cwd = os.getcwd()
    prev_argv = list(sys.argv)
    try:
        os.chdir(str(LDCQ_ROOT))
        sys.path.insert(0, str(LDCQ_ROOT))
        # train_q_net.py does `from per_utils import NaivePrioritizedBuffer`,
        # but per_utils.py lives under training/. Add it so the import resolves.
        sys.path.insert(0, str(LDCQ_ROOT / "training"))

        if args.stage == "eval":
            # Custom eval script under baselines/scripts/.
            argv = [str(Path(__file__).with_name("eval_ldcq_locomotion.py")),
                    "--env_name", args.env_d4rl_name,
                    "--seed", str(args.seed),
                    "--num_eval_episodes", "2" if args.smoke else "10"]
            for k, attr in (("skill_ckpt", args.skill_model_filename),
                            ("diffusion_ckpt", args.diffusion_model_filename),
                            ("q_ckpt", args.q_model_filename)):
                if attr:
                    argv += [f"--{k}", attr]
            sys.argv = argv
            with alarm_timeout(args.smoke_seconds if args.smoke else 0):
                try:
                    import runpy
                    runpy.run_path(argv[0], run_name="__main__")
                except (TimeoutSentinel, KeyboardInterrupt) as e:
                    print(f"[run_ldcq:eval] smoke timeout: {e}")
        else:
            # The `collect` stage runs *two* upstream scripts: first
            # collect_q_learning_dataset.py (produces _states/_latents/_sT/
            # _rewards), then collect_offline_q_learning_dataset.py (produces
            # _sample_latents.npy needed by train_q_net.py).
            scripts_to_run = [STAGE_TO_SCRIPT[args.stage]]
            if args.stage == "collect":
                scripts_to_run.append("training/collect_offline_q_learning_dataset.py")

            # Auto-discover the skill checkpoint once for the whole stage.
            skill_fn = None
            if args.stage in ("collect", "diffusion", "q_net"):
                skill_fn = args.skill_model_filename or _find_latest_skill_ckpt(
                    LDCQ_ROOT, args.env_d4rl_name,
                )
                if not skill_fn:
                    print(
                        f"[run_ldcq:{args.stage}] no skill checkpoint found for "
                        f"{args.env_d4rl_name!r}; run --stage skills first.",
                        file=sys.stderr,
                    )
                    return 65

            for script_rel in scripts_to_run:
                # Stage-1 (skills) uses --env_name; downstream stages use --env.
                env_flag = "--env_name" if args.stage == "skills" else "--env"
                argv = [script_rel, env_flag, args.env_d4rl_name]
                if args.smoke:
                    if args.stage == "skills":
                        argv += ["--num_epochs", "1"]
                    elif args.stage in ("diffusion", "q_net"):
                        argv += ["--n_epoch", "1"]
                    # collect_offline_q_learning_dataset allocates
                    # (N_transitions, num_diffusion_samples, z_dim) — default
                    # num_diffusion_samples=1000 explodes to >100 GB on HC.
                    # Smoke uses 10 samples which is enough for the q_net
                    # smoke (which needs the file to exist, not be high-quality).
                    if "collect_offline" in script_rel:
                        argv += ["--num_diffusion_samples", "10"]
                if skill_fn:
                    argv += ["--skill_model_filename", skill_fn]
                # train_skills.py has no --device arg (cuda is hardcoded). Others do.
                if args.stage in ("collect", "diffusion", "q_net"):
                    argv += ["--device", "cuda"]
                sys.argv = argv

                print(f"[run_ldcq:{args.stage}] running {script_rel}")
                with alarm_timeout(args.smoke_seconds if args.smoke else 0):
                    try:
                        import runpy
                        runpy.run_path(str(LDCQ_ROOT / script_rel), run_name="__main__")
                    except (TimeoutSentinel, KeyboardInterrupt) as e:
                        print(f"[run_ldcq:{args.stage}] smoke timeout: {e}")
                        break
    finally:
        os.chdir(prev_cwd)
        sys.argv = prev_argv
        end_mlflow_run("FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
