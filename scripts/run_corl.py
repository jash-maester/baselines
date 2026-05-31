"""Launcher for CORL methods (bc, cql, iql, td3bc, dt).

Approach:
  1. Install wandb_stub so CORL's `wandb.log` calls forward to MLflow.
  2. Start MLflow run.
  3. Build a *temporary* YAML config that copies the ship config and only
     adjusts `max_timesteps`, `eval_freq`, `n_episodes` for smoke runs.
     The ship config under `repos/CORL/configs/` is never modified.
  4. Hand off via runpy to `algorithms/offline/<algo>.py` with the temp YAML.

Usage:
    python scripts/run_corl.py --algo cql --env_d4rl_name halfcheetah-medium-v0 \
        --seed 0 --stage full [--smoke]

Notes:
  - CORL ships configs for halfcheetah/hopper/walker2d {medium, medium_replay,
    medium_expert} only. For Option C (medium-v0/expert-v0), we *temporarily*
    re-tag the config copy with the right Minari env name; the per-algo
    hyperparameters from the closest ship config (medium_v2) are used unchanged.
  - For Humanoid: no ship config exists -> we skip (decision D4).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

from _launch_common import (
    ROOT, REPOS, parse_common_args, env_to_minari, split_env_dataset,
    install_wandb_stub, mlflow_start,
)

ALGO_TO_SCRIPT = {
    "bc":    "algorithms/offline/any_percent_bc.py",
    "cql":   "algorithms/offline/cql.py",
    "iql":   "algorithms/offline/iql.py",
    "td3bc": "algorithms/offline/td3_bc.py",
    "dt":    "algorithms/offline/dt.py",
}

# CORL ship config directories (verified by inspection at HEAD eeeeef2)
ALGO_TO_CFG_DIR = {
    "bc":    "bc",
    "cql":   "cql",
    "iql":   "iql",
    "td3bc": "td3_bc",
    "dt":    "dt",
}

METHOD_FAMILY = {
    "bc": "behavior_cloning",
    "cql": "q_learning_offline",
    "iql": "q_learning_offline",
    "td3bc": "q_learning_offline",
    "dt": "sequence_modeling",
}

# Map dataset variant -> closest CORL ship-config filename (without .yaml).
CORL_SPLIT_TO_CFG = {
    "medium":         "medium_v2",
    "medium-replay":  "medium_replay_v2",
    "medium-expert":  "medium_expert_v2",
    "expert":         "medium_v2",   # CORL doesn't ship expert configs separately
}


def main() -> int:
    args = parse_common_args()
    algo = args.algo
    # User-dropped algos self-skip (rc=64) so a *running* orchestrator's already-
    # queued cells are excluded without a restart. Data-driven via
    # state/dropped_algos.json -> {"dropped": ["td3bc", ...]}.
    try:
        import json as _json
        _dropped = set(_json.loads((ROOT / "state" / "dropped_algos.json").read_text()).get("dropped", []))
    except Exception:
        _dropped = set()
    if algo in _dropped:
        print(f"[run_corl] algo {algo!r} is in the dropped list -> skipping (rc=64).", file=sys.stderr)
        return 64
    if algo not in ALGO_TO_SCRIPT:
        print(f"Unknown algo {algo!r}; expected one of {list(ALGO_TO_SCRIPT)}", file=sys.stderr)
        return 2
    env, dataset = split_env_dataset(args.env_d4rl_name)
    if env == "humanoid":
        print("CORL has no Humanoid ship config (decision D4 strict). Skipping.", file=sys.stderr)
        return 64  # custom signal so orchestrator can skip cleanly

    corl_root = REPOS / "CORL"
    cfg_dir = corl_root / "configs" / "offline" / ALGO_TO_CFG_DIR[algo] / env
    cfg_basename = CORL_SPLIT_TO_CFG.get(dataset, "medium_v2") + ".yaml"
    ship_cfg = cfg_dir / cfg_basename
    if not ship_cfg.exists():
        print(f"Ship config missing: {ship_cfg}", file=sys.stderr)
        return 2

    minari_id = env_to_minari(args.env_d4rl_name)

    # Build a working copy with the Option C env name + (if smoke) shrunk steps.
    with open(ship_cfg) as f:
        cfg = yaml.safe_load(f)
    cfg = dict(cfg) if cfg else {}
    # Replace the env name fields with the Option C name so the CORL script
    # asks for the right Minari dataset via our shim. DT reads `env_name`;
    # bc/cql/iql/td3bc read `env`. Set both to be safe.
    if "env" in cfg:
        cfg["env"] = args.env_d4rl_name
    if "env_name" in cfg:
        cfg["env_name"] = args.env_d4rl_name
    # Seed key varies per algo: bc/cql/iql/td3bc use `seed`; DT uses `train_seed`
    # + `eval_seed` AND does not declare `seed` at all (pyrallis is strict).
    if "seed" in cfg:
        cfg["seed"] = args.seed
    if "train_seed" in cfg:
        cfg["train_seed"] = args.seed
    if "eval_seed" in cfg:
        cfg["eval_seed"] = args.seed
    if args.smoke:
        # bc / cql / iql / td3bc keys
        if "max_timesteps" in cfg:
            cfg["max_timesteps"] = 200
        if "eval_freq" in cfg:
            cfg["eval_freq"] = 200
        if "n_episodes" in cfg:
            cfg["n_episodes"] = 2
        # DT keys
        if "update_steps" in cfg:
            cfg["update_steps"] = 200
        if "eval_every" in cfg:
            cfg["eval_every"] = 200
        if "eval_episodes" in cfg:
            cfg["eval_episodes"] = 2
        if "warmup_steps" in cfg:
            # DT's lr schedule does steps/warmup_steps — 0 would zero-div.
            cfg["warmup_steps"] = min(10, cfg.get("update_steps", 200))

    install_wandb_stub()

    run = mlflow_start(
        algo=algo,
        env=env,
        dataset=dataset,
        seed=args.seed,
        stage=args.stage,
        repo_url="https://github.com/corl-team/CORL",
        repo_path=corl_root,
        method_family=METHOD_FAMILY[algo],
        minari_id=minari_id,
        smoke=args.smoke,
        extra_config=cfg,
        extra_tags={"eval_mode": "det", "ship_config": str(ship_cfg.relative_to(corl_root))},
    )

    with tempfile.TemporaryDirectory() as td:
        temp_cfg = Path(td) / cfg_basename
        with open(temp_cfg, "w") as f:
            yaml.safe_dump(cfg, f)

        script = corl_root / ALGO_TO_SCRIPT[algo]
        # CORL scripts cd-around relative paths — run with corl_root as cwd.
        prev_cwd = os.getcwd()
        prev_argv = list(sys.argv)
        try:
            os.chdir(str(corl_root))
            sys.argv = [str(script), f"--config_path={temp_cfg}"]
            import runpy
            runpy.run_path(str(script), run_name="__main__")
        finally:
            os.chdir(prev_cwd)
            sys.argv = prev_argv
            try:
                run.__exit__(None, None, None)
            except Exception:
                pass
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
