"""Dispatch one launcher per smoke case. NOT a parallelizer — runs in series.

Default behavior prints the planned commands without executing. Pass `--execute`
to actually run them (after confirming GPU is free).

Usage:
    python scripts/smoke_all.py                 # dry run, prints commands
    python scripts/smoke_all.py --execute       # runs in series
    python scripts/smoke_all.py --only ldcq     # filter to one launcher
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

# (launcher, [extra args], env_d4rl_name, smoke_seconds)
SMOKES = [
    # CORL family
    ("run_corl.py",   ["--algo", "bc"],    "halfcheetah-medium-v0", 240),
    ("run_corl.py",   ["--algo", "cql"],   "halfcheetah-medium-v0", 240),
    ("run_corl.py",   ["--algo", "iql"],   "halfcheetah-medium-v0", 240),
    ("run_corl.py",   ["--algo", "td3bc"], "halfcheetah-medium-v0", 240),
    ("run_corl.py",   ["--algo", "dt"],    "halfcheetah-medium-v0", 240),

    # Trajectory diffusion
    ("run_diffuser.py", ["--algo", "diffuser"], "halfcheetah-medium-v0", 240),
    ("run_dd.py",       ["--algo", "dd"],       "halfcheetah-medium-v0", 240),

    # Diffusion policy
    ("run_dql.py",  ["--algo", "dql"],  "halfcheetah-medium-v0", 240),

    # Latent diffusion
    ("run_ldcq.py", ["--algo", "ldcq", "--stage", "skills"],    "halfcheetah-medium-v0", 240),
    ("run_ldcq.py", ["--algo", "ldcq", "--stage", "diffusion"], "halfcheetah-medium-v0", 240),
    ("run_ldcq.py", ["--algo", "ldcq", "--stage", "q_net"],     "halfcheetah-medium-v0", 240),
    # LDCQ eval is run only after stages 1-3 produce checkpoints — orchestrator does it.

    # EDA (3 stages)
    ("run_eda.py",  ["--algo", "eda", "--stage", "behavior"],   "halfcheetah-medium-v0", 240),
    ("run_eda.py",  ["--algo", "eda", "--stage", "critic"],     "halfcheetah-medium-v0", 240),
    ("run_eda.py",  ["--algo", "eda", "--stage", "finetune"],   "halfcheetah-medium-v0", 240),

    # QGPO (2 stages)
    ("run_qgpo.py", ["--algo", "qgpo", "--stage", "behavior"],  "halfcheetah-medium-v0", 240),
    ("run_qgpo.py", ["--algo", "qgpo", "--stage", "critic"],    "halfcheetah-medium-v0", 240),

    # Humanoid (only methods that accept --env / --env_name flag)
    ("run_ldcq.py", ["--algo", "ldcq", "--stage", "skills"],    "humanoid-medium-v0",    240),
    ("run_eda.py",  ["--algo", "eda",  "--stage", "behavior"],  "humanoid-medium-v0",    240),
    ("run_qgpo.py", ["--algo", "qgpo", "--stage", "behavior"],  "humanoid-medium-v0",    240),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true",
                   help="Actually run the smokes. Default is dry-run (print commands).")
    p.add_argument("--only", default=None,
                   help="Substring filter on launcher script name.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = os.environ.copy()
    env.setdefault("DATASET_OPTION", "C")
    env.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5555")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    for launcher, extra, env_name, secs in SMOKES:
        if args.only and args.only not in launcher:
            continue
        cmd = [sys.executable, str(SCRIPTS / launcher),
               *extra,
               "--env_d4rl_name", env_name,
               "--seed", str(args.seed),
               "--smoke",
               "--smoke_seconds", str(secs)]
        if not args.execute:
            print(" ".join(cmd))
            continue
        print(f"\n>>> {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT))
        print(f"    [exit {proc.returncode}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
