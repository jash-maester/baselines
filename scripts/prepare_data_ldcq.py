"""Build the `repos/ldcq/data/<env>.pkl` files LDCQ expects.

LDCQ's training scripts do:
    dataset_file = 'data/' + env_name + '.pkl'
    with open(dataset_file, 'rb') as f:
        dataset = pickle.load(f)

The pickled object is the dict returned by D4RL's `env.get_dataset()`. Our
compat shim provides that dict via `gym.make(name).get_dataset()`, so we just
pickle it through the shim.

Usage:
    DATASET_OPTION=C python scripts/prepare_data_ldcq.py \
        --env_d4rl_name halfcheetah-medium-v0
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env_d4rl_name", required=True)
    p.add_argument("--out", default=None,
                   help="Output .pkl path (defaults to repos/ldcq/data/<env>.pkl).")
    args = p.parse_args()

    import gym  # compat shim
    env = gym.make(args.env_d4rl_name)
    dataset = env.get_dataset()
    # LDCQ expects 'observations', 'actions', 'rewards', 'terminals', 'timeouts' keys.
    for k in ("observations", "actions", "rewards", "terminals", "timeouts"):
        if k not in dataset:
            raise SystemExit(f"missing key {k!r} in shim get_dataset()")
    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] / "repos" / "ldcq" / "data" /
        f"{args.env_d4rl_name}.pkl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {out_path}")
    print(f"  observations: {dataset['observations'].shape}")
    print(f"  actions:      {dataset['actions'].shape}")
    print(f"  rewards:      {dataset['rewards'].shape}, sum={float(dataset['rewards'].sum()):.1f}")
    print(f"  terminals nz: {int(dataset['terminals'].sum())}")
    print(f"  timeouts nz:  {int(dataset['timeouts'].sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
