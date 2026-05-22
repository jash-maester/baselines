"""Validate that all dataset IDs in the *active* mapping (Option B or C)
download successfully via minari. Run with DATASET_OPTION=C for this session.

Usage:
    DATASET_OPTION=C python scripts/validate_minari.py
"""
import os
import sys
import time

import minari

from compat_utils.minari_mapping import ACTIVE_MAP


def main() -> int:
    option = os.environ.get("DATASET_OPTION", "C").upper()
    print(f"DATASET_OPTION = {option}")
    print(f"Validating {len(ACTIVE_MAP)} dataset IDs from ACTIVE_MAP\n")

    ok, fail = [], []
    for d4rl_name, minari_id in ACTIVE_MAP.items():
        t0 = time.time()
        try:
            if minari_id.startswith("localD4RL/"):
                # Local datasets are validated by the conversion script, not here.
                print(f"SKIP {minari_id:50s}  (local; not in this session)")
                continue
            ds = minari.load_dataset(minari_id, download=True)
            n_eps = ds.total_episodes if hasattr(ds, "total_episodes") else "?"
            n_steps = ds.total_steps if hasattr(ds, "total_steps") else "?"
            ref_min = getattr(ds.spec, "reference_min_score", None)
            ref_max = getattr(ds.spec, "reference_max_score", None)
            dt = time.time() - t0
            print(
                f"OK   {minari_id:50s}  eps={n_eps} steps={n_steps} "
                f"ref=[{ref_min},{ref_max}] ({dt:.1f}s)"
            )
            ok.append(minari_id)
        except Exception as e:
            dt = time.time() - t0
            print(f"FAIL {minari_id:50s}  ({dt:.1f}s) -> {type(e).__name__}: {e}")
            fail.append((minari_id, str(e)))

    print()
    print(f"Summary: {len(ok)} ok, {len(fail)} failed")
    if fail:
        print("Failed datasets:")
        for mid, err in fail:
            print(f"  {mid} -> {err}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
