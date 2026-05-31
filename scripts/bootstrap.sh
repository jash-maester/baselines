#!/usr/bin/env bash
# Bootstrap the baselines meta-repo on a fresh machine.
#
# Assumes: same hardware class as the origin host (RTX 5090 / Blackwell sm_120),
# `uv` installed, Python 3.11 available to uv, and network access to the shared
# MLflow at $MLFLOW_TRACKING_URI.
#
# Idempotent: safe to re-run. Does NOT launch training — see the printed
# next-steps at the end.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TORCH_INDEX="https://download.pytorch.org/whl/cu128"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://10.21.186.205:5555}"

say() { printf '\n\033[1;36m[bootstrap]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
say "0/6  preflight"
command -v uv >/dev/null || { echo "ERROR: uv not found. Install: https://docs.astral.sh/uv/"; exit 1; }
command -v git >/dev/null || { echo "ERROR: git not found."; exit 1; }
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "WARN: nvidia-smi not found — GPU runs will fail, CPU-only smoke still works."
fi

# ---------------------------------------------------------------------------
say "1/6  submodules (7 baseline repos under repos/)"
git submodule update --init --recursive

# ---------------------------------------------------------------------------
say "2/6  venv (.venv, Python 3.11)"
[ -d .venv ] || uv venv --python 3.11 .venv

# ---------------------------------------------------------------------------
# torch first, from the cu128 index — the pinned +cu128 build is not on PyPI.
say "3/6  torch (cu128 — Blackwell sm_120)"
uv pip install --python .venv/bin/python torch==2.11.0 --index-url "$TORCH_INDEX"

# ---------------------------------------------------------------------------
say "4/6  compat shim (editable) + pinned deps"
uv pip install --python .venv/bin/python -e ./compat
# --extra-index-url lets uv satisfy torch==2.11.0+cu128 from the lockfile pin.
# --index-strategy unsafe-best-match: with two indexes, uv's default first-index
# strategy pins a package to the first index that has it (the cu128 index for
# e.g. certifi) and won't fall back to PyPI for the locked version, breaking
# resolution. unsafe-best-match lets each *exact pinned* version resolve from
# whichever index carries it. Versions are unchanged — only index selection.
uv pip install --python .venv/bin/python -r requirements.lock.txt \
  --extra-index-url "$TORCH_INDEX" \
  --index-strategy unsafe-best-match

# ---------------------------------------------------------------------------
say "5/6  Minari datasets (4 envs x {medium,expert} = 8 datasets, downloads ~GB)"
MLFLOW_TRACKING_URI="$MLFLOW_URI" .venv/bin/python scripts/validate_minari.py

# ---------------------------------------------------------------------------
say "6/6  import sanity check"
.venv/bin/python - <<'PY'
import importlib
for m in ("compat_utils.env_factory", "compat_utils.mlflow_helper",
          "gymnasium", "minari", "torch", "mlflow", "pyrallis"):
    importlib.import_module(m)
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"sm={'.'.join(map(str, torch.cuda.get_device_capability())) if torch.cuda.is_available() else 'n/a'}")
print("  imports OK")
PY

cat <<EOF

[bootstrap] DONE.

MLflow target: $MLFLOW_URI
  This is the shared logging container, reachable over LAN. Confirm it resolves:
    curl -sf $MLFLOW_URI/health && echo OK

Make MLflow the default for every command in this shell session:
    export MLFLOW_TRACKING_URI=$MLFLOW_URI

Next steps:
  1) Smoke-test all methods (each < 5 min):   make smoke
  2) Launch the long matrix (resumes state):  make matrix
  3) Live dashboard:                          make dashboard

The matrix state (state/matrix.json) ships all-pending; \`make matrix\` starts
from the top with easy-first ordering and bin-packed concurrency.
EOF
