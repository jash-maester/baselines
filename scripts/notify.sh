#!/usr/bin/env bash
# Bash wrapper around scripts/notify.py for use from Makefiles / cron / inside
# subprocesses where invoking Python directly is awkward.
#
# Usage:
#   ./scripts/notify.sh failure "ldcq:skills" "exhausted 3 retries"
#   ./scripts/notify.sh summary "autoresearch"  "7/7 repos green"
#
# $BASELINES_DISCORD_WEBHOOK env var overrides the baked-in URL.
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <level> <task> <msg>" >&2
  echo "  level ∈ {failure, summary, info, heartbeat}" >&2
  exit 2
fi

LEVEL="$1"; TASK="$2"; MSG="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Prefer the project's venv interpreter if present
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

"$PY" "$HERE/notify.py" --level "$LEVEL" --task "$TASK" --msg "$MSG"
