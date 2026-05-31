"""One-shot watcher: Discord-ping the moment the FIRST normalized score lands.

Polls the matrix MLflow experiment for the first run carrying a
`d4rl_normalized_score` (CORL/EDA/DQL) or `d4rl_normalized_score_gs*` (QGPO)
metric — these appear at the first in-training eval, often before a cell fully
finishes. On the first hit it sends one Discord summary, writes
state/first_score.flag (so other watchers/this session can react), and exits.

Run detached:
    setsid bash -c '.venv/bin/python scripts/first_score_notify.py' \
        </dev/null >logs/first_score_notify.out 2>&1 &
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from notify import send as discord_send          # noqa: E402

import mlflow                                      # noqa: E402
from mlflow.entities import ViewType              # noqa: E402

URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
POLL = int(os.environ.get("FIRSTSCORE_POLL_SEC", "60"))
FLAG = ROOT / "state" / "first_score.flag"


def _scored_runs():
    c = mlflow.tracking.MlflowClient(URI)
    e = c.get_experiment_by_name("latent_cep_baselines")
    if e is None:
        return []
    runs = c.search_runs([e.experiment_id], "", ViewType.ACTIVE_ONLY, max_results=500)
    hits = []
    for r in runs:
        m, t = r.data.metrics, r.data.tags
        val = m.get("d4rl_normalized_score")
        if val is None:
            gs = [v for k, v in m.items() if k.startswith("d4rl_normalized_score_gs")]
            val = max(gs) if gs else None
        if val is not None:
            hits.append((t.get("algo", "?"), t.get("env", "?"), t.get("dataset", "?"),
                         t.get("seed", "?"), r.info.status, float(val)))
    return hits


def main() -> int:
    print(f"[first-score] watching {URI} (poll {POLL}s)…")
    while True:
        try:
            hits = _scored_runs()
            if hits:
                hits.sort(key=lambda x: x[5], reverse=True)
                lines = [f"  {a}_{e}-{d}_s{s} = {v:.2f}  ({st})"
                         for (a, e, d, s, st, v) in hits[:8]]
                msg = (f":dart: First normalized scores landed — {len(hits)} run(s) with a "
                       f"d4rl_normalized_score:\n" + "\n".join(lines) +
                       "\nRun `make status` for the full Minari-vs-D4RL compare table.")
                ok = discord_send("summary", "matrix/first-scores", msg)
                FLAG.write_text(msg + f"\n(discord_sent={ok})\n")
                print(f"[first-score] notified (discord={ok}); exiting.")
                return 0
        except Exception as ex:
            print(f"[first-score] err: {ex}", file=sys.stderr)
        time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
