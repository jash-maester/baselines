"""Progress dashboard — read-only TUI over the orchestrator state files.

Shows:
  - Smoke loop status (state/autoresearch.json)
  - Matrix status (state/matrix.json) — counts, currently-running cell with
    elapsed time + estimated remaining, queue head with cumulative ETA.
  - GPU snapshot (nvidia-smi)
  - MLflow run counts per experiment

Default: one-shot. Use `--watch <sec>` to refresh.

Usage:
    python scripts/progress_dashboard.py
    python scripts/progress_dashboard.py --watch 30
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_AUTORES = ROOT / "state" / "autoresearch.json"
STATE_MATRIX  = ROOT / "state" / "matrix.json"
LOGS = ROOT / "logs"
REFERENCE_SCORES = ROOT / "reference_scores.json"


def _ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _hr(seconds: float) -> str:
    if seconds < 0:
        return "—"
    td = timedelta(seconds=int(seconds))
    return str(td)


def _last_metric_from_log(log_path: Path) -> str:
    """Grep the last line that looks like a loss/loss-like metric."""
    if not log_path.exists():
        return ""
    try:
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            f.seek(max(0, size - 4000))
            tail = f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    pat = re.compile(
        r"(?:train[._/]loss|Average\s+Loss|loss[._:=\s]+[-+0-9.eE]+)",
        re.IGNORECASE,
    )
    hits = pat.findall(tail)
    return hits[-1] if hits else ""


def _gpu_snapshot() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,utilization.gpu,memory.used,memory.total,name",
             "--format=csv,noheader,nounits"],
            timeout=5,
        ).decode().strip()
        return out
    except Exception as e:
        return f"(nvidia-smi failed: {e})"


def _mlflow_counts() -> dict:
    """Cheap count of runs per experiment via the REST API."""
    try:
        import urllib.request, urllib.error
        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
        out = {}
        for exp_name in ("latent_cep_baselines", "latent_cep_baselines_smoke"):
            req = urllib.request.Request(
                f"{uri}/api/2.0/mlflow/experiments/get-by-name?experiment_name={exp_name}",
                headers={"User-Agent": "baselines-dashboard"},
            )
            try:
                with urllib.request.urlopen(req, timeout=4) as r:
                    data = json.loads(r.read().decode())
                exp_id = data.get("experiment", {}).get("experiment_id")
                if not exp_id:
                    out[exp_name] = -1
                    continue
                # POST search-runs
                payload = json.dumps({"experiment_ids": [exp_id], "max_results": 1000}).encode()
                req2 = urllib.request.Request(
                    f"{uri}/api/2.0/mlflow/runs/search",
                    data=payload, method="POST",
                    headers={"User-Agent": "baselines-dashboard", "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req2, timeout=8) as r2:
                    runs = json.loads(r2.read().decode()).get("runs", [])
                out[exp_name] = len(runs)
            except urllib.error.HTTPError as e:
                out[exp_name] = -1
        return out
    except Exception:
        return {}


def _mlflow_runs_for_compare() -> list[dict]:
    """Pull every active baseline run with its latest `d4rl_normalized_score`.

    Returns a flat list of dicts: {algo, env, dataset, seed, status, score}.
    `score` is the most recent d4rl_normalized_score logged (None if absent).
    """
    try:
        import urllib.request
        uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
        # 1) experiment_id
        req = urllib.request.Request(
            f"{uri}/api/2.0/mlflow/experiments/get-by-name"
            f"?experiment_name=latent_cep_baselines",
            headers={"User-Agent": "baselines-dashboard"},
        )
        with urllib.request.urlopen(req, timeout=4) as r:
            exp_id = json.loads(r.read().decode())["experiment"]["experiment_id"]
        # 2) all active runs in that experiment (paginated)
        runs = []
        token = None
        for _ in range(20):  # safety cap
            payload = {"experiment_ids": [exp_id], "max_results": 1000,
                       "run_view_type": "ACTIVE_ONLY"}
            if token:
                payload["page_token"] = token
            req2 = urllib.request.Request(
                f"{uri}/api/2.0/mlflow/runs/search",
                data=json.dumps(payload).encode(), method="POST",
                headers={"User-Agent": "baselines-dashboard",
                         "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req2, timeout=10) as r2:
                resp = json.loads(r2.read().decode())
            runs.extend(resp.get("runs", []))
            token = resp.get("next_page_token")
            if not token:
                break
        # 3) extract score per run
        out = []
        for r in runs:
            info = r.get("info", {})
            data = r.get("data", {})
            tags = {t["key"]: t["value"] for t in data.get("tags", [])}
            metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
            score = metrics.get("d4rl_normalized_score")
            algo = tags.get("algo")
            env_d4rl = tags.get("env_d4rl_name") or _env_from_runname(
                tags.get("mlflow.runName", ""))
            dataset = tags.get("dataset")
            if not algo or not env_d4rl:
                continue
            out.append({
                "algo": algo,
                "env_d4rl_name": env_d4rl,
                "dataset": dataset,
                "seed": tags.get("seed"),
                "status": info.get("status"),
                "score": float(score) if score is not None else None,
            })
        return out
    except Exception:
        return []


def _env_from_runname(rn: str) -> str:
    """Fallback: parse env_d4rl_name from runs that omit the tag.

    Run names are `<algo>_<env-dataset>_seed<n>_<stage>`. The env-dataset
    segment is what we need; we append `-v0` since the matrix uses v0.
    """
    m = re.match(r"[a-z0-9]+_([a-z0-9-]+?)_seed", rn)
    if not m:
        return ""
    env_dataset = m.group(1)
    return env_dataset if env_dataset.endswith("-v0") else f"{env_dataset}-v0"


def _load_reference_scores() -> dict:
    if not REFERENCE_SCORES.exists():
        return {}
    try:
        return json.loads(REFERENCE_SCORES.read_text())
    except Exception:
        return {}


def _render_comparison() -> list[str]:
    """Compare our reproduced d4rl_normalized_score vs paper-reported numbers.

    Groups runs by (algo, env_d4rl_name), averages score across seeds,
    looks up the paper value, computes delta, flags discrepancies.
    """
    ref = _load_reference_scores()
    if not ref:
        return ["[compare vs paper]  reference_scores.json missing or unparseable"]
    runs = _mlflow_runs_for_compare()
    if not runs:
        return ["[compare vs paper]  no MLflow runs yet (or MLflow unreachable)"]

    warn_thr = float(ref.get("_warn_threshold", 5.0))
    fail_thr = float(ref.get("_fail_threshold", 15.0))
    paper = ref.get("scores", {})
    units = ref.get("_units", {})

    # Group: (algo, env) -> list of scores
    from collections import defaultdict
    grp: dict[tuple, list[float]] = defaultdict(list)
    grp_status: dict[tuple, list[str]] = defaultdict(list)
    for r in runs:
        if r["score"] is None:
            continue
        grp[(r["algo"], r["env_d4rl_name"])].append(r["score"])
        grp_status[(r["algo"], r["env_d4rl_name"])].append(r["status"] or "?")

    lines = [
        "[compare: Minari (ours) vs D4RL (paper)]  axis = same 0–100 scale, "
        "different dataset under the hood (Minari port vs original D4RL).",
        f"  Δ = ours − paper.  ✓ |Δ|≤{warn_thr:g}   ⚑ {warn_thr:g}<|Δ|≤{fail_thr:g}"
        f"   ✗ |Δ|>{fail_thr:g}   ?? unit mismatch (likely raw, not normalized)",
    ]
    header = f"  {'algo':<9} {'env':<22} {'unit':<16} {'n':>3} {'ours mean±std':>15}  {'paper':>7}  {'Δ':>9}  flag"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    # Stable ordering: by algo (paper-reference order), then env name.
    algo_order = list(paper.keys()) or sorted({a for (a, _) in grp})
    env_order = sorted({e for (_, e) in grp}) or [
        "halfcheetah-medium-v0", "halfcheetah-expert-v0",
        "hopper-medium-v0",      "hopper-expert-v0",
        "walker2d-medium-v0",    "walker2d-expert-v0",
        "humanoid-medium-v0",    "humanoid-expert-v0",
    ]

    for algo in algo_order:
        unit = units.get(algo, "?")
        for env in env_order:
            scores = grp.get((algo, env), [])
            ref_v = paper.get(algo, {}).get(env)
            if not scores and ref_v is None:
                continue
            if scores:
                n = len(scores)
                mean = sum(scores) / n
                if n > 1:
                    var = sum((s - mean) ** 2 for s in scores) / (n - 1)
                    std = var ** 0.5
                    ours = f"{mean:7.2f}±{std:5.2f}"
                else:
                    ours = f"{mean:7.2f}  —   "
            else:
                n = 0
                mean = None
                ours = "      —        "
            if ref_v is None:
                paper_s = "  —  "
                delta_s = "    —    "
                flag = ""
            elif mean is None:
                paper_s = f"{ref_v:6.2f}"
                delta_s = "    —    "
                flag = ""
            else:
                delta = mean - ref_v
                paper_s = f"{ref_v:6.2f}"
                delta_s = f"{delta:+8.2f}"
                ad = abs(delta)
                # Heuristic: if our score is 10× the paper number, units almost
                # certainly differ (raw vs normalized). Flag explicitly so the
                # user doesn't read it as a real reproduction gap.
                if unit == "d4rl_normalized" and ref_v != 0 and (
                        abs(mean) > 10 * max(abs(ref_v), 1.0)):
                    flag = "??"
                elif ad > fail_thr:
                    flag = "✗"
                elif ad > warn_thr:
                    flag = "⚑"
                else:
                    flag = "✓"
            lines.append(f"  {algo:<9} {env:<22} {unit:<16} {n:>3} {ours:>15}  {paper_s:>7}  {delta_s:>9}  {flag}")
    lines.append("  units + source paper for each algo: see reference_scores.json[\"_units\"] / [\"_sources\"]")
    return lines


def _render_autoresearch(state: dict | None) -> list[str]:
    if not state:
        return ["[autoresearch] state not yet created"]
    methods = state.get("methods", {})
    by = {}
    for k, m in methods.items():
        by.setdefault(m["status"], []).append((k, m))
    lines = ["[smoke autoresearch]"]
    n = len(methods)
    parts = []
    for s in ("green", "skipped", "red", "pending", "attempting", "awaiting_fix"):
        if s in by:
            parts.append(f"{s}={len(by[s])}")
    lines.append(f"  totals: {n} methods  |  " + "  ".join(parts))
    for k, m in by.get("attempting", []):
        lines.append(f"  ATTEMPTING  {k}  tries={len(m.get('attempts', []))}")
    for k, m in by.get("red", [])[:5]:
        lines.append(f"  RED         {k}  {m.get('notes','')}")
    return lines


def _render_matrix(state: dict | None) -> list[str]:
    if not state:
        return ["[matrix] state not yet created — call `long_runner.py --plan` to seed"]
    rows = state.get("rows", [])
    by = {}
    for r in rows:
        by.setdefault(r["status"], []).append(r)
    lines = ["[matrix]"]
    total = len(rows)
    parts = []
    for s in ("green", "running", "pending", "red", "skipped"):
        parts.append(f"{s}={len(by.get(s, []))}")
    lines.append(f"  totals: {total} cells  |  " + "  ".join(parts))

    # Currently running
    running = by.get("running", [])
    for r in running:
        t0 = _ts(r.get("started_at"))
        elapsed = (datetime.now(timezone.utc) - t0).total_seconds() if t0 else 0
        eta_remain = max(0, r["est_hours"] * 3600 - elapsed)
        log = LOGS / "matrix" / f"{r['key']}.log"
        last = _last_metric_from_log(log)
        lines.append(f"  RUNNING  {r['key']}  pid={r.get('pid')}  "
                     f"elapsed={_hr(elapsed)}  est_remain={_hr(eta_remain)}  "
                     f"last_metric=\"{last[:60]}\"")

    # Cumulative pending ETA
    pending_h = sum(r["est_hours"] for r in by.get("pending", []))
    lines.append(f"  pending est total: {pending_h:.1f}h = {pending_h/24:.2f}d (serial)")

    # Failure tail
    for r in by.get("red", [])[:5]:
        lines.append(f"  RED      {r['key']}  rc={r.get('rc')}  {r.get('notes','')[:80]}")
    return lines


def render() -> str:
    cols = shutil.get_terminal_size((100, 30)).columns
    bar = "=" * min(cols, 100)
    out = [bar, f"  baselines dashboard  ({datetime.now().isoformat(timespec='seconds')})", bar]

    out.extend(_render_autoresearch(_read_json(STATE_AUTORES)))
    out.append("")
    out.extend(_render_matrix(_read_json(STATE_MATRIX)))
    out.append("")
    out.append("[gpu]")
    out.append("  " + _gpu_snapshot())
    out.append("")
    out.append("[mlflow]")
    counts = _mlflow_counts()
    if counts:
        for k, v in counts.items():
            mark = "?" if v < 0 else str(v)
            out.append(f"  {k:>20} runs={mark}")
    else:
        out.append("  (mlflow unreachable)")
    out.append("")
    out.extend(_render_comparison())
    out.append(bar)
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--watch", type=int, default=0,
                   help="Refresh every N seconds (default 0 = one-shot).")
    args = p.parse_args()

    if args.watch <= 0:
        print(render())
        return 0
    try:
        while True:
            os.system("clear")
            print(render())
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
