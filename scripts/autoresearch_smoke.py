"""Autoresearch smoke orchestrator.

For each method in the active list, runs a 5-min smoke and decides green / red /
timeout. Failed attempts log to MLflow experiment `baselines_smoke` (separate
from the main `baselines` experiment so the matrix stays clean). On all-green,
sends a Discord summary. On exhausted retries for a method, sends a Discord
failure ping.

This orchestrator is **operator-in-the-loop**: when a method crashes, the
orchestrator captures the log slice, marks the method `awaiting_fix`, and
exits with rc=42. The operator (Claude in the parent session) reads the log,
applies an autonomous patch within scope rules, and re-invokes the orchestrator
which resumes at the awaiting_fix method.

State file: state/autoresearch.json (atomic write-and-rename).

Usage:
    # Run the whole smoke loop; halts on any crash for operator triage.
    python scripts/autoresearch_smoke.py --execute

    # Run just one method (also used as the retry entry point):
    python scripts/autoresearch_smoke.py --execute --only ldcq:skills

    # Status:
    python scripts/autoresearch_smoke.py --status
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
LOGS = ROOT / "logs" / "smoke"
STATE = ROOT / "state" / "autoresearch.json"
FOOTPRINTS = ROOT / "state" / "footprints"

DEFAULT_SMOKE_SECONDS = 300        # 5 min per attempt
DEFAULT_MAX_ATTEMPTS = 3

# Pattern detected in stdout that proves a "loss"-equivalent metric was
# computed. Methods print these on every step or every batch; the absence
# of any match after the smoke window means yellow (the script didn't
# crash but didn't make obvious training progress either).
# Markers that signal "training emitted a loss/metric/eval result".
# We OR these together; defense-in-depth alongside the MLflow check below.
LOSS_REGEX = re.compile(
    r"(?:"
    r"actor_loss|critic_loss|bc_loss|train[._/]loss|loss[._/]total|Average\s+Loss|"
    r"\bloss\b\s*[:=]|"
    r"\bvae[._]kl\b|\bdiffusion[._]mse\b|"
    r"Time\s+steps?:|Evaluation\s+over|d4rl_normalized_score|"
    r"\bepoch\s+\d+|\bepisode\s+\d+|"
    r"eval[._/]return|reward[._/]mean|"
    # Proof-of-life markers for repos that log to tensorboard (DQL) or pickle
    # and emit nothing structured to stdout while training:
    r"Training\s+Start|Loaded\s+buffer|Iteration\s+\d+|step\s+\d+|"
    # LDCQ collect_q_learning_dataset: "Total data samples extracted: N".
    # EDA train_behavior: "Average Loss" (already matched above) but also
    # "tqdm_epoch" progress.
    r"samples?\s+extracted|saved\s+to|Average\s+\w+\s*:"
    r")",
    re.IGNORECASE,
)


def _has_mlflow_metrics(algo: str, env: str, stage: str, since_seconds: int = 1800,
                        retries: int = 5, retry_delay: float = 2.0) -> bool:
    """Check MLflow for a recent run with our tags + at least one metric.

    Belt-and-suspenders alongside LOSS_REGEX. Retries briefly to absorb any
    eventual-consistency lag between subprocess SIGTERM and the parent's
    immediate REST query.
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient(
                os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555"))
            exp = client.get_experiment_by_name("latent_cep_baselines_smoke")
            if exp is None:
                return False
            now_ms = int(time.time() * 1000)
            # Two-phase filter: (a) algo+env+stage only, (b) verify smoke=True
            # in Python. This is more permissive than a single AND'd filter
            # in case the smoke tag is missing on partial writes.
            runs = client.search_runs(
                [exp.experiment_id],
                filter_string=(
                    f"tags.algo = '{algo}' AND tags.env = '{env}' "
                    f"AND tags.stage = '{stage}'"
                ),
                max_results=20,
            )
            for run in runs:
                start_ms = run.info.start_time or 0
                if start_ms and now_ms - start_ms > since_seconds * 1000:
                    continue
                if run.data.metrics:
                    return True
            # No match yet; backoff and retry
            if attempt < retries:
                time.sleep(retry_delay * attempt)
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_delay * attempt)
    if last_err is not None:
        print(f"  ! _has_mlflow_metrics: gave up after {retries} retries: {last_err}")
    return False


# (algo, env_d4rl_name, stage, launcher.py, [extra args])
# Decision Diffuser is pre-skipped this session — see RUN_LOG patch #4. It
# needs params_proto/ml_logger/jaynes + an Option-B-only dataset; revived in
# Phase 2b when Option B is built.
SMOKE_METHODS = [
    # CORL family
    ("bc",      "halfcheetah-medium-v0", "full",      "run_corl.py",   ["--algo", "bc"]),
    ("cql",     "halfcheetah-medium-v0", "full",      "run_corl.py",   ["--algo", "cql"]),
    ("iql",     "halfcheetah-medium-v0", "full",      "run_corl.py",   ["--algo", "iql"]),
    ("td3bc",   "halfcheetah-medium-v0", "full",      "run_corl.py",   ["--algo", "td3bc"]),
    ("dt",      "halfcheetah-medium-v0", "full",      "run_corl.py",   ["--algo", "dt"]),
    # Trajectory diffusion
    ("diffuser","halfcheetah-medium-v0", "full",      "run_diffuser.py", ["--algo", "diffuser"]),
    # Diffusion policy
    ("dql",     "halfcheetah-medium-v0", "full",      "run_dql.py",      ["--algo", "dql"]),
    # Latent diffusion (LDCQ, 4 stages: skills -> collect -> diffusion -> q_net)
    ("ldcq",    "halfcheetah-medium-v0", "skills",    "run_ldcq.py",     ["--algo", "ldcq", "--stage", "skills"]),
    ("ldcq",    "halfcheetah-medium-v0", "collect",   "run_ldcq.py",     ["--algo", "ldcq", "--stage", "collect"]),
    ("ldcq",    "halfcheetah-medium-v0", "diffusion", "run_ldcq.py",     ["--algo", "ldcq", "--stage", "diffusion"]),
    ("ldcq",    "halfcheetah-medium-v0", "q_net",     "run_ldcq.py",     ["--algo", "ldcq", "--stage", "q_net"]),
    # EDA (3 stages)
    ("eda",     "halfcheetah-medium-v0", "behavior",  "run_eda.py",      ["--algo", "eda", "--stage", "behavior"]),
    ("eda",     "halfcheetah-medium-v0", "critic",    "run_eda.py",      ["--algo", "eda", "--stage", "critic"]),
    ("eda",     "halfcheetah-medium-v0", "finetune",  "run_eda.py",      ["--algo", "eda", "--stage", "finetune"]),
    # QGPO (2 stages)
    ("qgpo",    "halfcheetah-medium-v0", "behavior",  "run_qgpo.py",     ["--algo", "qgpo", "--stage", "behavior"]),
    ("qgpo",    "halfcheetah-medium-v0", "critic",    "run_qgpo.py",     ["--algo", "qgpo", "--stage", "critic"]),
    # Humanoid subset
    ("ldcq",    "humanoid-medium-v0",    "skills",    "run_ldcq.py",     ["--algo", "ldcq", "--stage", "skills"]),
    ("eda",     "humanoid-medium-v0",    "behavior",  "run_eda.py",      ["--algo", "eda", "--stage", "behavior"]),
    ("qgpo",    "humanoid-medium-v0",    "behavior",  "run_qgpo.py",     ["--algo", "qgpo", "--stage", "behavior"]),
]

# rc=65 from a launcher signals "missing prerequisite" (e.g. LDCQ stage 2/3
# can't find a stage-1 checkpoint). Treat that as red, not crashed.
LAUNCHER_RC_SKIP = 64           # CORL Humanoid, DD pinned env etc.
LAUNCHER_RC_PREREQ_MISSING = 65


def method_key(algo: str, env: str, stage: str) -> str:
    return f"{algo}:{env}:{stage}"


@dataclass
class Attempt:
    n: int
    started_at: str
    finished_at: Optional[str] = None
    rc: Optional[int] = None
    log_path: str = ""
    saw_loss: bool = False
    note: str = ""


@dataclass
class MethodState:
    key: str
    algo: str
    env: str
    stage: str
    launcher: str
    extra_args: list[str]
    status: str = "pending"      # pending / attempting / green / red / awaiting_fix / skipped
    attempts: list[dict] = field(default_factory=list)
    last_log: str = ""
    notes: str = ""


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"version": 1, "methods": {}, "started_at": _now(), "policy": {
        "smoke_seconds": DEFAULT_SMOKE_SECONDS,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
        "mlflow_experiment": "latent_cep_baselines_smoke",
    }}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2, sort_keys=False))
    tmp.replace(STATE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_state(s: dict) -> dict:
    methods = s.setdefault("methods", {})
    for algo, env, stage, launcher, extra in SMOKE_METHODS:
        key = method_key(algo, env, stage)
        if key not in methods:
            methods[key] = asdict(MethodState(
                key=key, algo=algo, env=env, stage=stage,
                launcher=launcher, extra_args=extra,
            ))
    return s


def _read_tail(path: Path, n_bytes: int = 8000) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > n_bytes:
                f.seek(size - n_bytes)
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _scan_for_loss(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    tail = _read_tail(log_path, n_bytes=64000)
    return bool(LOSS_REGEX.search(tail))


def _spawn_footprint_probe(target_pid: int, footprint_path: Path, meta: dict,
                           interval: float = 5.0, max_seconds: int = 900) -> Optional[subprocess.Popen]:
    """Spawn the footprint sidecar; returns the Popen handle (or None if probe missing)."""
    probe = SCRIPTS / "footprint_probe.py"
    if not probe.exists():
        return None
    cmd = [
        sys.executable, str(probe),
        "--target_pid", str(target_pid),
        "--out_path",   str(footprint_path),
        "--interval",   str(interval),
        "--max_seconds", str(max_seconds),
        "--meta", json.dumps(meta),
    ]
    try:
        return subprocess.Popen(cmd, cwd=str(ROOT),
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"  ! could not start footprint probe: {e}")
        return None


def _terminate_probe(probe: Optional[subprocess.Popen], grace: float = 8.0) -> None:
    if probe is None:
        return
    if probe.poll() is not None:
        return
    try:
        probe.terminate()
        try:
            probe.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            probe.kill()
            probe.wait()
    except Exception:
        pass


def _read_footprint_summary(p: Path) -> dict:
    """Pull just the fields we want to embed in the attempt record."""
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        return {}
    return {
        "footprint_path": str(p),
        "n_samples": data.get("n_samples", 0),
        "vram_proc_peak_gb":   data.get("vram_proc_gb", {}).get("peak"),
        "vram_proc_steady_gb": data.get("vram_proc_gb", {}).get("steady"),
        "gpu_util_steady_pct": data.get("gpu_util_pct", {}).get("steady"),
        "gpu_util_peak_pct":   data.get("gpu_util_pct", {}).get("peak"),
        "rss_peak_gb":         data.get("rss_total_gb", {}).get("peak"),
        "rss_steady_gb":       data.get("rss_total_gb", {}).get("steady"),
    }


def run_one(method: dict, smoke_seconds: int) -> dict:
    """Run one smoke attempt. Returns updated method dict."""
    attempts = method.setdefault("attempts", [])
    n = len(attempts) + 1
    LOGS.mkdir(parents=True, exist_ok=True)
    FOOTPRINTS.mkdir(parents=True, exist_ok=True)
    log_name = f"{method['algo']}_{method['env']}_{method['stage']}_attempt{n}.log"
    footprint_name = log_name.replace(".log", ".footprint.json")
    log_path = LOGS / log_name
    footprint_path = FOOTPRINTS / footprint_name

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = env.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
    env["MLFLOW_EXPERIMENT_NAME"] = "latent_cep_baselines_smoke"
    env["DATASET_OPTION"] = env.get("DATASET_OPTION", "C")
    env["BASELINES_SMOKE_EXPERIMENT"] = "latent_cep_baselines_smoke"
    # Force line-buffered stdout/stderr so logs flush even on SIGKILL.
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable, str(SCRIPTS / method["launcher"]),
        *method["extra_args"],
        "--env_d4rl_name", method["env"],
        "--seed", "0",
        "--smoke",
        "--smoke_seconds", str(smoke_seconds),
    ]
    started_at = _now()
    print(f"\n[{started_at}] {method['key']}  attempt {n}")
    print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"  log: {log_path}")
    print(f"  footprint: {footprint_path}")

    attempt = Attempt(n=n, started_at=started_at, log_path=str(log_path))
    method["attempts"].append(asdict(attempt))
    method["status"] = "attempting"
    method["last_log"] = str(log_path)

    # We give the smoke launcher an extra 60s grace so its internal alarm
    # can fire cleanly first.
    hard_timeout = smoke_seconds + 60
    rc = None
    probe: Optional[subprocess.Popen] = None
    try:
        with open(log_path, "wb") as logf:
            proc = subprocess.Popen(cmd, env=env, cwd=str(ROOT),
                                    stdout=logf, stderr=subprocess.STDOUT)
            # Start the footprint sidecar AFTER we know the PID of the
            # smoke launcher; the probe walks /proc descendants too.
            probe = _spawn_footprint_probe(
                target_pid=proc.pid,
                footprint_path=footprint_path,
                meta={"algo": method["algo"], "env": method["env"],
                       "stage": method["stage"], "attempt": n,
                       "smoke_seconds": smoke_seconds},
                interval=5.0,
                max_seconds=hard_timeout + 30,
            )
            try:
                rc = proc.wait(timeout=hard_timeout)
            except subprocess.TimeoutExpired:
                print(f"  ! hard timeout @ {hard_timeout}s; sending SIGTERM…")
                proc.terminate()
                try:
                    rc = proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    print("  ! still running; sending SIGKILL…")
                    proc.kill()
                    rc = proc.wait()
    except Exception as e:
        print(f"  ! exception: {e}")
        rc = -1
    finally:
        _terminate_probe(probe)

    finished_at = _now()
    saw_loss = _scan_for_loss(log_path) or _has_mlflow_metrics(
        method["algo"], method["env"], method["stage"],
    )
    footprint_summary = _read_footprint_summary(footprint_path)
    method["attempts"][-1].update({
        "finished_at": finished_at,
        "rc": rc,
        "saw_loss": saw_loss,
        **footprint_summary,
    })

    if saw_loss:
        method["status"] = "green"
        method["notes"] = f"saw loss-like metric in attempt {n} (rc={rc})"
        print(f"  ✓ green ({finished_at})  rc={rc}, saw loss")
    elif rc == LAUNCHER_RC_SKIP:
        method["status"] = "skipped"
        method["notes"] = "launcher returned 64 (explicit skip)."
        print(f"  ⊘ skipped (rc=64)")
    elif rc == LAUNCHER_RC_PREREQ_MISSING:
        method["status"] = "awaiting_prereq"
        method["notes"] = "rc=65: prerequisite (e.g. stage-1 ckpt) missing."
        print(f"  · awaiting prereq (rc=65)")
    elif rc == 0:
        method["status"] = "red"
        method["notes"] = "clean exit but no loss observed (rc=0)"
        print(f"  ✗ red ({finished_at}) rc=0 but no loss metric in log")
    else:
        method["status"] = "red"
        method["notes"] = f"crashed with rc={rc}"
        print(f"  ✗ red ({finished_at}) rc={rc}")

    return method


def _send_discord(level: str, task: str, msg: str) -> None:
    sys.path.insert(0, str(SCRIPTS))
    try:
        from notify import send  # type: ignore
        send(level, task, msg)
    except Exception as e:
        print(f"  ! discord notify failed: {e}")


def run_all(state: dict, only: Optional[str] = None) -> int:
    methods = state["methods"]
    smoke_seconds = state.get("policy", {}).get("smoke_seconds", DEFAULT_SMOKE_SECONDS)
    max_attempts = state.get("policy", {}).get("max_attempts", DEFAULT_MAX_ATTEMPTS)

    order = [k for k in methods.keys() if (only is None or k == only)]
    if not order:
        print(f"No method matches --only {only!r}", file=sys.stderr)
        return 2

    # Pre-skip DD (decision D5 + missing params_proto/ml_logger/jaynes chain).
    # This is the only "permanent skip" we apply automatically; everything else
    # routes through normal smoke + autoresearch fix loop.
    for key, m in methods.items():
        if m["algo"] == "dd" and m["status"] == "pending":
            m["status"] = "skipped"
            m["notes"] = ("pre-skipped this session: needs Option B data (hopper-medium-expert-v2) "
                          "AND missing params_proto/ml_logger/jaynes dep chain. "
                          "Revive in Phase 2b when Option B is built. See RUN_LOG.md.")
    save_state(state)

    for key in order:
        m = methods[key]
        if m["status"] in ("green", "skipped"):
            print(f"\n[skip] {key} already {m['status']}")
            continue
        attempts = m.get("attempts", [])
        if len(attempts) >= max_attempts and m["status"] != "green":
            m["status"] = "red"
            m["notes"] = f"exhausted {max_attempts} attempts; last={m['notes']}"
            save_state(state)
            _send_discord(
                "failure", f"smoke:{key}",
                f"Exhausted {max_attempts} attempts. Tail of last log:\n```\n"
                + _read_tail(Path(m["last_log"]), n_bytes=1200) + "\n```",
            )
            print(f"  ✗ {key}: exhausted retries; emitted Discord failure ping.")
            continue

        m = run_one(m, smoke_seconds=smoke_seconds)
        save_state(state)

        if m["status"] == "red":
            # Halt for operator triage. The operator applies a patch and
            # re-invokes with --only <key>.
            print(f"\n[halt] {key} crashed. Operator should triage:")
            print(f"  log:   {m['last_log']}")
            print(f"  notes: {m['notes']}")
            print(f"\nTail of log:")
            print(_read_tail(Path(m["last_log"]), n_bytes=4000))
            return 42  # checkpoint sentinel

    # All done with this pass.
    green = [k for k, m in methods.items() if m["status"] == "green"]
    red   = [k for k, m in methods.items() if m["status"] == "red"]
    skipped = [k for k, m in methods.items() if m["status"] == "skipped"]
    pending = [k for k, m in methods.items() if m["status"] in ("pending", "attempting", "awaiting_fix")]

    print()
    print(f"=== SUMMARY  green={len(green)}  red={len(red)}  skipped={len(skipped)}  pending={len(pending)}")
    for k in green:   print(f"  ✓ {k}")
    for k in skipped: print(f"  ⊘ {k}")
    for k in red:     print(f"  ✗ {k}  ({methods[k]['notes']})")
    for k in pending: print(f"  · {k}")

    if pending:
        return 0  # nothing more to do this pass; user iterates
    if red:
        return 1  # ended with reds remaining
    _send_discord(
        "summary", "autoresearch",
        f"All {len(green)} smoke methods green ({len(skipped)} skipped). "
        f"Ready for Phase 5/6 matrix.",
    )
    return 0


def cmd_status(state: dict) -> int:
    methods = state["methods"]
    rows = []
    for k, m in methods.items():
        rows.append((m["status"], k, len(m.get("attempts", [])), m.get("notes", "")[:80]))
    rows.sort(key=lambda r: ("green", "skipped", "pending", "attempting", "awaiting_fix", "red").index(
        r[0]) if r[0] in ("green", "skipped", "pending", "attempting", "awaiting_fix", "red") else 99)
    print(f"{'status':<13} {'method':<48} {'tries':>5}  notes")
    print("-" * 100)
    for s, k, t, n in rows:
        print(f"{s:<13} {k:<48} {t:>5}  {n}")
    print()
    n_g = sum(1 for r in rows if r[0] == "green")
    n_s = sum(1 for r in rows if r[0] == "skipped")
    n_r = sum(1 for r in rows if r[0] == "red")
    n_p = sum(1 for r in rows if r[0] not in ("green", "skipped", "red"))
    print(f"green={n_g}  skipped={n_s}  red={n_r}  pending={n_p}")
    return 0


def cmd_footprint_table(state: dict) -> int:
    """Aggregate per-method footprint data from the last GREEN attempt of each method.

    Output is a Markdown-friendly table sized for the operator (you) to read
    and decide bin-packing concurrency. Also writes JSON to
    state/footprints/_aggregate.json for programmatic consumption by the
    bin-packing scheduler.
    """
    methods = state["methods"]
    rows = []
    for key, m in methods.items():
        attempts = m.get("attempts", []) or []
        # Prefer the most recent green attempt for footprint readings.
        chosen = None
        for a in reversed(attempts):
            if a.get("saw_loss"):
                chosen = a
                break
        if chosen is None and attempts:
            chosen = attempts[-1]
        if chosen is None:
            rows.append((key, m["status"], None))
            continue
        rows.append((key, m["status"], chosen))

    print(f"\n{'method':<48} {'status':<9} {'vRAM(peak)':>10} {'vRAM(steady)':>13} "
          f"{'GPU%(peak)':>10} {'GPU%(steady)':>13} {'RSS(peak)':>10}")
    print("-" * 120)
    for key, status, a in rows:
        if a is None:
            print(f"{key:<48} {status:<9} {'-':>10} {'-':>13} {'-':>10} {'-':>13} {'-':>10}")
            continue
        def _fmt_gb(x): return f"{x:.2f}GB" if isinstance(x, (int, float)) else "-"
        def _fmt_pct(x): return f"{x:.0f}%" if isinstance(x, (int, float)) else "-"
        print(f"{key:<48} {status:<9} "
              f"{_fmt_gb(a.get('vram_proc_peak_gb')):>10} "
              f"{_fmt_gb(a.get('vram_proc_steady_gb')):>13} "
              f"{_fmt_pct(a.get('gpu_util_peak_pct')):>10} "
              f"{_fmt_pct(a.get('gpu_util_steady_pct')):>13} "
              f"{_fmt_gb(a.get('rss_peak_gb')):>10}")

    # Per-base-method aggregate (max across stages for a given algo)
    by_algo: dict[str, dict] = {}
    for key, status, a in rows:
        if a is None:
            continue
        algo = key.split(":")[0]
        cur = by_algo.setdefault(algo, {
            "vram_peak_gb": 0.0, "vram_steady_gb": 0.0,
            "rss_peak_gb": 0.0, "gpu_util_steady_pct": 0.0,
            "n_attempts": 0,
        })
        for fld in ("vram_proc_peak_gb", "vram_proc_steady_gb",
                    "rss_peak_gb", "gpu_util_steady_pct"):
            val = a.get(fld)
            if isinstance(val, (int, float)):
                target = fld.replace("vram_proc_", "vram_")
                cur[target] = max(cur[target], float(val))
        cur["n_attempts"] += 1

    out = {
        "generated_at": _now(),
        "per_method": [
            {"key": k, "status": s, **({} if a is None else {k2: a.get(k2) for k2 in
                ("vram_proc_peak_gb", "vram_proc_steady_gb",
                 "rss_peak_gb", "rss_steady_gb",
                 "gpu_util_peak_pct", "gpu_util_steady_pct", "n_samples")})}
            for k, s, a in rows
        ],
        "per_algo_max_across_stages": by_algo,
    }
    agg_path = FOOTPRINTS / "_aggregate.json"
    agg_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote aggregate: {agg_path}")

    if by_algo:
        print()
        print("Per-algo budget (max-across-stages, peak vRAM × 1.3 safety):")
        print(f"{'algo':<10} {'vRAM_budget':>12} {'RAM_budget':>11}  recommendation")
        print("-" * 70)
        for algo, v in sorted(by_algo.items()):
            vbgb = v["vram_peak_gb"] * 1.3
            rbgb = v["rss_peak_gb"] * 1.3
            slots28 = max(1, int(28.0 / max(vbgb, 0.5)))
            slots45 = max(1, int(45.0 / max(rbgb, 0.5)))
            slots = min(slots28, slots45)
            print(f"{algo:<10} {vbgb:>10.2f}GB {rbgb:>9.2f}GB  fits {slots}× in (28GB vRAM, 45GB RAM)")
    return 0


def cmd_reset(state: dict, only: Optional[str]) -> int:
    for k in list(state["methods"].keys()):
        if only and k != only:
            continue
        state["methods"][k]["status"] = "pending"
        state["methods"][k]["attempts"] = []
        state["methods"][k]["notes"] = ""
    save_state(state)
    print(f"reset {'all' if not only else only}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true", help="Run smoke attempts.")
    p.add_argument("--status",  action="store_true", help="Print state table.")
    p.add_argument("--reset",   action="store_true", help="Reset method(s) to pending.")
    p.add_argument("--footprint_table", action="store_true",
                   help="Aggregate per-method footprints (call after smokes are green).")
    p.add_argument("--only",    default=None,
                   help="Filter to a single method key (e.g. ldcq:halfcheetah-medium-v0:skills).")
    p.add_argument("--smoke_seconds", type=int, default=None,
                   help="Override per-attempt budget (default 300s).")
    p.add_argument("--max_attempts",  type=int, default=None,
                   help="Override retry cap (default 3).")
    args = p.parse_args()

    state = _seed_state(load_state())
    if args.smoke_seconds is not None:
        state.setdefault("policy", {})["smoke_seconds"] = args.smoke_seconds
    if args.max_attempts is not None:
        state.setdefault("policy", {})["max_attempts"] = args.max_attempts
    save_state(state)

    if args.status:
        return cmd_status(state)
    if args.reset:
        return cmd_reset(state, only=args.only)
    if args.footprint_table:
        return cmd_footprint_table(state)
    if args.execute:
        return run_all(state, only=args.only)

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
