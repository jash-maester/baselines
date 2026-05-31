"""OOM-resilience watchdog for the baselines matrix.  (you launch this yourself)

What long_runner.py ALREADY does (so this watchdog need not duplicate it):
  - bin-packs cells by measured vRAM/RAM footprint with OOM-safety floors,
  - marks any killed/crashed cell `red`, Discord-pings it, and CONTINUES with
    the next cells (red cells are not retried -> effectively skipped).

What THIS watchdog adds:
  1. OOM attribution — classifies a killed cell as system-RAM OOM (SIGKILL /
     rc -9|137 or "Killed" in log) or CUDA/VRAM OOM ("CUDA out of memory" in
     log), and Discord-pings with that context (long_runner only says rc=...).
     Non-OOM crashes are left to long_runner's own ping (no duplicate spam).
  2. Continuity — if the OOM killer kills the long_runner ORCHESTRATOR itself
     (rare; it's tiny, the big training cell is usually killed first), auto-
     resume it so the *next* baselines keep running. Before resuming it skips
     any "poison" cell already attempted >= MAX_ATTEMPTS (marks it red), and
     it lowers concurrency on repeated restarts to relieve memory pressure.
  3. Notes — every OOM/kill it sees is recorded in state/oom_watchdog.json.

No root required: uses cell exit codes + cell logs (kernel dmesg is restricted
on this host). Safe to run alongside matrix_monitor.py; it only writes
state/matrix.json when the orchestrator is DEAD (no race with a live scheduler).

Launch (example):
    setsid bash -c '/home/jash/baselines/.venv/bin/python \
        /home/jash/baselines/scripts/oom_watchdog.py' \
        </dev/null >logs/oom_watchdog.out 2>&1 &
Or as a systemd --user service: run with `--print-unit` to get the unit file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STATE_MATRIX = ROOT / "state" / "matrix.json"
PID_FILE = ROOT / "state" / "long_runner.pid"
WD_STATE = ROOT / "state" / "oom_watchdog.json"
MATRIX_LOGS = ROOT / "logs" / "matrix"

sys.path.insert(0, str(SCRIPTS))
try:
    from notify import send as _discord_send
except Exception:                                    # pragma: no cover
    def _discord_send(level, task, msg):
        print(f"[oom-wd] (notify down) {level} {task}: {msg}", file=sys.stderr); return False

POLL_SEC      = int(os.environ.get("OOMWD_POLL_SEC", "60"))
# A cell is "poison" only after THIS watchdog has seen it 'running' at this many
# orchestrator deaths (NOT raw matrix `attempts`, which also grows on benign
# restarts like power-cycles/resumes).
POISON_DEATHS = int(os.environ.get("OOMWD_POISON_DEATHS", "3"))
BASE_CONCURRENCY = int(os.environ.get("OOMWD_BASE_CONCURRENCY", "12"))
MIN_CONCURRENCY  = int(os.environ.get("OOMWD_MIN_CONCURRENCY", "4"))
RESTART_WINDOW_SEC = int(os.environ.get("OOMWD_RESTART_WINDOW_SEC", "1800"))  # 30 min
RESTART_MAX_IN_WINDOW = int(os.environ.get("OOMWD_RESTART_MAX", "4"))


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _save_wd(s: dict) -> None:
    tmp = WD_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2)); tmp.replace(WD_STATE)


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0); return True
    except Exception:
        return False


def _orch_state():
    if not PID_FILE.exists():
        return ("absent", None)
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return ("absent", None)
    return ("alive" if _pid_alive(pid) else "dead", pid)


def _log_tail(key: str, n_bytes: int = 6000) -> str:
    p = MATRIX_LOGS / f"{key}.log"
    if not p.exists():
        return ""
    try:
        sz = p.stat().st_size
        with open(p, "rb") as f:
            if sz > n_bytes:
                f.seek(sz - n_bytes)
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _classify_kill(row: dict) -> tuple[str, str]:
    """Return (kind, detail). kind ∈ {ram_oom, vram_oom, crash}."""
    rc = row.get("rc")
    tail = _log_tail(row["key"]).lower()
    if ("cuda out of memory" in tail or "cublas_status_alloc_failed" in tail
            or ("out of memory" in tail and "cuda" in tail)):
        return ("vram_oom", f"CUDA/VRAM OOM (rc={rc})")
    # rc -9/137 == SIGKILL (the OOM killer's signal); log markers corroborate.
    if rc in (-9, 137) or "killed" in tail or "oom-kill" in tail or "memoryerror" in tail:
        return ("ram_oom", f"RAM/OOM kill (rc={rc})")
    return ("crash", f"rc={rc}")


def _counts(rows) -> dict:
    c = {}
    for r in rows:
        c[r.get("status", "?")] = c.get(r.get("status", "?"), 0) + 1
    return c


def _resume_matrix(concurrency: int) -> bool:
    """Relaunch the scheduler (long_runner's own startup recovers interrupted
    'running' cells back to pending). Uses make so settings stay single-source."""
    try:
        env = os.environ.copy()
        env.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5555")
        r = subprocess.run(["make", "matrix", f"CONCURRENCY={concurrency}"],
                           cwd=str(ROOT), env=env, timeout=60,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(f"[oom-wd] resume rc={r.returncode}: {r.stdout.decode(errors='replace')[-200:]}")
        return r.returncode == 0
    except Exception as e:
        print(f"[oom-wd] resume failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    if "--print-unit" in sys.argv:
        print(f"""[Unit]
Description=Baselines OOM-resilience watchdog
After=network-online.target

[Service]
Type=simple
WorkingDirectory={ROOT}
Environment=MLFLOW_TRACKING_URI=http://localhost:5555
Environment=PATH=/usr/local/bin:/usr/bin:/bin
ExecStart={sys.executable} {ROOT}/scripts/oom_watchdog.py
Restart=always
RestartSec=10

[Install]
WantedBy=default.target""")
        return 0

    st = _load(WD_STATE, {})
    st.setdefault("notified_red", [])          # cell keys already OOM-notified
    st.setdefault("restarts", [])              # timestamps of auto-resumes
    st.setdefault("skipped_poison", [])
    notified = set(st["notified_red"])
    print(f"[oom-wd] started {_iso()} poll={POLL_SEC}s poison_deaths={POISON_DEATHS}")

    while True:
        try:
            d = _load(STATE_MATRIX, {"rows": []})
            rows = d.get("rows", [])
            c = _counts(rows)
            # cells that finished green are not poison; drop their death tally
            dr = st.setdefault("death_running", {})
            for r in rows:
                if r.get("status") == "green" and r["key"] in dr:
                    del dr[r["key"]]

            # 1) classify + notify NEW killed cells (OOM ones only; crashes are
            #    long_runner's to ping). Record notes either way.
            for r in rows:
                if r.get("status") == "red" and r["key"] not in notified:
                    kind, detail = _classify_kill(r)
                    if kind in ("ram_oom", "vram_oom"):
                        label = "system-RAM OOM" if kind == "ram_oom" else "CUDA/VRAM OOM"
                        _discord_send("failure", f"matrix/oom:{r['key']}",
                                      f":warning: {label} — `{r['key']}` ({detail}). "
                                      f"Marked red & SKIPPED; matrix continues with the next cells. "
                                      f"[{c.get('green',0)} done / {c.get('pending',0)} pending]")
                        st.setdefault("notes", []).append(
                            {"ts": _iso(), "key": r["key"], "kind": kind, "detail": detail})
                    notified.add(r["key"])
            st["notified_red"] = sorted(notified)

            # 2) continuity: orchestrator dead while work remains -> auto-resume
            orch, pid = _orch_state()
            if orch == "dead" and c.get("pending", 0) > 0:
                now = time.time()
                st["restarts"] = [t for t in st["restarts"] if now - t < RESTART_WINDOW_SEC]
                if len(st["restarts"]) >= RESTART_MAX_IN_WINDOW:
                    _discord_send("failure", "matrix/oom-watchdog",
                                  f":rotating_light: orchestrator died & auto-resume looped "
                                  f"{len(st['restarts'])}x in {RESTART_WINDOW_SEC//60}min — STOPPING auto-resume. "
                                  f"Manual check needed (likely a systemic OOM). pending={c.get('pending')}.")
                    _save_wd(st); time.sleep(POLL_SEC); continue

                # Count cells that were 'running' at THIS orchestrator death (the
                # victims). A cell present at >= POISON_DEATHS deaths is poison.
                dr = st.setdefault("death_running", {})
                for r in rows:
                    if r.get("status") == "running":
                        dr[r["key"]] = dr.get(r["key"], 0) + 1
                # poison-skip: mark such cells red so the resumed scheduler skips
                # them (state write is safe here: the orchestrator is dead).
                skipped = []
                for r in rows:
                    if r.get("status") in ("running", "pending") and dr.get(r["key"], 0) >= POISON_DEATHS:
                        r["status"] = "red"; r["pid"] = None
                        r["notes"] = (f"auto-skipped: running at {dr[r['key']]} orchestrator "
                                      f"deaths (suspected OOM/poison)")
                        skipped.append(r["key"])
                if skipped:
                    tmp = STATE_MATRIX.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(d, indent=2)); tmp.replace(STATE_MATRIX)
                    st["skipped_poison"] = sorted(set(st["skipped_poison"]) | set(skipped))

                conc = max(MIN_CONCURRENCY, BASE_CONCURRENCY - 2 * len(st["restarts"]))
                ok = _resume_matrix(conc)
                st["restarts"].append(now)
                _discord_send("failure", "matrix/oom-watchdog",
                              f":arrows_counterclockwise: orchestrator was dead (likely OOM-killed) — "
                              f"{'auto-resumed' if ok else 'RESUME FAILED for'} the matrix at concurrency={conc}. "
                              + (f"Skipped {len(skipped)} poison cell(s): {skipped}. " if skipped else "")
                              + f"pending={c.get('pending')}.")

            _save_wd(st)
        except Exception as e:
            print(f"[oom-wd] loop error: {e}", file=sys.stderr)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())
