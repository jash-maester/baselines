"""Health monitor for the long matrix run — Discord alerts on failure.

Complements long_runner.py (which already pings per red cell). This watchdog
catches the failure modes long_runner can't report itself:

  - orchestrator_down : state/long_runner.pid present but the process is dead
                        while cells remain pending (i.e. it crashed mid-run).
  - tunnel_down       : http://localhost:5555/health unreachable (MLflow tunnel
                        down -> cells will start failing).
  - gpu_idle_stuck    : cells marked running but GPU utilization ~0% for a
                        sustained window (likely hung).
  - matrix_complete   : no pending/running left -> one summary ping.
  - heartbeat         : a low-frequency "still alive" summary (default daily).

Alerts are de-duplicated/throttled via state/monitor_state.json: each incident
pings once on entry, re-pings at most hourly while still bad, and sends a
recovery ping when it clears. Failures-only policy is respected (heartbeat is
infrequent and clearly tagged).

Run detached:
    setsid bash -c '.venv/bin/python scripts/matrix_monitor.py' \
        </dev/null >logs/matrix_monitor.out 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
STATE_MATRIX = ROOT / "state" / "matrix.json"
PID_FILE = ROOT / "state" / "long_runner.pid"
MON_STATE = ROOT / "state" / "monitor_state.json"

sys.path.insert(0, str(SCRIPTS))
try:
    from notify import send as _discord_send
except Exception:                                   # pragma: no cover
    def _discord_send(level, task, msg):
        print(f"[monitor] (notify unavailable) {level} {task}: {msg}", file=sys.stderr)
        return False

# --- tunables (env-overridable) ---
CHECK_INTERVAL   = int(os.environ.get("MONITOR_INTERVAL_SEC", "120"))
MLFLOW_URI       = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
HEARTBEAT_SEC    = int(os.environ.get("MONITOR_HEARTBEAT_SEC", str(24 * 3600)))
REALERT_SEC      = int(os.environ.get("MONITOR_REALERT_SEC", "3600"))
GPU_IDLE_PCT     = float(os.environ.get("MONITOR_GPU_IDLE_PCT", "5"))
GPU_IDLE_SEC     = int(os.environ.get("MONITOR_GPU_IDLE_SEC", "1800"))   # 30 min
TUNNEL_GRACE_SEC = int(os.environ.get("MONITOR_TUNNEL_GRACE_SEC", "90")) # systemd auto-heal window


def _now() -> float:
    return time.time()


def _iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _load(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _save_state(s: dict) -> None:
    tmp = MON_STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2))
    tmp.replace(MON_STATE)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, ValueError, TypeError):
        return False


def _orchestrator_state() -> tuple[str, int | None]:
    """Returns (state, pid). state ∈ {alive, dead, absent}."""
    if not PID_FILE.exists():
        return ("absent", None)
    try:
        pid = int(PID_FILE.read_text().strip())
    except Exception:
        return ("absent", None)
    return ("alive" if _pid_alive(pid) else "dead", pid)


def _matrix_counts() -> dict:
    d = _load(STATE_MATRIX, {"rows": []})
    rows = d.get("rows", [])
    c = {"green": 0, "running": 0, "pending": 0, "red": 0, "skipped": 0,
         "awaiting_prereq": 0, "total": len(rows)}
    for r in rows:
        c[r.get("status", "?")] = c.get(r.get("status", "?"), 0) + 1
    return c


def _tunnel_ok() -> bool:
    try:
        req = urllib.request.Request(f"{MLFLOW_URI}/health",
                                     headers={"User-Agent": "matrix-monitor"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _gpu_util() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=8).decode().strip().splitlines()
        return max(float(x) for x in out) if out else None
    except Exception:
        return None


def _completed(c: dict) -> int:
    return c["green"] + c["red"] + c["skipped"] + c["awaiting_prereq"]


def _summary_line(c: dict) -> str:
    return (f"green={c['green']} running={c['running']} pending={c['pending']} "
            f"red={c['red']} skipped={c['skipped']} / total={c['total']}")


def _should_alert(st: dict, key: str) -> bool:
    """First entry, or re-alert after REALERT_SEC while still active."""
    last = st.get("alerts", {}).get(key, 0)
    return (_now() - last) >= REALERT_SEC


def _mark_alert(st: dict, key: str) -> None:
    st.setdefault("alerts", {})[key] = _now()


def _clear_alert(st: dict, key: str) -> bool:
    """Returns True if the alert was previously active (=> send recovery)."""
    if key in st.get("alerts", {}):
        del st["alerts"][key]
        return True
    return False


def main() -> int:
    st = _load(MON_STATE, {})
    st.setdefault("alerts", {})
    st.setdefault("last_heartbeat", 0)
    st.setdefault("gpu_idle_since", None)
    st.setdefault("matrix_done_pinged", False)
    print(f"[monitor] started {_iso()}  interval={CHECK_INTERVAL}s  uri={MLFLOW_URI}")

    while True:
        try:
            c = _matrix_counts()
            orch, pid = _orchestrator_state()
            tunnel = _tunnel_ok()
            gpu = _gpu_util()
            active = c["running"] > 0 or c["pending"] > 0

            # 1) tunnel / MLflow (allow systemd a grace window before crying wolf)
            if not tunnel:
                # confirm sustained outage past the grace window
                t0 = st.get("tunnel_down_since")
                if t0 is None:
                    st["tunnel_down_since"] = _now()
                elif (_now() - t0) >= TUNNEL_GRACE_SEC and _should_alert(st, "tunnel_down"):
                    _discord_send("failure", "matrix/tunnel",
                                  f"MLflow at {MLFLOW_URI} unreachable for "
                                  f">{TUNNEL_GRACE_SEC}s. Cells may fail to log/start. "
                                  f"Check the systemd tunnel: `systemctl --user status mlflow-tunnel`.")
                    _mark_alert(st, "tunnel_down")
            else:
                st["tunnel_down_since"] = None
                if _clear_alert(st, "tunnel_down"):
                    _discord_send("info", "matrix/tunnel", "MLflow tunnel recovered (localhost:5555 healthy again).")

            # 2) orchestrator died mid-run (pid file present but process dead, work remains)
            if orch == "dead" and c["pending"] > 0:
                if _should_alert(st, "orchestrator_down"):
                    _discord_send("failure", "matrix/orchestrator",
                                  f"long_runner (pid {pid}) is DEAD with {c['pending']} cells "
                                  f"still pending. Matrix is NOT progressing. "
                                  f"Resume: `cd {ROOT} && make matrix`.\n{_summary_line(c)}")
                    _mark_alert(st, "orchestrator_down")
            elif orch == "alive":
                if _clear_alert(st, "orchestrator_down"):
                    _discord_send("info", "matrix/orchestrator", "long_runner is back up and scheduling.")

            # 3) matrix completion (no pending, no running) -> one summary
            if c["total"] > 0 and c["pending"] == 0 and c["running"] == 0:
                if not st.get("matrix_done_pinged"):
                    _discord_send("summary", "matrix/complete",
                                  f"Matrix finished. {_summary_line(c)}. "
                                  f"Run `make status` for the compare table.")
                    st["matrix_done_pinged"] = True
            else:
                st["matrix_done_pinged"] = False

            # 4) GPU-idle stall (cells running but GPU ~idle for a sustained window)
            if c["running"] > 0 and gpu is not None and gpu < GPU_IDLE_PCT and tunnel and orch != "dead":
                since = st.get("gpu_idle_since")
                if since is None:
                    st["gpu_idle_since"] = _now()
                elif (_now() - since) >= GPU_IDLE_SEC and _should_alert(st, "gpu_idle"):
                    mins = int((_now() - since) / 60)
                    _discord_send("failure", "matrix/gpu",
                                  f"GPU util <{GPU_IDLE_PCT:.0f}% for ~{mins} min while "
                                  f"{c['running']} cells are 'running' — possible hang. "
                                  f"Check `make status` / cell logs in logs/matrix/.")
                    _mark_alert(st, "gpu_idle")
            else:
                if st.get("gpu_idle_since") is not None:
                    st["gpu_idle_since"] = None
                    _clear_alert(st, "gpu_idle")

            # 5) heartbeat (low frequency; reassurance the monitor is alive)
            if active and (_now() - st.get("last_heartbeat", 0)) >= HEARTBEAT_SEC:
                _discord_send("heartbeat", "matrix",
                              f"alive @ {_iso()} — {_summary_line(c)} | "
                              f"orchestrator={orch} tunnel={'OK' if tunnel else 'DOWN'} "
                              f"gpu={gpu if gpu is not None else '?'}%")
                st["last_heartbeat"] = _now()

            _save_state(st)
        except Exception as e:                       # never let the monitor die silently
            print(f"[monitor] check error: {e}", file=sys.stderr)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
