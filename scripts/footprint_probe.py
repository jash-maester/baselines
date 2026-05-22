"""Sidecar process that samples resource footprints during a smoke run.

Spawned by `scripts/autoresearch_smoke.py` once the smoke launcher's PID is
known. Polls nvidia-smi + /proc/<pid>/status at a configurable interval and
records the max-observed (vRAM, GPU utilization, system RSS) plus a steady-state
estimate (median of the last 80% of samples). Writes a JSON file at exit.

Designed to be cheap (just subprocess calls + file I/O; no torch / no cuda).

The probe stops automatically when:
  - the target PID disappears, or
  - it receives SIGTERM/SIGINT, or
  - --max_seconds elapses.

Usage:
    python scripts/footprint_probe.py --target_pid 12345 \\
        --out_path state/footprints/ldcq_halfcheetah-medium-v0_skills_attempt1.json \\
        --interval 5 --max_seconds 600
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _descendants(root_pid: int) -> set[int]:
    """All descendant PIDs of root_pid (best-effort, single read of /proc)."""
    out = {root_pid}
    try:
        proc_dirs = [p for p in os.listdir("/proc") if p.isdigit()]
    except FileNotFoundError:
        return out
    # Build child->parent map
    child_to_parent: dict[int, int] = {}
    for d in proc_dirs:
        try:
            with open(f"/proc/{d}/status", "r") as f:
                ppid = None
                for line in f:
                    if line.startswith("PPid:"):
                        ppid = int(line.split()[1])
                        break
            if ppid is not None:
                child_to_parent[int(d)] = ppid
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    # BFS down from root_pid
    frontier = {root_pid}
    while frontier:
        nxt = set()
        for child, parent in child_to_parent.items():
            if parent in frontier and child not in out:
                nxt.add(child)
                out.add(child)
        frontier = nxt
    return out


def _query_compute_apps() -> dict[int, int]:
    """Return {pid: vram_mib} from nvidia-smi compute-apps; empty if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode()
    except Exception:
        return {}
    res = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                res[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return res


def _query_gpu_util() -> tuple[int, int] | None:
    """Return (gpu_util_pct, memory_used_mib) for GPU 0."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits",
             "--id=0"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        return None
    return None


def _proc_rss_kb(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # KB
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def _summary(samples: list[float], suffix: str = "") -> dict:
    if not samples:
        return {f"{suffix}n": 0}
    return {
        f"{suffix}n":       len(samples),
        f"{suffix}peak":    max(samples),
        f"{suffix}mean":    statistics.fmean(samples),
        f"{suffix}median":  statistics.median(samples),
        f"{suffix}p95":     sorted(samples)[int(0.95 * (len(samples) - 1))],
        f"{suffix}steady":  statistics.median(samples[max(0, int(0.2 * len(samples))):]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target_pid", type=int, required=True)
    p.add_argument("--out_path", type=Path, required=True)
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--max_seconds", type=int, default=900)
    p.add_argument("--meta", type=str, default="{}",
                   help="JSON object to embed in the output (algo/env/stage tags).")
    args = p.parse_args()

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        meta = json.loads(args.meta)
        if not isinstance(meta, dict):
            meta = {}
    except json.JSONDecodeError:
        meta = {}

    # Sample buffers
    vram_per_proc_mib: list[float] = []   # sum across target's descendant PIDs in compute-apps
    vram_per_proc_breakdown: dict[int, list[int]] = {}
    gpu_util_pct: list[float] = []
    gpu_mem_total_mib: list[float] = []   # whole-GPU memory.used
    rss_total_kb: list[float] = []

    started_at = _now()
    t0 = time.time()
    stop = {"flag": False}

    def _handler(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)

    n_samples = 0
    while not stop["flag"]:
        if not _alive(args.target_pid):
            break
        if time.time() - t0 > args.max_seconds:
            break

        desc = _descendants(args.target_pid)
        apps = _query_compute_apps()
        sum_this_pid = 0
        for pid in desc:
            if pid in apps:
                sum_this_pid += apps[pid]
                vram_per_proc_breakdown.setdefault(pid, []).append(apps[pid])
        vram_per_proc_mib.append(sum_this_pid)

        util = _query_gpu_util()
        if util is not None:
            gpu_util_pct.append(util[0])
            gpu_mem_total_mib.append(util[1])

        total_rss = 0
        for pid in desc:
            r = _proc_rss_kb(pid)
            if r:
                total_rss += r
        rss_total_kb.append(total_rss)

        n_samples += 1
        time.sleep(args.interval)

    ended_at = _now()
    result = {
        "version": 1,
        "target_pid": args.target_pid,
        "started_at": started_at,
        "ended_at":   ended_at,
        "interval_sec": args.interval,
        "n_samples":  n_samples,
        "meta": meta,
        "vram_proc_mib":   _summary(vram_per_proc_mib),
        "vram_proc_gb":    _summary([m / 1024.0 for m in vram_per_proc_mib]),
        "vram_gpu_total_mib": _summary(gpu_mem_total_mib),
        "gpu_util_pct":    _summary(gpu_util_pct),
        "rss_total_kb":    _summary(rss_total_kb),
        "rss_total_gb":    _summary([k / (1024.0 ** 2) for k in rss_total_kb]),
        "vram_breakdown_by_pid": {
            str(pid): {"peak_mib": max(vs), "n": len(vs)}
            for pid, vs in vram_per_proc_breakdown.items()
        },
    }
    args.out_path.write_text(json.dumps(result, indent=2))
    print(f"[footprint_probe] wrote {args.out_path}  samples={n_samples}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
