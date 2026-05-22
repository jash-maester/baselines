"""Discord webhook notifier for baselines orchestration.

User policy (2026-05-19):
  - Autoresearch smoke loop: ONE summary ping when all 7 repos finish their
    5-min smoke (green or red). On a per-method exhausted-retries failure,
    emit an early failure ping so the operator can intervene.
  - Long-run matrix: failures only. No start/completion pings.

Webhook is hardcoded; override via $BASELINES_DISCORD_WEBHOOK env var if needed.

Usage:
    python scripts/notify.py --level failure --task "ldcq:skills" \\
        --msg "exhausted 3 retries, last error: KeyError 'observations'"
    python scripts/notify.py --level summary --task "autoresearch" \\
        --msg "7/7 repos green in 38 min"

Importable:
    from scripts.notify import send_failure, send_summary
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_WEBHOOK = (
    "https://discord.com/api/webhooks/"
    "1506326854145933505/"
    "o8EQZWg-1z_ycffFplEg5_NF50YJd5g_y9xksW-x5if5sECX9Uu0P7dFJw-24EOLcQjQ"
)


def _webhook_url() -> str:
    return os.environ.get("BASELINES_DISCORD_WEBHOOK", DEFAULT_WEBHOOK)


def _format(content: str) -> str:
    # Discord caps a single message at 2000 chars. Trim with a tail marker.
    if len(content) <= 1900:
        return content
    return content[:1900] + "\n…(truncated)"


_USER_AGENT = "baselines-orchestrator (https://github.com/anthropics/claude-code, 1.0)"


def _post(content: str, retries: int = 3, backoff: float = 1.5) -> bool:
    payload = json.dumps({"content": _format(content)}).encode("utf-8")
    url = _webhook_url()
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={
                    "Content-Type": "application/json",
                    # Discord 403s the default Python-urllib UA.
                    "User-Agent": _USER_AGENT,
                },
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return True
                print(f"[notify] webhook returned {resp.status}", file=sys.stderr)
        except urllib.error.HTTPError as e:
            # Discord rate-limit: 429 with Retry-After header
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                try:
                    delay = float(ra) if ra else backoff ** attempt
                except ValueError:
                    delay = backoff ** attempt
                print(f"[notify] rate-limited; sleeping {delay:.1f}s", file=sys.stderr)
                time.sleep(delay)
                continue
            print(f"[notify] attempt {attempt}/{retries} failed: HTTP {e.code} {e.reason}",
                  file=sys.stderr)
        except (urllib.error.URLError, socket.timeout) as e:
            print(f"[notify] attempt {attempt}/{retries} failed: {e}", file=sys.stderr)
        time.sleep(backoff ** attempt)
    return False


def _prefix(level: str) -> str:
    return {
        "failure":  ":boom: **FAILURE**",
        "summary":  ":checkered_flag: **SUMMARY**",
        "info":     ":information_source: **INFO**",
        "heartbeat": ":heartbeat: **HEARTBEAT**",
    }.get(level, "**NOTICE**")


def _host_tag() -> str:
    return f"`{socket.gethostname()}`"


def send(level: str, task: str, msg: str) -> bool:
    """Generic send. `level` ∈ {failure, summary, info, heartbeat}.

    Per policy, only `failure` and `summary` levels are emitted during normal
    operation; `info`/`heartbeat` are available for explicit debugging.
    """
    body = (
        f"{_prefix(level)}  •  `{task}`  •  {_host_tag()}\n"
        f"{msg}"
    )
    ok = _post(body)
    if not ok:
        print(f"[notify] giving up after retries: {task!r} / {msg!r}", file=sys.stderr)
    return ok


def send_failure(task: str, msg: str) -> bool:
    return send("failure", task, msg)


def send_summary(task: str, msg: str) -> bool:
    return send("summary", task, msg)


def send_info(task: str, msg: str) -> bool:
    return send("info", task, msg)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--level", required=True,
                   choices=["failure", "summary", "info", "heartbeat"])
    p.add_argument("--task", required=True)
    p.add_argument("--msg", required=True)
    args = p.parse_args()
    ok = send(args.level, args.task, args.msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
