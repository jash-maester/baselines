"""Shared bootstrap for every per-method launcher.

Provides:
  - PYTHONPATH wiring for compat-shim (handled by editable install, here for safety)
  - wandb_stub installation (replaces real wandb)
  - MLflow start_run + tagging
  - repo SHA capture
  - argparse helper
  - in-process timeout helper for smokes
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = ROOT / "repos"
COMPAT_SRC = ROOT / "compat" / "src"

# Ensure compat is on path even if editable install was bypassed
sys.path.insert(0, str(COMPAT_SRC))


def repo_sha(repo_path: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def parse_common_args(extra_args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--algo", required=True)
    p.add_argument("--env_d4rl_name", required=True,
                   help="D4RL-style name, e.g. halfcheetah-medium-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default="full")
    p.add_argument("--smoke", action="store_true",
                   help="Truncate training to a few hundred steps.")
    p.add_argument("--smoke_seconds", type=int, default=180,
                   help="Wall-clock cap for an in-process smoke (default 180s).")
    p.add_argument("--dataset_option", default=os.environ.get("DATASET_OPTION", "C"))
    if extra_args:
        for name, kw in extra_args:
            p.add_argument(name, **kw)
    return p.parse_args()


def env_to_minari(d4rl_name: str) -> str:
    from compat_utils.minari_mapping import D4RL_NAME_TO_MINARI_ID
    return D4RL_NAME_TO_MINARI_ID[d4rl_name]


def split_env_dataset(d4rl_name: str) -> tuple[str, str]:
    """'halfcheetah-medium-v0' -> ('halfcheetah', 'medium')."""
    base, _v = d4rl_name.rsplit("-v", 1)
    parts = base.split("-")
    env = parts[0]
    dataset = "-".join(parts[1:]) if len(parts) > 1 else "default"
    return env, dataset


def install_wandb_stub():
    from compat_utils import wandb_stub
    wandb_stub.install()


def install_tb_shim():
    """Redirect torch.utils.tensorboard.SummaryWriter scalars to MLflow.

    For repos (QGPO/CEP, Diffusion-QL) that log via TensorBoard rather than
    wandb. Call after install_wandb_stub(), before the repo import runs.
    """
    from compat_utils import tb_shim
    tb_shim.install()


def mlflow_start(algo, env, dataset, seed, stage, repo_url, repo_path,
                 method_family, minari_id, smoke=False, extra_config=None,
                 extra_tags=None):
    from compat_utils.mlflow_helper import start_run
    cfg = {
        "minari_id": minari_id,
        "repo_url": repo_url,
        "repo_sha": repo_sha(repo_path),
        "method_family": method_family,
        "smoke": bool(smoke),
        "dataset_option": os.environ.get("DATASET_OPTION", "C"),
    }
    if extra_config:
        cfg.update(extra_config)
    # Log the normalization reference scores so every normalized score is
    # losslessly invertible to raw later (and re-normalizable to D4RL refs).
    # Also hand them to the metric shims so they emit `raw_return` alongside
    # `d4rl_normalized_score` for every method.
    tags = dict(extra_tags or {})
    try:
        from compat_utils.env_factory import _minari_ref_scores
        ref_min, ref_max = _minari_ref_scores(f"{env}-{dataset}-v0")
        tags.update({"ref_min": ref_min, "ref_max": ref_max,
                     "ref_source": "minari_expert_mean", "norm_scheme": "minari_v0"})
        from compat_utils import wandb_stub, tb_shim
        wandb_stub.set_refs(ref_min, ref_max)
        tb_shim.set_refs(ref_min, ref_max)
    except Exception:
        pass
    return start_run(algo, env, dataset, seed, stage, cfg, extra_tags=tags)


class TimeoutSentinel(Exception):
    pass


@contextmanager
def alarm_timeout(seconds: int):
    """Raise TimeoutSentinel after `seconds` via SIGALRM.

    Use to bound smoke runs whose target scripts hard-code epoch counts.
    Catches `KeyboardInterrupt` and `TimeoutSentinel` so the parent can end
    the MLflow run cleanly.
    """
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutSentinel(f"smoke timeout hit after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def end_mlflow_run(status: str = "FINISHED") -> None:
    try:
        import mlflow
        mlflow.end_run(status=status)
    except Exception:
        pass
