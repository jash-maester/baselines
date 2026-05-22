"""Single entry-point for MLflow setup, run naming, and tag/param logging.

All baseline launchers go through `start_run(...)` so the experiment, tags, and
parameter logging conventions stay identical across the matrix.
"""
import os
import mlflow

DEFAULT_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
# The MLflow `baselines` experiment is owned by an unrelated project on this
# host (trading backtests). To avoid mixing, the offline-RL reproduction uses
# `latent_cep_baselines` (long matrix) + `latent_cep_baselines_smoke`
# (autoresearch loop). Rename decided 2026-05-19; recorded in RUN_LOG.md.
EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "latent_cep_baselines")


def init():
    mlflow.set_tracking_uri(DEFAULT_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def run_name(algo, env, dataset, seed, stage):
    return f"{algo}_{env}-{dataset}_seed{seed}_{stage}"


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else f"{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def start_run(algo, env, dataset, seed, stage, config, extra_tags=None):
    init()
    name = run_name(algo, env, dataset, seed, stage)
    run = mlflow.start_run(run_name=name)
    tags = {
        "algo": algo,
        "env": env,
        "dataset": dataset,
        "seed": str(seed),
        "stage": stage,
        "minari_id": str(config.get("minari_id", "")),
        "repo_url": str(config.get("repo_url", "")),
        "repo_sha": str(config.get("repo_sha", "")),
        "method_family": str(config.get("method_family", "")),
        "dataset_option": os.environ.get("DATASET_OPTION", "C"),
        "smoke": str(config.get("smoke", False)),
    }
    if extra_tags:
        tags.update({k: str(v) for k, v in extra_tags.items()})
    mlflow.set_tags(tags)
    try:
        mlflow.log_params(_flatten(config))
    except Exception:
        # Some configs contain non-serializable values; log what we can.
        for k, v in _flatten(config).items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass
    return run
