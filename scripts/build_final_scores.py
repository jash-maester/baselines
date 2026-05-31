"""Build state/final_scores.csv — Minari vs paper-reported D4RL comparison.

Reads MLflow experiment `latent_cep_baselines` (id=40), aggregates 3 seeds per
(algo, env, dataset, [guidance_scale]), and joins paper D4RL scores from
reference_scores.json. Also computes a D4RL-equivalent score from raw_return
using the original D4RL paper's ref constants (humanoid blank — not reported
in D4RL papers).

Stage picked per algo for the "final" score:
  bc / cql / iql / dt / dql -> 'full'
  eda                       -> 'finetune'  (post-finetune policy)
  qgpo                      -> 'critic'    (trained guided policy)

Run:   uv run python scripts/build_final_scores.py
"""
from __future__ import annotations
import csv, json, re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import mlflow

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri("http://localhost:5555")

# D4RL paper ref constants (raw_return -> 0..100 normalized).
# Source: D4RL paper, Fu et al. 2020 — original repo's `infos.py`.
D4RL_REFS = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper":      ( -20.272305,  3234.3),
    "walker2d":    (   1.629008,  4592.3),
    # humanoid: not in D4RL — Minari-only env -> no D4RL-equiv conversion
}

STAGE_FOR_ALGO = {
    "bc": "full", "cql": "full", "iql": "full", "dt": "full", "dql": "full",
    "eda": "finetune",
    "qgpo": "critic",
}

ENV_ORDER  = ["halfcheetah", "hopper", "walker2d", "humanoid"]
DATA_ORDER = ["medium", "expert"]
ALGO_ORDER = ["bc", "cql", "iql", "dt", "dql", "eda", "qgpo"]


def main() -> None:
    ref = json.loads((ROOT / "reference_scores.json").read_text())
    paper_scores  = ref["scores"]
    paper_sources = ref["_sources"]

    c = mlflow.tracking.MlflowClient()
    runs = c.search_runs(["40"], max_results=5000)

    # Group: (algo, env, dataset, variant_kind, variant_val) -> [(minari_score, raw_return), ...]
    # variant_kind in {None, 'gs', 'target'}: qgpo splits by guidance scale, dt by target return.
    groups: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for r in runs:
        t = r.data.tags
        algo, stage = t.get("algo"), t.get("stage")
        if algo not in STAGE_FOR_ALGO or stage != STAGE_FOR_ALGO[algo]:
            continue
        if t.get("smoke", "False") == "True":
            continue
        env, dataset = t.get("env"), t.get("dataset")
        m = r.data.metrics

        if algo == "qgpo":
            for k in m:
                if not k.startswith("d4rl_normalized_score_gs"):
                    continue
                gs = k[len("d4rl_normalized_score_gs"):]
                rr = m.get(f"raw_return_gs{gs}")
                if rr is None:
                    continue
                groups[(algo, env, dataset, "gs", gs)].append((m[k], rr))
        elif algo == "dt":
            # CORL DT evaluates at 2 target returns per env -> 2 variants.
            targets = sorted({mt.group(1) for k in m
                              if (mt := re.match(r"eval\.([\d.]+)_normalized_score_mean$", k))})
            for tg in targets:
                ns = m.get(f"eval.{tg}_normalized_score_mean")
                rr = m.get(f"eval.{tg}_return_mean")
                if ns is None or rr is None:
                    continue
                groups[(algo, env, dataset, "target", tg)].append((ns, rr))
        else:
            ns = m.get("d4rl_normalized_score")
            rr = m.get("raw_return")
            if ns is None or rr is None:
                continue
            groups[(algo, env, dataset, None, None)].append((ns, rr))

    rows_out = []
    for (algo, env, dataset, vkind, vval), seedvals in groups.items():
        n = len(seedvals)
        minari = [s for s, _ in seedvals]
        raws   = [r for _, r in seedvals]
        minari_mean, minari_std = mean(minari), (pstdev(minari) if n > 1 else 0.0)
        raw_mean,    raw_std    = mean(raws),   (pstdev(raws)   if n > 1 else 0.0)

        if env in D4RL_REFS:
            rmin, rmax = D4RL_REFS[env]
            d4rl_each = [(rr - rmin) / (rmax - rmin) * 100 for rr in raws]
            d4rl_mean = mean(d4rl_each)
            d4rl_std  = pstdev(d4rl_each) if n > 1 else 0.0
        else:
            d4rl_mean = d4rl_std = None  # humanoid

        env_d4rl = f"{env}-{dataset}-v0"
        paper = paper_scores.get(algo, {}).get(env_d4rl)
        delta = (d4rl_mean - paper) if (paper is not None and d4rl_mean is not None) else None

        rows_out.append({
            "algo": algo,
            "env": env,
            "dataset": dataset,
            "env_d4rl_name": env_d4rl,
            "guidance_scale": vval if vkind == "gs" else "",
            "target_return":  vval if vkind == "target" else "",
            "stage": STAGE_FOR_ALGO[algo],
            "n_seeds": n,
            "minari_score_mean": round(minari_mean, 3),
            "minari_score_std":  round(minari_std,  3),
            "raw_return_mean":   round(raw_mean,    3),
            "raw_return_std":    round(raw_std,     3),
            "d4rl_equiv_mean": "" if d4rl_mean is None else round(d4rl_mean, 3),
            "d4rl_equiv_std":  "" if d4rl_std  is None else round(d4rl_std,  3),
            "paper_d4rl_score": "" if paper is None else paper,
            "paper_source": paper_sources.get(algo, ""),
            "delta_d4rl_equiv_minus_paper": "" if delta is None else round(delta, 3),
            "notes": "humanoid not in D4RL papers" if env == "humanoid" else "",
        })

    def sort_key(r):
        variant = r["guidance_scale"] or r["target_return"] or ""
        return (
            ALGO_ORDER.index(r["algo"])    if r["algo"]    in ALGO_ORDER else 99,
            ENV_ORDER.index(r["env"])      if r["env"]     in ENV_ORDER  else 99,
            DATA_ORDER.index(r["dataset"]) if r["dataset"] in DATA_ORDER else 99,
            float(variant) if variant else -1.0,
        )
    rows_out.sort(key=sort_key)

    out = ROOT / "state" / "final_scores.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
