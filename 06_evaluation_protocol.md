# 06 — Evaluation Protocol

## Single source of truth: the eval function

Every algorithm's final eval, regardless of internal training framework, goes through one shared function:

```python
# scripts/eval_common.py
import numpy as np
import torch
import gym                                      # the compat shim
import minari


def evaluate_policy(
    policy_fn,                                  # callable: state (np.ndarray) -> action (np.ndarray)
    d4rl_env_name,                              # e.g. "humanoid-medium-v0" or "halfcheetah-medium-v2"
    n_episodes=10,
    seed_offset=0,
    max_episode_steps=1000,
    deterministic=True,
):
    """Returns dict with raw_returns, normalized_scores, episode_lengths, mean/std."""
    env = gym.make(d4rl_env_name)
    raw_returns, lens, norm_scores = [], [], []

    for ep in range(n_episodes):
        ep_seed = seed_offset * 1000 + ep
        env.unwrapped.reset(seed=ep_seed)
        obs = env.reset()
        ep_return = 0.0
        ep_len = 0
        done = False
        while not done and ep_len < max_episode_steps:
            with torch.no_grad():
                action = policy_fn(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
        raw_returns.append(ep_return)
        lens.append(ep_len)
        norm_scores.append(env.get_normalized_score(ep_return) * 100.0)

    env.close()
    return dict(
        raw_returns=raw_returns,
        normalized_scores=norm_scores,
        episode_lengths=lens,
        raw_return_mean=float(np.mean(raw_returns)),
        raw_return_std=float(np.std(raw_returns)),
        normalized_score_mean=float(np.mean(norm_scores)),
        normalized_score_std=float(np.std(norm_scores)),
    )
```

One function eliminates the most common "baseline number doesn't match the paper" cause: differences in eval — episode count, deterministic vs stochastic, episode length cap, normalization choice. Pin all of these.

## Eval episode count

| Use case | N episodes |
|---|---|
| Smoke test | 2 |
| Per-epoch eval during training | 5 |
| **Final reported number** | **10** |
| High-confidence (optional, post-hoc) | 100 |

Defaults assume 10. Increase to 100 only after the full matrix has run and you want to tighten error bars on top performers.

## Deterministic vs stochastic per algo

| Algo | Eval mode | Tag value |
|---|---|---|
| BC, CQL, IQL, TD3+BC | deterministic (mean action) | `eval_mode=det` |
| DT | deterministic (greedy decode) | `eval_mode=det` |
| Diffuser, DD | stochastic (single sample) | `eval_mode=stoch_single` |
| Diffusion-QL | stochastic + 50-candidate Q-resample | `eval_mode=stoch_resample_50` |
| QGPO | stochastic with CEP energy guidance | `eval_mode=stoch_cep` |
| LDCQ | stochastic + Q-argmax over latents | `eval_mode=stoch_argmax_q` |
| EDA | stochastic + 4-candidate rejection sampling | `eval_mode=stoch_resample_4` |

Each algo's `policy_fn` returned by its launcher encodes its native eval mode. Log via MLflow tag `eval_mode`.

## Normalized scoring

Minari publishes per-dataset reference scores: `dataset.spec.reference_min_score` and `.reference_max_score`. Our shim's `env.get_normalized_score(raw_return)` does:

```
score = (raw_return - ref_min) / (ref_max - ref_min) * 100
```

### For Option B datasets (HalfCheetah/Hopper/Walker2d, paper-comparable)

Use D4RL's original reference scores. When the conversion script `scripts/d4rl_to_minari.py` builds the local Minari datasets, it embeds these as `reference_min_score` / `reference_max_score` in the dataset metadata:

```python
D4RL_REFS = {
    'halfcheetah-medium-v2':         (-280.178953, 12135.0),
    'halfcheetah-medium-replay-v2':  (-280.178953, 12135.0),
    'halfcheetah-medium-expert-v2':  (-280.178953, 12135.0),
    'halfcheetah-expert-v2':         (-280.178953, 12135.0),
    'hopper-medium-v2':              (-20.272305,   3234.3),
    'hopper-medium-replay-v2':       (-20.272305,   3234.3),
    'hopper-medium-expert-v2':       (-20.272305,   3234.3),
    'hopper-expert-v2':              (-20.272305,   3234.3),
    'walker2d-medium-v2':            (1.629008,     4592.3),
    'walker2d-medium-replay-v2':     (1.629008,     4592.3),
    'walker2d-medium-expert-v2':     (1.629008,     4592.3),
    'walker2d-expert-v2':            (1.629008,     4592.3),
}
```

### For Option C datasets (Humanoid + smoke)

Use whatever `dataset.spec.reference_min_score` / `reference_max_score` Minari provides. For Humanoid, there are no paper-published reference scores, so the normalized number is meaningful only within our matrix.

## Per-episode seed scheme

```
env.reset(seed = global_seed * 1000 + episode_index)
```

`global_seed` is the run's seed (0, 1, 2). Episode index 0..N-1. Wide enough that there's no collision across seeds. **Do not** use the same seed for every eval episode — you'd be measuring policy variance × 0 environment variance.

## Sanity-check expectations (for HalfCheetah/Hopper/Walker2d Option-B runs)

Approximate paper-published normalized scores on D4RL locomotion `medium-v2`, averaged across the 3 envs:

| Method | Approx. paper score |
|---|---|
| BC | ~42 |
| CQL | ~50 |
| IQL | ~57 |
| TD3+BC | ~58 |
| DT | ~50 |
| Diffuser | ~55 |
| Decision Diffuser | ~56 |
| Diffusion-QL | ~70 |
| QGPO | ~73 |
| LDCQ | reported only on AntMaze in main table; locomotion in appendix |
| EDA | ~76 |

**Caveat:** Per "no hyperparameter tuning," EDA in particular will run with `--beta 0.1` for every task instead of the per-task tuned values in the paper's Appendix Table 2. Expect **lower than paper** on medium-expert tasks (paper used β=2.0). Tag with `tuning=none_default`. The number is still meaningful: it's the score of EDA's default configuration on D4RL data, which is the apples-to-apples comparison for someone else also reporting "default settings."

**If a method scores < 30% of expected:** something is wrong. Check:
1. Data load — print `obs.shape`, `mean(rewards)`, total transitions. Compare to D4RL's published dataset sizes.
2. Eval — run BC; BC's number is the most predictable lower bound.
3. If both look right, log as-is with tag `reproduction_health=degraded`. Do not tune.

## Humanoid expectations

**Nothing.** No paper baseline exists. The first run of each method *is* the expectation. The interpretation is purely "which methods rank higher than which" within our matrix.

Likely outcomes:
- BC and CQL on Humanoid: probably weak (Humanoid is hard for low-capacity methods).
- Diffusion methods: more flexible — possibly better, possibly worse depending on whether the default network widths (256) are sufficient for the 376-dim obs space.
- EDA / QGPO / LDCQ: the methods we have most confidence will at least run end-to-end.

Document Humanoid results separately in `results.md` since no paper comparison exists.

## Aggregation script

`scripts/build_results_table.py`:

```python
import mlflow
import pandas as pd
import os

client = mlflow.MlflowClient("http://localhost:5555")
exp = client.get_experiment_by_name("latent_cep_baselines")  # renamed from "baselines" 2026-05-19

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="tags.stage IN ('eval', 'full', 'finetune', 'critic')",
)
runs = runs.dropna(subset=["metrics.final.normalized_score_mean"])

agg = (runs
    .groupby(["tags.algo", "tags.env", "tags.dataset"])
    .agg(
        score_mean=("metrics.final.normalized_score_mean", "mean"),
        score_std=("metrics.final.normalized_score_mean", "std"),
        n_seeds=("metrics.final.normalized_score_mean", "count"),
    )
    .reset_index()
)
agg["task"] = agg["tags.env"] + "-" + agg["tags.dataset"]
agg["score"] = agg.apply(
    lambda r: f"{r['score_mean']:.1f} ± {r['score_std']:.1f} (n={r['n_seeds']})", axis=1,
)
table = agg.pivot(index="tags.algo", columns="task", values="score")
table.to_csv("results.csv")

with open("results.md", "w") as f:
    f.write("# Baseline Comparison\n\n")
    f.write(f"Generated: {pd.Timestamp.now()}\n\n")
    f.write("Scores are normalized × 100. Humanoid columns use Minari reference scores; ")
    f.write("HalfCheetah/Hopper/Walker2d columns use D4RL reference scores via Option-B local datasets.\n\n")
    f.write(table.to_markdown())
    f.write("\n")
```

Run after all training jobs complete.

## When a baseline fails

Procedure when a baseline's number is < 30% of expected (or crashes mid-training):

1. Verify data load (printed shapes, total transitions).
2. Verify eval works (manual run with `n_episodes=2`).
3. If both work and the method genuinely diverges, log run with `reproduction_health=degraded`.
4. **Do not tune** — that violates the rule. Note in `results.md` footnote.

The aggregation table reports each method's actual run, however bad. The point of "no tuning" is honesty about default-config performance.
