# 02 — Dataset Situation: Read Before Anything Else

## TL;DR

Minari publishes **`simple`, `medium`, `expert`** for `mujoco/{halfcheetah,hopper,walker2d,humanoid}` — and that's it.

The original D4RL `medium-replay-v2` and `medium-expert-v2` mixtures are **not** in Minari for any of the four envs. **Three options** for what to do.

## What Minari publishes (verified 2026-05-19)

Endpoint: https://minari.farama.org/datasets/mujoco/index.html

| Env | simple | medium | expert | medium-replay | medium-expert |
|---|:---:|:---:|:---:|:---:|:---:|
| HalfCheetah | ✓ `mujoco/halfcheetah/simple-v0` | ✓ `mujoco/halfcheetah/medium-v0` | ✓ `mujoco/halfcheetah/expert-v0` | ❌ | ❌ |
| Hopper | ✓ | ✓ | ✓ | ❌ | ❌ |
| Walker2d | – (not listed) | ✓ | ✓ | ❌ | ❌ |
| **Humanoid** | ✓ | ✓ | ✓ | ❌ | ❌ |

Walker2d's `simple` may or may not exist depending on Minari version — verify when the validation script runs. Humanoid has the same three splits as the others.

There is no `Humanoid` dataset under the `D4RL/` namespace either. Humanoid is only available under `mujoco/`.

## Why this matters

The three headliner papers report on `medium-v2`, `medium-replay-v2`, `medium-expert-v2`. Six of those nine slots **don't exist in Minari for any env**. For Humanoid, none of the original papers report on it at all, so there are no "paper numbers" to match anyway.

This means:
- For HalfCheetah, Hopper, Walker2d: we can run on `medium-v0` and `expert-v0` and have approximate analogues to paper splits — but the numbers won't be directly comparable to papers (different generation policy, different MuJoCo version, different reference scores).
- For Humanoid: we can run on `medium-v0`, `expert-v0`, (and `simple-v0`) — but there's no paper baseline anywhere. Numbers are purely relative across methods.

## Option A — Generate datasets ourselves

Use `minari.DataCollector` wrapping a Gymnasium-v5 env, train SAC to expert level, save trajectories at multiple checkpoints to build `medium-replay` and `medium-expert` analogues.

**Cost:** ~3 GPU-days per env (SAC training) × 4 envs = ~12 GPU-days. Plus the data still won't match D4RL's reference policy.

**Verdict:** Don't. Too much work for a baseline-reproduction task.

## Option B — Convert original D4RL hdf5s to local Minari datasets (recommended for HalfCheetah/Hopper/Walker2d)

D4RL is deprecated but the hdf5 files are still on the Berkeley mirror. URLs in `Farama-Foundation/D4RL/d4rl/infos.py`. Download once, convert to local Minari datasets under namespace `localD4RL/<env>/<split>-v2`.

**Cost:**
- 1-time data fetch: ~30 min (12 hdf5 files, ~4 GB total)
- Python 3.9 venv with `mujoco_py` for the fetch step (the conversion to Minari can happen in main venv)
- Conversion script: ~30 lines of Python

**Verdict:** Recommended for HalfCheetah / Hopper / Walker2d because it gives paper-comparable numbers.

**Humanoid:** Option B doesn't apply — there is no original-D4RL Humanoid hdf5 on the Berkeley mirror. Humanoid was never in D4RL.

## Option C — Use Minari-native datasets only

Train and eval on whatever Minari publishes. Three splits per env × 4 envs = up to 12 task-dataset combos.

**Cost:** Zero data prep.

**Verdict:** Required for Humanoid. Recommended as Phase-1 smoke-test data for the other three envs too.

## Recommended strategy: hybrid B + C

**Phase 1 — Smoke tests + Humanoid + Walker2d simple/medium/expert (Option C):**
- Use `mujoco/{halfcheetah,hopper,walker2d,humanoid}/{medium,expert}-v0` for all smoke tests.
- Final Humanoid numbers come from Option C — no choice.

**Phase 2 — Final HalfCheetah/Hopper/Walker2d numbers (Option B):**
- Once smoke is green, fetch D4RL hdf5s, convert to local Minari `localD4RL/<env>/{medium,medium-replay,medium-expert}-v2`.
- Re-run the matrix on Option-B datasets for paper-comparable numbers.

**Two separate dataset namespaces, two separate columns in results table** — but at least all methods are evaluated on identical data within each column.

## Final task-dataset matrix

| Env | medium-v0 (C) | expert-v0 (C) | medium-v2 (B) | medium-replay-v2 (B) | medium-expert-v2 (B) | expert-v2 (B) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| HalfCheetah | ✓ smoke | ✓ smoke | ✓ final | ✓ final | ✓ final | (✓ if time) |
| Hopper | ✓ smoke | ✓ smoke | ✓ final | ✓ final | ✓ final | (✓ if time) |
| Walker2d | ✓ smoke | ✓ smoke | ✓ final | ✓ final | ✓ final | (✓ if time) |
| **Humanoid** | ✓ **final** | ✓ **final** | – | – | – | – |

That's 9 final task-dataset combos for HalfCheetah/Hopper/Walker2d (Option B) + 2 final for Humanoid (Option C) = **11 total**.

Multiply by 11 methods (8 active + 3 headliners) × 3 seeds = **~360 training runs** at minimum.

## Minari ID mappings (compat layer)

```python
# compat_utils/minari_mapping.py

# Option B: paper-faithful HalfCheetah/Hopper/Walker2d (locally created)
D4RL_TO_MINARI_OPTION_B = {
    'halfcheetah-medium-v2':         'localD4RL/halfcheetah/medium-v2',
    'halfcheetah-medium-replay-v2':  'localD4RL/halfcheetah/medium-replay-v2',
    'halfcheetah-medium-expert-v2':  'localD4RL/halfcheetah/medium-expert-v2',
    'halfcheetah-expert-v2':         'localD4RL/halfcheetah/expert-v2',
    'hopper-medium-v2':              'localD4RL/hopper/medium-v2',
    'hopper-medium-replay-v2':       'localD4RL/hopper/medium-replay-v2',
    'hopper-medium-expert-v2':       'localD4RL/hopper/medium-expert-v2',
    'hopper-expert-v2':              'localD4RL/hopper/expert-v2',
    'walker2d-medium-v2':            'localD4RL/walker2d/medium-v2',
    'walker2d-medium-replay-v2':     'localD4RL/walker2d/medium-replay-v2',
    'walker2d-medium-expert-v2':     'localD4RL/walker2d/medium-expert-v2',
    'walker2d-expert-v2':            'localD4RL/walker2d/expert-v2',
}

# Option C: Minari-native (only path for Humanoid)
D4RL_TO_MINARI_OPTION_C = {
    'halfcheetah-medium-v0': 'mujoco/halfcheetah/medium-v0',
    'halfcheetah-expert-v0': 'mujoco/halfcheetah/expert-v0',
    'hopper-medium-v0':      'mujoco/hopper/medium-v0',
    'hopper-expert-v0':      'mujoco/hopper/expert-v0',
    'walker2d-medium-v0':    'mujoco/walker2d/medium-v0',
    'walker2d-expert-v0':    'mujoco/walker2d/expert-v0',
    'humanoid-medium-v0':    'mujoco/humanoid/medium-v0',
    'humanoid-expert-v0':    'mujoco/humanoid/expert-v0',
    'humanoid-simple-v0':    'mujoco/humanoid/simple-v0',
}
```

The shim picks based on env var `DATASET_OPTION=B|C`. For Humanoid, only `C` is meaningful.

**Validation before any training:**

```python
import minari
all_ids = list(D4RL_TO_MINARI_OPTION_B.values()) + list(D4RL_TO_MINARI_OPTION_C.values())
for mid in all_ids:
    try:
        if mid.startswith("localD4RL/"):
            # local — these are validated by the conversion script
            continue
        minari.load_dataset(mid, download=True)
        print(f"OK   {mid}")
    except Exception as e:
        print(f"FAIL {mid}: {e}")
```

Run this first thing. If `mujoco/walker2d/simple-v0` (or whatever) doesn't exist, drop it from the matrix and update the mapping.

## Reference scores

For Option B (paper-comparable), embed the original D4RL reference scores from `D4RL/d4rl/infos.py` into the local Minari dataset metadata:

```python
D4RL_REFS = {
    'halfcheetah-medium-v2':         (-280.178953, 12135.0),
    'halfcheetah-medium-replay-v2':  (-280.178953, 12135.0),
    'halfcheetah-medium-expert-v2':  (-280.178953, 12135.0),
    'hopper-medium-v2':              (-20.272305,   3234.3),
    'hopper-medium-replay-v2':       (-20.272305,   3234.3),
    'hopper-medium-expert-v2':       (-20.272305,   3234.3),
    'walker2d-medium-v2':            (1.629008,     4592.3),
    'walker2d-medium-replay-v2':     (1.629008,     4592.3),
    'walker2d-medium-expert-v2':     (1.629008,     4592.3),
}
```

For Option C (Humanoid + smoke), use whatever Minari provides via `dataset.spec.reference_min_score` / `reference_max_score`. For Humanoid specifically, **no paper has a reference number**, so the normalized score is informative only across runs in our matrix.

## Humanoid-specific notes

- **No paper baseline.** All Humanoid numbers are head-to-head between methods. No "expected score" to compare to.
- **Higher-dimensional.** Humanoid has obs dim 376 (v5) vs HalfCheetah's 17. Default network widths in some repos (e.g. Diffusion-QL's 256-unit MLPs) may be undersized. **Do not increase widths** — that's a hyperparameter change. Run and let the results show what happens.
- **More expensive.** Eval episodes are ~1000 steps; the larger network state space means each rollout is ~3x slower than HalfCheetah. Budget accordingly.
- **Per-method viability:** Some baselines that ship Humanoid-incompatible defaults (no Humanoid YAML in CORL; no Humanoid block in Diffuser/DD/D-QL configs) will be **skipped on Humanoid**. Only LDCQ, EDA, QGPO, and your method run on Humanoid. This is a consequence of the "no modifications" rule — accept it and document.
