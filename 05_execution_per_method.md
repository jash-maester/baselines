# 05 — Execution Per Method (Default Configs Only)

Per-method recipes. Each block contains:
- Repo path
- Pre-training touch list (the small things to fix, all import-level)
- Smoke command
- Full training command(s) — **using exact repo defaults, no hyperparameter overrides**
- MLflow integration approach
- Humanoid feasibility

Common environment variables for every training script:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5555
export PYTHONPATH=$(pwd)/compat/src:$PYTHONPATH
export DATASET_OPTION=B            # or C for Humanoid + smoke
export CUDA_VISIBLE_DEVICES=0
```

`scripts/run_one.py` is the standard wrapper. It:
1. Imports `compat_utils.wandb_stub` and calls `install()`.
2. Calls `compat_utils.mlflow_helper.start_run(...)`.
3. Dispatches to the repo's training entrypoint via `runpy.run_path(...)`.
4. Tags the MLflow run with the repo's git SHA and the resolved Minari ID.

## CORL — BC, CQL, IQL, TD3+BC, DT

**Repo:** `repos/CORL/`
**Default configs:** `configs/offline/<algo>/<env>/<split>.yaml` ship with the repo. **Use as-is.** Do not edit.

```bash
cd repos/CORL

# BC
python algorithms/offline/any_percent_bc.py \
    --config_path=configs/offline/any_percent_bc/halfcheetah/medium_v2.yaml

# CQL
python algorithms/offline/cql.py \
    --config_path=configs/offline/cql/halfcheetah/medium_v2.yaml

# IQL
python algorithms/offline/iql.py \
    --config_path=configs/offline/iql/halfcheetah/medium_v2.yaml

# TD3+BC
python algorithms/offline/td3_bc.py \
    --config_path=configs/offline/td3_bc/halfcheetah/medium_v2.yaml

# DT
python algorithms/offline/dt.py \
    --config_path=configs/offline/dt/halfcheetah/medium_v2.yaml
```

For other envs/splits, swap the YAML path. CORL ships configs for:
- halfcheetah/{medium, medium_replay, medium_expert}_v2
- hopper/{medium, medium_replay, medium_expert}_v2
- walker2d/{medium, medium_replay, medium_expert}_v2

**Touch list:**
- CORL by default logs to W&B. Our `wandb_stub` makes `wandb.init`/`wandb.log` no-ops that forward to MLflow.
- CORL's `requirements.txt` lists `d4rl` and `gym` — strip them before `uv pip install` to avoid shadowing conflicts.
- `env.get_normalized_score()` — our shim provides this.

**Smoke command per algo (100 gradient steps, 1 eval episode):**
```bash
# Edit YAML in a temp copy — DO NOT edit ship configs
python -c "
import yaml, shutil
shutil.copy('configs/offline/cql/halfcheetah/medium_v2.yaml', '/tmp/cql_smoke.yaml')
with open('/tmp/cql_smoke.yaml') as f: c = yaml.safe_load(f)
c['max_timesteps'] = 100
c['eval_freq'] = 100
c['n_episodes'] = 1
with open('/tmp/cql_smoke.yaml','w') as f: yaml.safe_dump(c, f)
"
python algorithms/offline/cql.py --config_path=/tmp/cql_smoke.yaml
```

The smoke config is a temporary copy; final training always uses the ship config unmodified.

**MLflow integration (`scripts/run_corl.py`):**

```python
import os, sys, importlib, runpy
sys.path.insert(0, "compat/src")          # picks up fake gym + d4rl
from compat_utils import wandb_stub
wandb_stub.install()                      # forwards wandb.log -> mlflow.log_metric
from compat_utils.mlflow_helper import start_run

algo, env, dataset, seed = sys.argv[1:5]
config_path = sys.argv[5]
config = {"config_path": config_path,
          "minari_id": ...,
          "repo_url": "https://github.com/corl-team/CORL",
          "method_family": {"bc":"behavior_cloning",
                            "cql":"q_learning_offline",
                            "iql":"q_learning_offline",
                            "td3bc":"q_learning_offline",
                            "dt":"sequence_modeling"}[algo]}
with start_run(algo, env, dataset, seed, "full", config):
    sys.argv = ["script", f"--config_path={config_path}"]
    runpy.run_path(f"repos/CORL/algorithms/offline/{algo}.py", run_name="__main__")
```

**Humanoid:** CORL ships no Humanoid config. Per the "exact repo defaults" rule, **skip CORL methods for Humanoid.** Document in `results.md`.

## Diffusion-QL

**Repo:** `repos/Diffusion-Policies-for-Offline-RL/`
**Default config:** Hyperparameters hard-coded per task in `main.py`. The README's exact command (one line per env/split) is the default.

```bash
cd repos/Diffusion-Policies-for-Offline-RL

# Exactly as README — no flag changes
python main.py --env_name halfcheetah-medium-v2 --device 0 --ms online --lr_decay
python main.py --env_name halfcheetah-medium-replay-v2 --device 0 --ms online --lr_decay
python main.py --env_name halfcheetah-medium-expert-v2 --device 0 --ms online --lr_decay
python main.py --env_name hopper-medium-v2          --device 0 --ms online --lr_decay
# ... etc
```

**Touch list:**
- `import gym`, `import d4rl` — auto-shimmed.
- Custom logger; for MLflow, we wrap `main.py` via `runpy` in our launcher and rely on the periodic eval prints. We add MLflow logging at the launcher level, polling the repo's output log file (cleaner than editing `main.py`).
- Alternative: monkey-patch `print` to also call `mlflow.log_metric` when output matches a regex. Hacky but zero source edits.

**Resampling:** Diffusion-QL's eval uses 50-candidate sampling + Q-argmax. This is part of the default config — keep it.

**Humanoid:** No Humanoid hyperparameters in `main.py`. Skip.

## Diffuser

**Repo:** `repos/diffuser/`
**Default config:** `config/locomotion.py` (Python config file, loaded by `--config config.locomotion`).

```bash
cd repos/diffuser

# Three stages, all using config.locomotion defaults
python scripts/train.py --dataset halfcheetah-medium-v2 --config config.locomotion
python scripts/train_values.py --dataset halfcheetah-medium-v2 --config config.locomotion
python scripts/plan_guided.py --dataset halfcheetah-medium-v2 --logbase ./logs
```

**Touch list:**
- Uses an older `conda environment.yml`. Skip; use our venv.
- Hard-coded log paths — override only with `--logbase`, not other flags.
- Default `n_train_steps=1e6` — keep.

**Humanoid:** `config/locomotion.py` lists hyperparameters only for halfcheetah/hopper/walker2d. **Skip Humanoid for Diffuser.**

## Decision Diffuser

**Repo:** `repos/decision-diffuser/code/`
**Default config:** `code/analysis/default_inv.py`

```bash
cd repos/decision-diffuser/code
python analysis/train.py
python analysis/eval.py
```

To switch env/split, edit `default_inv.py` — but the user has said no edits. The repo's defaults run only on `hopper-medium-expert-v2`. To run other envs/splits, the only option per the rules is to set them via environment variable or argparse if supported. Inspect `default_inv.py` for what's exposed:

```bash
# If env override is exposed (verify by reading default_inv.py)
DATASET=halfcheetah-medium-v2 python analysis/train.py
```

If env is *not* CLI-exposed, **DD will only run on its hard-coded default task** (`hopper-medium-expert-v2`). Run that as the DD smoke + final. Document the gap.

**Humanoid:** Not in repo. Skip.

## LDCQ

**Repo:** `repos/ldcq/`
**Default config:** argparse defaults in `training/*.py`. For locomotion, horizon=10 is the default — leave it.

Three stages + eval:

```bash
cd repos/ldcq

# Stage 1: skill encoder
python training/train_skills.py --env_name halfcheetah-medium-v2

# Stage 2: latent diffusion prior
python training/train_diffusion.py --env_name halfcheetah-medium-v2 \
    --skill_model_filename <ckpt-from-stage1>

# Stage 3: Q-learning over latents
python training/train_q_net.py --env_name halfcheetah-medium-v2 \
    --skill_model_filename <ckpt> --diffusion_model_filename <ckpt>

# Stage 4: EVAL — NOT in the repo; we write scripts/eval_ldcq_locomotion.py
python ../../scripts/eval_ldcq_locomotion.py \
    --env_name halfcheetah-medium-v2 \
    --skill_ckpt <s> --diffusion_ckpt <d> --q_ckpt <q> \
    --num_eval_episodes 10
```

**Touch list:**
- `from comet_ml import Experiment` in 3 training scripts — handled by fake comet_ml package.
- `data/<env>.pkl` expected at training time. Pre-build via `scripts/prepare_data_ldcq.py` which calls `d4rl.qlearning_dataset()` through our shim and pickles the result to `repos/ldcq/data/<env>.pkl`.
- LDCQ's `utils/utils.py:49` chunk-filter threshold (`np.all(norms <= 0.8)`) — **DO NOT MODIFY** per the no-source-edits rule. If chunk loss exceeds 50% (we monitor by tagging `data_loss_warning=ldcq_chunk_filter`), note it in `results.md`.

**Stage-4 eval script (`scripts/eval_ldcq_locomotion.py`):**

This is the one new file we have to write — but it does NOT modify LDCQ source. It instantiates `SkillModel`, `Model_Cond_Diffusion`, and the DQN agent from LDCQ as a library, and runs eval episodes using their existing `q_policy()` and decode logic. Strip the goal-reaching parts of `eval/plan_skills_diffusion.py`; everything else is reused. ~150 lines.

**Humanoid:** LDCQ takes `--env_name` as an argparse flag — no per-env config file. **Humanoid works**: `--env_name humanoid-medium-v0` → shim routes to `mujoco/humanoid/medium-v0`.

## EDA

**Repo:** `repos/Efficient-Diffusion-Alignment/`
**Default config:** argparse defaults. README example uses `--beta 0.1` — that's the default invocation, keep it.

Three stages:

```bash
cd repos/Efficient-Diffusion-Alignment

TASK=halfcheetah-medium-v2 ; SEED=0

# Stage 1: behavior pretraining
python -u train_behavior.py --expid ${TASK}-baseline-seed${SEED} --env $TASK --seed $SEED

# Stage 2: critic training
python -u train_critic.py --expid ${TASK}-baseline-seed${SEED} --env $TASK --seed $SEED

# Stage 3: policy fine-tuning (β as in README example — DO NOT vary per task)
python -u finetune_policy.py \
    --expid ${TASK}-baseline-seed${SEED} --env $TASK --seed $SEED \
    --actor_load_path  ./EDA_model_factory/${TASK}-baseline-seed${SEED}/behavior_ckpt200.pth \
    --critic_load_path ./EDA_model_factory/${TASK}-baseline-seed${SEED}/critic_ckpt150.pth \
    --beta 0.1
```

**Important note on β:** The EDA paper's Table 2 (Appendix E) lists per-task β values. **Per the "no tuning" rule, ignore Table 2.** Use the README's `--beta 0.1` for every task-dataset. The resulting numbers may be lower than the paper for some tasks (e.g. medium-expert tasks where β=2.0 is paper-optimal) — that's the expected behavior of "no tuning."

**Touch list:**
- Clean repo. Only standard import shims.
- `EDA_model_factory/` directory created relative to cwd — pre-create.
- Each stage = separate MLflow run.

**Humanoid:** EDA takes `--env` as CLI flag — Humanoid works: `--env humanoid-medium-v0`.

## QGPO

**Repo:** `repos/CEP-energy-guided-diffusion/Offline_RL_2D/`
**Default config:** README's exact command, including all flags.

Two stages:

```bash
cd repos/CEP-energy-guided-diffusion/Offline_RL_2D

TASK=halfcheetah-medium-expert-v2 ; SEED=0

# Stage 1: behavior pretraining
python -u train_behavior.py --expid ${TASK}${SEED}reproduce --env $TASK --seed $SEED

# Stage 2: critic + eval (this script logs the final number)
python -u train_critic.py \
    --actor_load_path ./models_rl/${TASK}${SEED}reproduce/behavior_ckpt.pth \
    --expid ${TASK}${SEED}reproduce \
    --env $TASK \
    --diffusion_steps 15 \
    --seed $SEED \
    --alpha 3 \
    --q_alpha 1 \
    --method "CEP"
```

The `--diffusion_steps 15 --alpha 3 --q_alpha 1 --method "CEP"` are documented in the README as the default invocation — they're not per-task hyperparameter overrides.

**Touch list:**
- `import gym`, `import d4rl` — auto-shimmed.
- Eval logic embedded in `train_critic.py` — the final `final.normalized_score_mean` is logged at end-of-training.

**Humanoid:** Takes `--env` as CLI flag — Humanoid works: `--env humanoid-medium-v0`.

## "My method" placeholder

Reserved row in the table. Train and log under `algo=mymethod`, env name passed via CLI, MLflow tags applied per `03_mlflow_conventions.md`. Use the same `OldGymEnvWrapper.get_normalized_score()` for comparability.

## Seeds and parallelism

3 seeds per (algo, env, dataset) at minimum.

Total runs (with the active 11 methods + dataset matrix from `02_dataset_situation.md`):

| Method | HC × {3 splits} | Hop × {3 splits} | Walker × {3 splits} | Humanoid × {2 splits} | Total per seed |
|---|---|---|---|---|---|
| BC, CQL, IQL, TD3+BC, DT (CORL × 5) | 5×3 | 5×3 | 5×3 | – (no config) | 45 |
| Diffuser, DD, D-QL (× 3) | 3×3 | 3×3 | 3×3 | – | 27 |
| LDCQ, EDA, QGPO (× 3) | 3×3 | 3×3 | 3×3 | 3×2 | 33 |
| **TOTAL per seed** | | | | | **105** |

× 3 seeds = **315 training pipelines**. With multi-stage methods (LDCQ has 4 stages, EDA has 3, QGPO has 2), MLflow run count is ~**450 runs**. Plus your method.

## Eval episodes per run

10 minimum. Run-time impact on Humanoid is ~3x other envs because obs/state space is larger (376 dims vs 17), so budget accordingly.

## Skipped methods (not run, only cited)

For these, the row in `results.md` gets a "from paper" footnote — no MLflow run, no reproduction:

- **BCQ** — weak baseline, old repo
- **SfBC** — dominated by QGPO
- **IDQL** — JAX, separate stack

These don't appear in MLflow at all; their numbers come from the source papers as written, with a clear footnote that they're not reproduced.
