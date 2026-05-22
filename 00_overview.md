# 00 — Overview: Locomotion Baseline Reproduction on Minari

## What this folder is

A plan for re-running the **offline-RL locomotion baselines** cited by **LDCQ**, **EDA**, and **QGPO** on the **Minari** dataset API (since D4RL is deprecated), logging everything to a **locally-hosted MLflow Docker container at `http://localhost:5555`**.

## Hard constraints from the user

1. **Locomotion only.** Four environments: **HalfCheetah, Hopper, Walker2d, Humanoid**. No AntMaze, no Kitchen, no Maze2D, no Adroit.
2. **No hyperparameter tuning.** Use each repo's default config verbatim. If a config file ships with the repo, use it as-is. If hyperparameters are hard-coded in the script, leave them alone. The goal is to reproduce the repo's own defaults on Minari data, not to optimize anything.
3. **No source-code modifications to baselines.** The only edits to cloned repos are import-line redirects to our compat shim and silencing of failed telemetry calls (comet_ml / wandb with empty API keys). All real changes are confined to the `compat/` shim package.
4. **Toolchain:** `uv` and/or `pixi` (both available; pick one). Defaults below show both for the same operations.

## Read order

1. **`00_overview.md`** — this file.
2. **`01_baselines_catalog.md`** — every baseline cited by the three headliner papers, with repo URLs and language stacks.
3. **`02_dataset_situation.md`** — *read carefully*. Minari does not host original D4RL `medium-replay-v2` / `medium-expert-v2`. Decision required.
4. **`03_mlflow_conventions.md`** — naming scheme, tags.
5. **`04_compat_layer.md`** — the shared compatibility package design.
6. **`05_execution_per_method.md`** — one section per baseline with exact commands using the **default repo configs**.
7. **`06_evaluation_protocol.md`** — eval episodes, normalized scoring.
8. **`07_task_checklist.md`** — numbered checklist Claude Code can tick off.
9. **`08_risks_and_decisions.md`** — open decisions; resolve before kicking off long jobs.

## Headliners

| Paper | Method | arXiv | Repo |
|---|---|---|---|
| Venkatraman et al. 2023 | **LDCQ** | 2309.06599 | https://github.com/ldcq/ldcq |
| Chen et al. 2024 (NeurIPS) | **EDA** | 2407.09024 | https://github.com/thu-ml/Efficient-Diffusion-Alignment |
| Lu et al. 2023 (ICML) | **QGPO / CEP** | 2304.12824 | https://github.com/thu-ml/CEP-energy-guided-diffusion |

## Baseline matrix

| Baseline | Repo | LDCQ-cited | EDA-cited | QGPO-cited |
|---|---|:---:|:---:|:---:|
| BC | CORL | ✓ | ✓ | – |
| BCQ | CORL / sfujim | ✓ | ✓ | ✓ |
| CQL | CORL | ✓ | ✓ | ✓ |
| IQL | CORL | ✓ | ✓ | ✓ |
| TD3+BC | CORL | – | ✓ | – |
| DT | CORL / kzl | ✓ | ✓ | ✓ |
| Diffuser | jannerm | ✓ | – | ✓ |
| Decision Diffuser | anuragajay | ✓ | – | ✓ |
| Diffusion-QL | Zhendong-Wang | – | ✓ | ✓ |
| SfBC | ChenDRAG | – | – | ✓ |
| IDQL (JAX) | philippe-eecs | – | ✓ | – |

15 methods total once you add the 3 headliners.

## Environments and dataset splits

**Four environments × N splits per dataset option:**

| Env | Gymnasium env | Minari group | Splits available natively (Option C) |
|---|---|---|---|
| HalfCheetah | `HalfCheetah-v5` | `mujoco/halfcheetah` | simple, medium, expert |
| Hopper | `Hopper-v5` | `mujoco/hopper` | simple, medium, expert |
| Walker2d | `Walker2d-v5` | `mujoco/walker2d` | simple, medium, expert |
| **Humanoid** | `Humanoid-v5` | `mujoco/humanoid` | simple, medium, expert |

**Good news:** Humanoid IS available in Minari's `mujoco/humanoid` namespace. All four envs are on equal footing.

**Bad news:** Minari only ships `simple`, `medium`, `expert` for all four envs. The original D4RL `medium-replay-v2` and `medium-expert-v2` mixtures are not in Minari. See `02_dataset_situation.md` for what to do about this.

## Hard cardinal rules

1. **Never modify repo logic.** Only edits permitted in cloned repos: (a) import-line rewrites (transparently via shim packages on `PYTHONPATH`), (b) commenting out telemetry that crashes on empty creds. No hyperparameter changes. No code changes.
2. **All new code goes in `compat/`** — under the project root, plus `scripts/` for orchestration. No `configs/` directory — we use the repos' own configs.
3. **Every training run logs to MLflow** under experiment `latent_cep_baselines` (long matrix) or `latent_cep_baselines_smoke` (autoresearch smoke loop) with tags per `03_mlflow_conventions.md`. The plan originally said `baselines`; renamed 2026-05-19 to avoid collision with pre-existing unrelated runs on this host's MLflow.
4. **Smoke-test before scaling.** Before the full matrix, run a tiny training session per method on `halfcheetah` data. Each smoke should finish in < 5 min. Only after every smoke run logs an MLflow run with at least `train.loss` do you launch the matrix.

## Final deliverable

A pandas DataFrame (CSV + Markdown table) with rows = methods, columns = task-dataset combos, cells = normalized score mean ± std across seeds. Your algorithm as the last row.

All experiments queryable in MLflow at `http://localhost:5555` under experiment `latent_cep_baselines` (and `latent_cep_baselines_smoke` for autoresearch).

## Tooling: uv vs pixi

Both are supported. Use whichever is already on the host.

```bash
# uv
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ./compat
uv pip install -r requirements.txt

# pixi (alternative)
pixi init --pyproject
pixi add "python=3.11" "torch>=2.4,<2.7" "gymnasium[mujoco]>=1.0" "minari>=0.5" \
        "mujoco>=3.1" "numpy<2.0" "mlflow>=2.9" "h5py" "scikit-learn"
pixi run pip install -e ./compat
```

Pixi notes:
- Uses conda-forge by default, which is fine for `gymnasium`, `mujoco`, `mlflow`.
- The compat shim is installed via `pip install -e` inside the pixi env because it shadows pip-installed package names (`gym`, `d4rl`, `comet_ml`) and pixi/conda-forge don't have the same shadowing semantics.

Pick one, document it in `RUN_LOG.md`, do not mix.
