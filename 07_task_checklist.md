# 07 — Task Checklist for Claude Code

Sequential. Each task gates the next. Do not start task N+1 until task N is verified-green.

## Phase 0 — Decisions (10 min)

- [ ] **T0.1** — Read `08_risks_and_decisions.md`. Resolve open decisions: seed count (default 3), eval episode count (default 10), tooling (uv or pixi).
- [ ] **T0.2** — Confirm GPU: `nvidia-smi` shows ≥1 GPU with ≥12 GB VRAM. Record GPU model + driver in `RUN_LOG.md`.
- [ ] **T0.3** — Confirm MLflow container: `make mlflow_status`. If down: `make mlflow_up`. Open `http://localhost:5555` in browser; should render empty experiment list.
- [ ] **T0.4** — Confirm tooling: `uv --version` and/or `pixi --version`. Pick one. Record in `RUN_LOG.md`.

## Phase 1 — Compat layer (1 day)

- [ ] **T1.1** — Create project structure per `04_compat_layer.md`.
- [ ] **T1.2** — Write `compat/pyproject.toml` declaring `name = "compat-shim"` with packages `gym`, `d4rl`, `comet_ml`, `compat_utils`.
- [ ] **T1.3** — Implement `compat/src/gym/__init__.py`.
- [ ] **T1.4** — Implement `compat/src/d4rl/__init__.py`.
- [ ] **T1.5** — Implement `compat/src/comet_ml/__init__.py`.
- [ ] **T1.6** — Implement `compat_utils/minari_mapping.py` with both Option B and Option C maps. Humanoid in the C map.
- [ ] **T1.7** — Implement `compat_utils/env_factory.py`.
- [ ] **T1.8** — Implement `compat_utils/d4rl_dict_converter.py`.
- [ ] **T1.9** — Implement `compat_utils/torch_load_patch.py`.
- [ ] **T1.10** — Implement `compat_utils/mlflow_helper.py`.
- [ ] **T1.11** — Implement `compat_utils/wandb_stub.py`.
- [ ] **T1.12** — Set up venv: `uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -e ./compat && uv pip install <project deps>` (per `04_compat_layer.md`). If pixi: equivalent commands per that doc.
- [ ] **T1.13** — Validate shims (under `DATASET_OPTION=C` for Humanoid):
  ```bash
  export DATASET_OPTION=C
  python -c "import gym; e=gym.make('humanoid-medium-v0'); print(e); print(e.get_dataset()['observations'].shape)"
  python -c "import d4rl, gym; print(d4rl.qlearning_dataset(gym.make('halfcheetah-medium-v0')).keys())"
  ```
  Both must succeed.

## Phase 2 — Dataset preparation (1 day total)

### Phase 2a — Option C validation (30 min)

- [ ] **T2.1** — Run the Minari ID validation loop from `02_dataset_situation.md`. All Option C IDs (halfcheetah/hopper/walker2d/humanoid × {medium, expert} v0) must download. If any fail, drop from matrix and update `compat_utils/minari_mapping.py`.

### Phase 2b — Option B preparation (1 day; skip if HC/Hop/Walker columns are not required)

- [ ] **T2.2** — Create a separate Python-3.9 venv: `uv venv --python 3.9 .venv-d4rl`. Install `d4rl` from Farama fork: `uv pip install git+https://github.com/Farama-Foundation/d4rl.git@master` plus `mujoco-py==2.1.2.14`. Configure `LD_LIBRARY_PATH` per mujoco_py README.
- [ ] **T2.3** — In that venv, write `scripts/fetch_d4rl_hdf5.py`. Downloads 12 hdf5 files (halfcheetah/hopper/walker2d × {medium, medium-replay, medium-expert, expert} v2) from URLs in `Farama-Foundation/D4RL/d4rl/infos.py`.
- [ ] **T2.4** — Switch back to main venv. Write `scripts/d4rl_to_minari.py`. Reads each hdf5, creates a local Minari dataset under `localD4RL/<env>/<split>-v2`. Embeds `reference_min_score` / `reference_max_score` from the D4RL_REFS dict in `06_evaluation_protocol.md`.
- [ ] **T2.5** — Verify: `python -c "import minari; ds=minari.load_dataset('localD4RL/halfcheetah/medium-v2'); print(sum(1 for _ in ds.iterate_episodes()), 'episodes')"`. Sanity-check against D4RL published dataset sizes (HalfCheetah-medium-v2 ≈ 1M transitions ≈ 1k episodes).
- [ ] **T2.6** — Verify via shim: `DATASET_OPTION=B python -c "import gym; e=gym.make('halfcheetah-medium-v2'); d=e.get_dataset(); print(d['observations'].shape, d['rewards'].sum())"`.

## Phase 3 — Repo cloning (30 min)

- [ ] **T3.1** — Clone 7 active PyTorch repos:
  ```
  repos/ldcq
  repos/Efficient-Diffusion-Alignment
  repos/CEP-energy-guided-diffusion
  repos/diffuser
  repos/decision-diffuser
  repos/Diffusion-Policies-for-Offline-RL
  repos/CORL
  ```
- [ ] **T3.2** — Record each repo's `git rev-parse HEAD` to `RUN_LOG.md`.
- [ ] **T3.3** — For each repo, audit `import gym`, `import d4rl`, `import comet_ml`, `import wandb`, `import mujoco_py` lines. Most resolve to shims automatically.
- [ ] **T3.4** — For CORL: edit `requirements.txt` to remove `d4rl` and `gym` lines. Don't install CORL's deps separately — they should be a subset of our top-level deps.

## Phase 4 — Smoke tests (1 day; <5 min each × 11)

For each method, run a tiny training session on `halfcheetah-medium-v0` (Option C, fast). Verify MLflow run appears with `train.loss` metric.

- [ ] **T4.1** — CORL BC smoke (max_timesteps=100)
- [ ] **T4.2** — CORL CQL smoke
- [ ] **T4.3** — CORL IQL smoke
- [ ] **T4.4** — CORL TD3+BC smoke
- [ ] **T4.5** — CORL DT smoke
- [ ] **T4.6** — Diffuser smoke (n_train_steps=100)
- [ ] **T4.7** — Decision Diffuser smoke
- [ ] **T4.8** — Diffusion-QL smoke (a few hundred grad steps)
- [ ] **T4.9** — LDCQ smoke (1 epoch per stage, 2-episode eval)
- [ ] **T4.10** — Eval script for LDCQ: write `scripts/eval_ldcq_locomotion.py`. Test on smoke checkpoints.
- [ ] **T4.11** — EDA smoke (3 stages × 1 epoch each)
- [ ] **T4.12** — QGPO smoke (2 stages × 1 epoch each)

**Humanoid smoke** (subset — only methods that can run on Humanoid):
- [ ] **T4.13** — LDCQ smoke on `humanoid-medium-v0`
- [ ] **T4.14** — EDA smoke on `humanoid-medium-v0`
- [ ] **T4.15** — QGPO smoke on `humanoid-medium-v0`

**Gate:** All 15 smoke runs must log to MLflow with at least `train.loss`. Fix any failures before Phase 5.

## Phase 5 — Verify default configs (half day)

This phase ensures we are using REPO defaults, not editing them.

- [ ] **T5.1** — For each CORL method, confirm `configs/offline/<algo>/<env>/<split>.yaml` is being passed to the training script unchanged (`git diff configs/` should be empty).
- [ ] **T5.2** — For each non-CORL method, document in `RUN_LOG.md` what the default invocation is — the exact CLI command from the README or argparse defaults. No flag changes from there.
- [ ] **T5.3** — Write `scripts/run_one.py` that takes `(algo, env, split, seed)` and dispatches to the right repo's entrypoint. It should ONLY pass: env name, seed, and (where relevant) the path to the upstream config YAML. No hyperparameter flags.

## Phase 6 — Full training matrix

### Phase 6a — HalfCheetah/Hopper/Walker2d on Option B (1–3 weeks GPU time)

- [ ] **T6.1** — CORL × 5 methods × 3 envs × 3 splits × 3 seeds = 135 runs. ~1.5h each. Linear single-GPU: ~9 days.
- [ ] **T6.2** — Diffusion-QL × 3 envs × 3 splits × 3 seeds = 27 runs × ~6h = ~7 days.
- [ ] **T6.3** — Diffuser × 3 envs × 3 splits × 3 seeds = 27 runs × ~10h = ~11 days.
- [ ] **T6.4** — Decision Diffuser — limited by config. Run only `hopper-medium-expert-v2` × 3 seeds = 3 runs unless env override is exposed.
- [ ] **T6.5** — LDCQ × 3 envs × 3 splits × 3 seeds = 27 runs × ~15h = ~17 days. (Multi-stage; MLflow run count = 27 × 4 stages = 108.)
- [ ] **T6.6** — EDA × 3 envs × 3 splits × 3 seeds = 27 runs × ~9h. (MLflow run count = 27 × 3 stages = 81.)
- [ ] **T6.7** — QGPO × 3 envs × 3 splits × 3 seeds = 27 runs × ~10h. (MLflow run count = 27 × 2 stages = 54.)

### Phase 6b — Humanoid on Option C (1 week GPU time)

- [ ] **T6.8** — LDCQ × Humanoid × 2 splits × 3 seeds = 6 runs × ~20h ≈ 5 days. (Larger state space → slower.)
- [ ] **T6.9** — EDA × Humanoid × 2 splits × 3 seeds = 6 runs × ~12h ≈ 3 days.
- [ ] **T6.10** — QGPO × Humanoid × 2 splits × 3 seeds = 6 runs × ~14h ≈ 3.5 days.
- [ ] **T6.11** — Your method × Humanoid × 2 splits × 3 seeds.

**Monitoring (daily during Phase 6):**
- `make mlflow_status` daily — restart if container dies.
- Spot-check 1 run per day: `train.loss` decreasing, `eval.normalized_score_mean` non-NaN.
- If a run NaN-crashes: tag `reproduction_health=crashed`, move on. No tuning.

## Phase 7 — Aggregation and reporting (1 day)

- [ ] **T7.1** — Write `scripts/build_results_table.py` per `06_evaluation_protocol.md`.
- [ ] **T7.2** — Run it; produce `results.csv` and `results.md`.
- [ ] **T7.3** — Annotate degraded/crashed runs in `results.md` with footnote.
- [ ] **T7.4** — Add a "Cited-from-paper" appendix table with BCQ/SfBC/IDQL numbers footnoted as "not reproduced — values from source paper" — these aren't in our MLflow.
- [ ] **T7.5** — Generate comparison plots: bar chart per task-dataset, methods sorted by score. Save as `results.png`.
- [ ] **T7.6** — Commit `results.csv`, `results.md`, `results.png`, `RUN_LOG.md`.

## Out-of-band (no gating)

- [ ] **OOB-1** — Bump seed count 3→5 for top-3 methods per task once initial matrix completes.
- [ ] **OOB-2** — Bump eval episodes 10→100 for top-3 methods to tighten error bars.
- [ ] **OOB-3** — Generate Humanoid `medium-replay`/`medium-expert` ourselves if Humanoid coverage needs expanding (~3 GPU-days; only if needed).
- [ ] **OOB-4** — Add IDQL via separate JAX env if needed.

## File layout at the end

```
~/offline_rl_repro/
├── compat/                              # our shim package
├── repos/                               # 7 cloned repos, UNTOUCHED
├── scripts/
│   ├── prepare_data_ldcq.py
│   ├── fetch_d4rl_hdf5.py               # in .venv-d4rl
│   ├── d4rl_to_minari.py                # in main venv
│   ├── eval_common.py
│   ├── eval_ldcq_locomotion.py          # the one new eval script
│   ├── run_one.py                       # universal launcher
│   ├── run_corl.py
│   ├── run_diffuser.py
│   ├── ... (per-method launchers)
│   └── build_results_table.py
├── docker/                              # MLflow compose (pre-existing)
├── pyproject.toml                       # or pixi.toml
├── Makefile
├── RUN_LOG.md                           # decisions, GPU/driver, repo SHAs, run timings
├── results.csv
├── results.md
└── results.png
```
