# Baselines Reproduction — RUN_LOG

Project root: `/home/jash/diffusion-rl-research-02/baselines/`
Started: 2026-05-19

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (32607 MiB) |
| NVIDIA driver | 595.58.03 |
| System Python | 3.13.12 |
| Project Python (planned) | 3.11 via `uv venv` |
| uv | 0.11.7 |
| MLflow tracking URI | http://localhost:5555 (HTTP 200, container `docker-mlflow-1`, up 2 weeks) |
| MLflow experiment | `latent_cep_baselines` (matrix), `latent_cep_baselines_smoke` (autoresearch). Renamed 2026-05-19 from the plan's literal `baselines` to avoid colliding with 15 pre-existing trading-backtest runs already in that experiment on this host. |
| Postgres (MLflow backend) | container `docker-postgres-1`, up 3 weeks |
| Toolchain choice | **uv** (pixi available but uv selected per plan default) |
| Host OS | Linux 7.0.0-14-generic |

The MLflow container is a **shared service** — `make mlflow_up`/`mlflow_down` are intentionally not invoked by this reproduction; we only set `MLFLOW_TRACKING_URI=http://localhost:5555` and write to experiment `latent_cep_baselines` (long matrix) / `latent_cep_baselines_smoke` (smoke autoresearch). (Per the project-wide "do not restart shared services" rule.)

## Decisions (D1–D8 per docs/08)

| ID | Decision |
|---|---|
| D1 (HC/Hop/Walker dataset) | **Option C this session** (Minari-native `medium-v0` / `expert-v0`). Option B (D4RL hdf5 → local Minari) explicitly deferred to a later session. |
| D2 (seed count) | 3 (Phase 6 only; smokes are 1 seed). |
| D3 (eval episodes) | 10 for final, 2 for smokes. |
| D4 (CORL on Humanoid) | Strict: skip — no Humanoid YAML ships in CORL. |
| D5 (Diffuser/DD on Humanoid) | Skip — no Humanoid block in their configs. |
| D6 (cited-only baselines) | BCQ, SfBC, IDQL — cited from papers, not reproduced. |
| D7 (EDA β) | `--beta 0.1` for every task (README default). Tag `tuning=none_default`. |
| D8 (DQL resampling) | Keep 50-candidate resample (`eval_mode=stoch_resample_50`). |

## Active baseline shortlist (this session)

1. **CORL × 5**: bc, cql, iql, td3bc, dt
2. **Diffuser** (jannerm)
3. **Decision Diffuser** (anuragajay)
4. **Diffusion-QL** (Zhendong-Wang)
5. **LDCQ** (ldcq)
6. **EDA** (thu-ml/Efficient-Diffusion-Alignment)
7. **QGPO** (thu-ml/CEP-energy-guided-diffusion)

Skipped: BCQ, SfBC, IDQL (cite-only). "Your method" row is out of scope this session.

## Repo SHAs

Recorded after Phase 3 clone.

| Repo | Path | git rev-parse HEAD |
|---|---|---|
| Repo | Path | Upstream HEAD (frozen) | Local re-init HEAD |
|---|---|---|---|
| CORL | repos/CORL | `eeeeef2671235fcce3a3eb255576a207d78840e6` | `c9cc4e58f4418fbad07861eb4bc2519d50c9776e` |
| ldcq | repos/ldcq | `7ec96f3682e04e89385fe18e17a20f5315f0048d` | `27821a531b8cd97a5207a6e83dc978a56865c4e0` |
| Efficient-Diffusion-Alignment | repos/Efficient-Diffusion-Alignment | `2f969292590354b7fb0eb033c1ad75ba4293c63a` | `16a6ca08774c2d277b21cba7f424ed08df40128b` |
| CEP-energy-guided-diffusion | repos/CEP-energy-guided-diffusion | `4cc0622fbe6a032df3e2fd661c5adcdc69cf4b0f` | `e05e32a5ce8492f283f6eb6d50c97f83eb70cda2` |
| diffuser | repos/diffuser | `7ea422860cc0106e5ca5949d980f04b799d5462c` | `80dcccff26a55934c868deb44417100c9726a845` |
| decision-diffuser | repos/decision-diffuser | `01ce528c30b4733dc59aa6203e46ec165561158d` | `70831826c31990f35ddf173d16b93515b536b6d6` |
| Diffusion-Policies-for-Offline-RL | repos/Diffusion-Policies-for-Offline-RL | `d871f5c6b4a3a3a19a10c662a54f32d5819dfcdb` | `3e0d0925c1a7cb18de296818405492be2216215f` |

Repos were originally `git clone --depth=1` from upstream on 2026-05-19. On the same day, after the CORL requirements.txt edit, the upstream `.git/` was removed and each repo was `git init -b main`-ed with a single snapshot commit. From here on, *every* commit in each repo represents an autonomous patch applied by this reproduction effort. Upstream HEAD is frozen above as ground truth.

## Patch log (autonomous edits inside repos/)

User policy (2026-05-19): permissive — autonomous fixes to `repos/*` are allowed, but
  - **mathematical algorithms must not be modified** (no changing diffusion update rules, Q-learning targets, loss formulations, etc.)
  - helper functions, utility functions, and call signatures *may* be changed
  - if model function calls need to be changed, they must stay mathematically equivalent
  - every edit goes in this table with function name + full path + rationale

| # | Date | Repo | File | Symbol(s) | Kind | Why |
|---|---|---|---|---|---|---|
| 1 | 2026-05-19 | CORL | `requirements/requirements.txt` | n/a (dep list) | dependency | Removed `d4rl` and `gym` lines to avoid shadowing the compat-shim packages. Pre-approved in user instructions. Baked into the local repo's initial commit. |
| 2 | 2026-05-19 | ldcq (local) | `.gitignore` | n/a | local-only | Existing repo `.gitignore` already covers `*.pkl` — no change applied. |
| 3 | 2026-05-19 | compat (own code) | `compat/src/compat_utils/env_factory.py` | `_attach_d4rl_methods_to_inner`, `make_d4rl_compatible_env` | helper | B1 fix. Diffuser's `diffuser/datasets/d4rl.py` does `env = wrapped_env.unwrapped` then `env.get_dataset()`. Without attaching `get_dataset`/`get_normalized_score`/`spec_d4rl_name` to the inner Gymnasium env too, that call AttributeErrors. Mathematical invariant preserved — same dataset, same scoring. |
| 4 | 2026-05-19 | scripts (own code) | `scripts/run_corl.py` | smoke-config override block | helper | B4 fix. DT's CORL config uses `train_seed`/`eval_seed` keys, not `seed`. Override now writes all three, so `--seed 1/2` actually changes DT's training seed. Same effect as the existing `seed` override for the other 4 CORL algos. |
| 5 | 2026-05-19 | scripts (own code) | `scripts/run_ldcq.py` | `STAGE_TO_SCRIPT`, `_find_latest_skill_ckpt`, smoke-arg block | helper | B3 fix part 1. Added `collect` stage that runs `training/collect_q_learning_dataset.py` after `skills`. Stages `collect`/`diffusion`/`q_net` now auto-discover the most recent `checkpoints/skill_model_<env>_*_best.pth` if no `--skill_model_filename` provided. Returns rc=65 if no checkpoint exists (signals "run skills first"). |
| 6 | 2026-05-19 | scripts (own code) | `scripts/autoresearch_smoke.py` | `SMOKE_METHODS`, `run_all` pre-skip block, `LAUNCHER_RC_*` | orchestration | (a) Removed DD from active smoke list — needs `params_proto`+`ml_logger`+`jaynes` (not installed) AND Option-B-only `hopper-medium-expert-v2` data. Auto-tagged as `skipped` with explicit reason. Revisit in Phase 2b. (b) Added LDCQ `collect` stage between `skills` and `diffusion`. (c) Recognize rc=65 = `awaiting_prereq` (e.g. earlier stage hasn't produced its checkpoint yet) so the operator can fix-and-rerun without it looking like a crash. |
| 7 | 2026-05-19 | compat (own code) | `compat/src/gym/__init__.py`, `compat/src/gym/wrappers.py` | `gym.wrappers.{TransformObservation, TransformReward}` re-exports | helper | CORL's `any_percent_bc.py:94` calls `gym.wrappers.TransformObservation(env, func)` (2-arg gym-0.21 form), but Gymnasium's wrapper now requires a 3rd `observation_space` arg. New `compat/src/gym/wrappers.py` exposes a `TransformObservation` subclass that defaults `observation_space` to `env.observation_space`. All other wrappers re-exported as-is from `gymnasium.wrappers`. |
| 8 | 2026-05-19 | compat (own code) | `compat/pyproject.toml` | `torch` pin | dependency | RTX 5090 is Blackwell (sm_120). Plan's `torch>=2.4,<2.7` cap predates Blackwell — only torch>=2.7 ships sm_120 kernels (verified by `torch.cuda.get_arch_list()`). Loosened pin to `torch>=2.7`. Installed torch 2.11.0+cu128 (same family as latent-cep-rl venv, which has 2.11.0+cu130). CUDA op test passes. |
| 12 | 2026-05-19 | diffuser (local) | `diffuser/utils/config.py:21` | `Config` class | helper | `collections.Mapping` was removed in Py3.10; replaced with `collections.abc.Mapping`. No behavior change. |
| 13 | 2026-05-19 | diffuser (local) | `diffuser/datasets/buffer.py:12` | `ReplayBuffer.__init__` | helper | `np.int` removed in numpy ≥1.24; replaced with `np.int64`. Same precision. |
| 14 | 2026-05-19 | DQL (local) | `main.py:38` | `hyperparameters` dict | dependency-style | Added v0 hyperparameter rows (halfcheetah/hopper/walker2d × {medium, expert}-v0) by cloning the corresponding v2 row. Required because `main.py` looks up hyperparameters by env_name keyword; v0 keys weren't in the upstream table. No tuning — values copied 1-to-1 from v2. |
| 15 | 2026-05-19 | EDA (local) | `train_critic.py:46` | save in `train_critic()` | helper | Save `score_model.q[0].state_dict()` (the IQL_Critic submodule) instead of `score_model.state_dict()` so `finetune_policy.py:74`'s `score_model.q[0].load_state_dict(ckpt)` finds the keys it expects. Upstream had this commented out on the previous line — restored. |
| 16 | 2026-05-19 | EDA (local) | `BDiffusion.py:113-119` | `BDiffusion_Policy.update_policy` | gradient-flow bug | `xt_model_energy = beta * (diffusion_policy.get_energy(...) - baseline)` was inside `with torch.no_grad():`, detaching the policy energy from the autograd graph and making `loss.backward()` raise *"element 0 does not require grad"*. Moved out so the policy gradient flows. `baseline` (frozen behavior model) stays under no_grad. **Math invariant — same formula, just with the gradient correctly tracked.** |
| 17 | 2026-05-19 | QGPO/CEP (local) | `train_behavior.py:31` | `train_behavior()` | helper | Save `behavior_ckpt0.pth` once before the inner epoch loop so very-short smokes still produce an artifact for the critic stage to load. Upstream save schedule (every 100 epochs + at epoch 599) unchanged; long-training artifacts identical. |
| 18 | 2026-05-19 | compat (own code) | `compat/src/compat_utils/env_factory.py` | `OldGymEnvWrapper.__getattr__` | helper | Gymnasium ≥1.0 dropped `Wrapper.__getattr__`; restored old gym-0.21 attribute forwarding so Diffuser's `wrapped_env._max_episode_steps` and CORL's `env.seed(seed)` on the outermost wrapper fall through to the inner TimeLimit/wrapper layers. |
| 19 | 2026-05-19 | compat (own code) | `compat/src/compat_utils/env_factory.py` | `_attach_d4rl_methods_to_inner` | helper | Also attach `seed` to the unwrapped inner env (Diffuser's `sequence.py:23` calls `self.env.seed(seed)` after `env.unwrapped`). |
| 20 | 2026-05-19 | compat (own code) | `compat/src/compat_utils/env_factory.py` | `_FULLOBS_FALLBACK` | mapping | Map Diffuser's `HalfCheetahFullObs-v2 / HopperFullObs-v2 / Walker2dFullObs-v2 / AntFullObs-v2` to base Gymnasium env IDs since the FullObs subclasses can't be registered through our `gym.register`-less shim. Rendering paths get a working env; eval geometry is identical to base. |
| 21 | 2026-05-19 | compat (own code) | `compat/src/gym/__init__.py`, `compat/src/gym/wrappers.py` | `gym.wrappers.{TransformObservation, TransformReward}` | helper | Wrappers re-export with `_AttrForwardMixin` (old gym-0.21 `__getattr__`) + bilingual `reset`/`step` (consume old-API 4-tuple from `OldGymEnvWrapper` underneath, emit either old or new API based on outer caller). |
| 22 | 2026-05-19 | compat (own code) | `compat/src/compat_utils/wandb_stub.py` | `init()`, `finish()`, `_ConfigStub.update()` | helper | `init()` sets `sys.modules["wandb"].run = _StubRun()` so CORL's `wandb.run.save()` works; `finish()` clears it. `_ConfigStub.update()` accepts `argparse.Namespace` via `vars()` for EDA's `wandb.config.update(args)`. |
| 23 | 2026-05-19 | scripts (own code) | `scripts/autoresearch_smoke.py` | `_has_mlflow_metrics`, `LOSS_REGEX` | orchestration | Made the green-detection MLflow-fallback retry up to 5× with backoff (absorbs eventual-consistency lag between subprocess SIGTERM and parent REST query). Broadened LOSS_REGEX to also match `Training Start`, `Loaded buffer`, `samples extracted`, `Average <fld>:`, `Iteration N`, `step N` etc. — required because several methods log only to tensorboard or print only tqdm bars. |
| 24 | 2026-05-19 | scripts (own code) | `scripts/run_ldcq.py` | stage chaining + auto-discovery | orchestration | `collect` stage now runs BOTH `collect_q_learning_dataset.py` AND `collect_offline_q_learning_dataset.py` (the second produces `_sample_latents.npy` for q_net). Smoke caps `--num_diffusion_samples 10` to avoid 116 GB allocation. `--device cuda` only added for stages whose argparse accepts it (skills uses `--env_name`, hardcoded cuda). Also adds `repos/ldcq/training/` to `sys.path` so `from per_utils import ...` resolves. |
| 25 | 2026-05-19 | scripts (own code) | `scripts/run_corl.py` | smoke-config override | helper | Set `warmup_steps = min(10, update_steps)` for DT (zero would zero-divide in the lr-lambda); only override `seed`/`train_seed`/`eval_seed` if present in the YAML (pyrallis rejects unknown fields like setting `seed` on DT). |
| 26 | 2026-05-19 | autoresearch (decision) | `scripts/autoresearch_smoke.py` SMOKE_METHODS, `scripts/long_runner.py` manifest | LDCQ removed entirely | scope | User decision 2026-05-19: drop LDCQ from the baselines matrix. LDCQ's `train_skills.py` eager-chunks every transition × horizon × obs_dim into RAM (~40 GB for Humanoid-medium-v0 = 965426 × 30 × 348 × float32). OOM-killed on this 62 GB host. To be reproduced on a higher-RAM machine outside this session. Smoke-evidence preserved in MLflow experiment `latent_cep_baselines_smoke` for forensics. |
| 27 | 2026-05-19 | autoresearch (operational) | n/a | OOM during `ldcq:humanoid-medium-v0:skills` | incident | Out-of-Memory killer terminated the smoke loop and the parent autoresearch orchestrator. Root cause: LDCQ chunk-extraction RAM blow-up (see patch #26). RAM has been verified clean after restart (58 GB available). No data loss — state file `state/autoresearch.json` was already atomic. Patch #26 prevents recurrence. |
| 28 | 2026-05-19 | scripts (own code) | `scripts/long_runner.py` `_default_manifest()` | DD entry removed | scope | Same reasoning as the LDCQ drop (patch #26). DD's matrix cells need `params_proto`+`ml_logger`+`jaynes` and Option-B `hopper-medium-expert-v2`, neither of which we have. Cells would only crash. Matrix manifest now 246 cells / 1131 hrs serial. |
| 29 | 2026-05-19 | scripts (own code) | `scripts/long_runner.py` `cmd_execute` | bin-packing scheduler | helper/orchestration | Replaced the serial loop with a footprint-driven bin-packing scheduler. Default ordering is **easy-first** (ascending est_hours; quick wins first, heaviest cells run alone at the end). Budget: 28 GB vRAM, 45 GB system RAM; per-method vRAM/RAM × 1.3 safety multiplier; +1.8× Humanoid scale. **OOM-safety pre-check**: cells held when `MemAvailable < 5 GB` or `GPU memory.free < 2 GB` (prevents recurrence of patch #27). Per-process env caps: `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. |
| 30 | 2026-05-19 | scripts (own code) | `scripts/long_runner.py` `cmd_reassign_pool`, `cmd_export_pool`, `--pool` flag | multi-host split | helper | Every cell carries a `pool` field (default `"default"`). Scheduler `--pool <name>` filters to that pool only. `--reassign <pattern> --to_pool <name>` reassigns matching cells; `--export_pool` writes a portable manifest for another server. Cells in non-default pools are marked `delegated` locally after export. Makefile targets: `make matrix POOL=`, `make matrix-reassign`, `make matrix-export`. |

(Future autonomous patches from Phase 4's autoresearch loop append here.)
| 9 | 2026-05-19 | compat (own code) | `compat/src/gym/wrappers.py` | `_AttrForwardMixin`, `TransformObservation`, `TransformReward` | helper | Gymnasium ≥1.0 dropped `Wrapper.__getattr__`, so `env.seed(seed)` from CORL on the outermost wrapper no longer falls through to our `OldGymEnvWrapper`. Mixin restores gym-0.21-era attribute forwarding. Both wrappers also override `reset`/`step` to be **bilingual**: detect tuple-arity to handle our inner old-API 4-tuple while passing the right shape upward. No mathematical change. |
| 10 | 2026-05-19 | compat (own code) | `compat/src/compat_utils/wandb_stub.py` | `init()`, `finish()` | helper | CORL's `any_percent_bc.py:183` does `wandb.run.save()` after `wandb.init()`. Stub previously left `mod.run = None` after install(); `init()` now sets `sys.modules["wandb"].run = _StubRun()` and `finish()` clears it. Forwarding semantics (wandb.log → mlflow.log_metric) unchanged. |
| 11 | 2026-05-19 | scripts (own code) | `scripts/autoresearch_smoke.py` | `LOSS_REGEX`, `_has_mlflow_metrics()`, `run_one()` saw_loss check | orchestration | False negatives from regex-only green detection (CORL logs `actor_loss` through wandb.log → mlflow.log_metric but never prints "loss" to stdout). Added MLflow query fallback: green if a recent run with matching algo/env/stage tags has ≥1 metric. Also broadened the regex to match `actor_loss`/`Time steps:`/`Evaluation over`/`d4rl_normalized_score`/`epoch N`. |

(Future autonomous patches from Phase 4's autoresearch loop append here.)

## Smoke status (Phase 4 — blocked on GPU availability)

| # | Method | Env | Status | MLflow run | Notes |
|---|---|---|---|---|---|
| 1 | CORL BC | halfcheetah-medium-v0 | pending | – | – |
| 2 | CORL CQL | halfcheetah-medium-v0 | pending | – | – |
| 3 | CORL IQL | halfcheetah-medium-v0 | pending | – | – |
| 4 | CORL TD3+BC | halfcheetah-medium-v0 | pending | – | – |
| 5 | CORL DT | halfcheetah-medium-v0 | pending | – | – |
| 6 | Diffuser | halfcheetah-medium-v0 | pending | – | – |
| 7 | Decision Diffuser | (hopper-medium-expert-v2 hard-coded) | pending | – | DD's `default_inv.py` is task-pinned. Will smoke on its own default. |
| 8 | Diffusion-QL | halfcheetah-medium-v0 | pending | – | – |
| 9a | LDCQ stage 1 (skills) | halfcheetah-medium-v0 | pending | – | – |
| 9b | LDCQ stage 2 (diffusion) | halfcheetah-medium-v0 | pending | – | – |
| 9c | LDCQ stage 3 (q-net) | halfcheetah-medium-v0 | pending | – | – |
| 9d | LDCQ stage 4 (eval) | halfcheetah-medium-v0 | pending | – | Uses `scripts/eval_ldcq_locomotion.py`. |
| 10a | EDA stage 1 (behavior) | halfcheetah-medium-v0 | pending | – | – |
| 10b | EDA stage 2 (critic) | halfcheetah-medium-v0 | pending | – | – |
| 10c | EDA stage 3 (finetune) | halfcheetah-medium-v0 | pending | – | – |
| 11a | QGPO stage 1 (behavior) | halfcheetah-medium-v0 | pending | – | – |
| 11b | QGPO stage 2 (critic) | halfcheetah-medium-v0 | pending | – | – |
| 12 | LDCQ | humanoid-medium-v0 | pending | – | – |
| 13 | EDA | humanoid-medium-v0 | pending | – | – |
| 14 | QGPO | humanoid-medium-v0 | pending | – | – |

## Notes / known caveats

- **DD task rigidity.** Decision Diffuser's `default_inv.py` hard-codes `hopper-medium-expert-v2`. Per "no edits," DD will only smoke on that task — not halfcheetah-medium-v0. Documented in docs/08 R6.
- **LDCQ chunk-filter threshold** at `utils/utils.py:49` (`np.all(norms <= 0.8)`) is AntMaze-tuned. Per docs/08 R4: do NOT modify; tag `data_loss_warning=ldcq_chunk_filter` if >50%.
- **CORL Humanoid** is intentionally absent (D4 strict). Same for Diffuser/DD on Humanoid (D5).
- **PyTorch 2.6+** changed `torch.load` default to `weights_only=True`. Compat layer installs a `_patched_load` to restore `weights_only=False`.
- **`numpy<2.0` pin** is mandatory (docs/08 R7) for sklearn-driven imports inside EDA.
- **Humanoid obs space is 348-dim** in Gymnasium-v5, not the 376 quoted in plan docs (v4 vs v5 difference). Recorded for downstream sanity checks.
- **Reference scores in Minari are `None`** for all `mujoco/*` datasets in v0 metadata. Our shim's `get_normalized_score()` falls back to `(0.0, 1.0)` (raw return passthrough) when refs are missing. For Humanoid this is fine (no paper baseline anyway). For HC/Hop/Walker we'll embed D4RL refs in Option B (deferred).

## Phase 1 / Phase 2a verification (completed 2026-05-19)

- `compat-shim` editable-installed into `.venv` (Python 3.11.15). torch==2.6.0, numpy==1.x, mlflow installed, gymnasium[mujoco], minari[hf]>=0.5.
- Shim sanity:
  - `gym.make('halfcheetah-medium-v0')` → `OldGymEnvWrapper<TimeLimit<...HalfCheetah-v5>>`, obs=Box(17,), act=Box(6,).
  - `gym.make('humanoid-medium-v0')` → obs=Box(348,), act=Box(17,).
- Data sanity:
  - `d4rl.qlearning_dataset(gym.make('halfcheetah-medium-v0'))` returned 1,000,000 transitions; rewards sum 12,089,212, mean 12.089; timeouts=1000, terminals=0.
  - `env.get_dataset()` on `humanoid-medium-v0` returned 999,153 transitions; rewards mean 8.165.
- **All 12 Option C Minari IDs validated** (4 envs × {simple, medium, expert} v0). Datasets cached under `~/.minari/`.

## Machine 2 (helix, 10.21.186.72) bring-up + logging fixes — 2026-05-22

Second-machine setup (RTX 5090, sm_120, torch 2.11.0+cu128). MLflow reached via
SSH tunnel localhost:5555 -> shared server (Makefile left at committed
localhost:5555). All edits below are telemetry-only / orchestration; no
hyperparameter or algorithm changes. Repo edits are math-preserving.

**Orchestration / compat (scripts/, compat/ — not repos/):**
- #29 `scripts/bootstrap.sh`: add `--index-strategy unsafe-best-match` to the
  requirements.lock install — uv's default first-index strategy couldn't resolve
  the lockfile across PyPI + the cu128 index (certifi/pyrallis pinned on PyPI).
  Versions unchanged.
- #30 `compat/.../wandb_stub.py`: `_StubRun.log` was a no-op via `__getattr__`,
  silently dropping every `run.log(...)` (EDA logs through the run object, not
  module-level `wandb.log`). Added a forwarding `log` method. Verified: EDA
  `loss.diffusion` now lands in MLflow.
- #31 `compat/.../tb_shim.py` (NEW) + `_launch_common.install_tb_shim`: drop-in
  `SummaryWriter` forwarding add_scalar/add_scalars -> mlflow.log_metric
  (buffered). QGPO/DQL log via TensorBoard. Verified: qgpo `actor.loss`, dql
  `BC Loss`/`QL Loss`/`Critic Loss` land in MLflow.
- #32 `compat/.../d4rl_dict_converter.get_normalized_score`: was reading
  `ds.spec.reference_min/max_score` (None for all Minari datasets) -> fell back
  to (0,1) -> returned RAW. EDA/QGPO normalize via this module fn while
  CORL/DQL use `env.get_normalized_score`. Delegated it to the same
  `_minari_ref_scores`, so all methods share one normalization. Verified both
  paths agree (29.55 @ raw=4800 halfcheetah-medium).
- #33 `scripts/autoresearch_smoke.py::_has_mlflow_metrics`: queried
  `tags.env=<full d4rl name>` but launchers tag base env + dataset -> 0 matches
  -> false-red for tqdm-only methods (DT). Now derives base env + dataset.
- #34 `scripts/run_qgpo.py`: install_tb_shim; smoke-only env
  `BASELINES_SMOKE_QGPO_ITERS=50` so the critic smoke reaches its epoch-0 eval.
- #35 `scripts/run_dql.py`: install_tb_shim; set env `BASELINES_DQL_SMOKE` for
  smoke. IMPORTANT: DQL runs main.py via `runpy.run_path` (no main() fn) -> a
  FRESH namespace, so launcher-side patches to the imported module
  (hyperparameters, eval_policy) are NOT seen by the run (verified empirically).
  Therefore DQL's score logging + smoke truncation live in main.py (#39), not as
  launcher monkeypatches.
- #36 `scripts/long_runner.py` + `state/matrix.json`: dropped diffuser from the
  matrix (228 cells). Its single-stage "full" cell only ran train.py; the score
  comes from plan_guided.py (a stage the matrix never ran). Excluded like
  LDCQ/DD rather than restructured. (User decision 2026-05-22.)

**Repo edits (repos/* — telemetry only, matrix math unchanged):**
- #37 CEP `Offline_RL_2D/train_critic.py`: log `d4rl_normalized_score_gs{gs}`
  (= d4rl.get_normalized_score(env, raw_eval_return)*100) beside the existing
  raw `eval/rew{gs}`. Per user "log all 7 guidance scales, no single headline".
  Also: inner critic-update loop count reads `BASELINES_SMOKE_QGPO_ITERS`
  (default 10000 -> matrix unchanged; smoke uses 50).
- #38 EDA `finetune_policy.py`: `mean` from pallaral_simple_eval_policy is the
  D4RL-normalized fraction; log `d4rl_normalized_score` (= mean*100) at step 0
  and per eval epoch.
- #39 DQL `main.py`: (a) `writer = None` -> `SummaryWriter(output_dir)` so the
  agent's training scalars flow through the tb_shim; (b) after eval,
  `writer.add_scalar("d4rl_normalized_score", eval_norm_res*100, curr_epoch)`
  (+ raw return); (c) env-gated `BASELINES_DQL_SMOKE` truncation
  (num_epochs/eval_freq/num_steps_per_epoch/eval_episodes) so the smoke reaches
  eval. Verified d4rl_normalized_score=-3.01 (untrained smoke policy, sane
  magnitude). NOTE: DQL v0 eval default is 100 episodes of stoch_resample_50 in
  the matrix -> slow; left at repo default (changing it = methodology).

**Verified (smoke, latent_cep_baselines_smoke exp):** wandb->mlflow (EDA loss),
tb->mlflow (qgpo/dql losses), normalization consistency, EDA
`d4rl_normalized_score`, QGPO `d4rl_normalized_score_gs{gs}`. NOTE: smoke eval
*values* are from untrained 1-epoch checkpoints (garbage, e.g. large negatives)
— only the logging path is verified; real matrix values to be sanity-checked
against reference_scores.json early per docs/06 (<30%-of-expected check).

**Known follow-ups:** DQL v0 eval default is 100 episodes (slow); QGPO compare
cell intentionally blank (gs-suffixed keys, no headline) — dashboard could be
extended to surface per-gs scores later.

## Raw + reference-score logging (2026-05-23, machine 2)

To make every score losslessly convertible (raw <-> Minari-normalized <-> D4RL-
normalized) without re-running:
- #40 `scripts/_launch_common.mlflow_start`: log `ref_min`, `ref_max`,
  `ref_source=minari_expert_mean`, `norm_scheme=minari_v0` as run tags (uniform,
  all methods); hand the refs to the metric shims via set_refs().
- #41 `compat/.../wandb_stub.py` + `tb_shim.py`: `set_refs()` + when any
  `d4rl_normalized_score*` metric is forwarded, also emit the inverted
  `raw_return*` (= norm/100*(ref_max-ref_min)+ref_min). Covers CORL/EDA (wandb
  path) and QGPO/DQL (tensorboard path) with no per-repo edits; DQL/QGPO also
  keep their native raw metrics. Verified live: bc logs d4rl_normalized_score +
  raw_return together; QGPO gs variants too.
- Operational: restarted the in-flight cells so they pick up the new logging
  (freeze oom_watchdog+matrix-monitor via SIGSTOP -> kill+reset running->pending
  -> make matrix -> SIGCONT). Deleted 20 KILLED runs (power-down + restart
  orphans); exp40 now holds only live RUNNING + future FINISHED.
- Minari refs (deterministic, for offline conversion): halfcheetah (0,16242.9),
  hopper (0,3857.8), walker2d (0,6847.8), humanoid (0,8602.9). D4RL refs:
  halfcheetah (-280.18,12135), hopper (-20.27,3234.3), walker2d (1.63,4592.3).

## Dropped TD3+BC (2026-05-23, user: old algorithm)

- #42 Removed `td3bc` from `long_runner` CORL generator (re-seeds exclude it) +
  added `state/dropped_algos.json` ({"dropped":["td3bc"]}) read by
  `run_corl.py`, which now returns rc=64 (skip) for any dropped algo BEFORE
  mlflow_start. This excludes td3bc from the *running* orchestrator (which holds
  its pending list in memory) without a restart: its 18 queued cells self-skip
  (-> status skipped, no MLflow runs) as the scheduler reaches them. Effective
  matrix: 228 -> 210 cells. Verified: a td3bc launch returns rc=64 instantly.
