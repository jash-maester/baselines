# 08 — Risks and Open Decisions

## Open decisions — resolve before starting

### D1. Dataset option for HalfCheetah/Hopper/Walker2d

Pick one (or both, in phases) from `02_dataset_situation.md`:

| | Effort | Comparability to papers | Risk |
|---|---|---|---|
| **B** (fetch original D4RL hdf5, convert) | ~1 day | High | Medium — `mujoco_py` install pain |
| C (Minari-native only) | 0 | Very low (only `medium`/`expert`, no `medium-replay`/`medium-expert`) | Low |

**Recommendation:** Option B for the final HalfCheetah/Hopper/Walker2d table, Option C for Humanoid (no choice) and smoke. Document in `RUN_LOG.md`.

### D2. Seed count

3 seeds minimum. Pick:
- **3** — fast, paper-minimum, recommended
- 5 — paper-standard
- 10 — overkill for an initial table

**Recommendation:** 3 for first matrix; bump top-3 methods to 5 in OOB phase.

### D3. Eval episodes per run

- **10** — fast (default)
- 50 — D4RL paper-standard
- 100 — LDCQ paper-standard

**Recommendation:** 10 during matrix; rerun top performers at 100 in OOB phase.

### D4. CORL methods on Humanoid

CORL ships no Humanoid config. Two interpretations of "no modifications":

(a) **Strict (recommended):** Do not add Humanoid configs. Run CORL methods on HalfCheetah/Hopper/Walker2d only. Humanoid column has cells filled only for LDCQ/EDA/QGPO + your method.

(b) **Relaxed:** Copy `configs/offline/cql/halfcheetah/medium_v2.yaml` to `humanoid/medium_v0.yaml`, change only env name + dataset id. This is a "configuration extension," not a hyperparameter change.

**Recommendation:** (a). The cleaner story is "the only methods that ran on Humanoid are the ones whose default config is env-agnostic."

### D5. Diffuser/DD on Humanoid

Same situation as D4. Diffuser's `config/locomotion.py` and DD's `default_inv.py` don't include Humanoid. Skip Humanoid for Diffuser and DD. Document.

### D6. Cited-only baselines

Three baselines are not run (per `01_baselines_catalog.md`):
- **BCQ** — weak baseline, old repo
- **SfBC** — dominated by QGPO
- **IDQL** — JAX, separate stack

Their numbers come from source papers and appear in `results.md` with a "from paper, not reproduced" footnote. Confirm this is acceptable, or override to include any of them.

### D7. EDA β per task

The EDA paper's Appendix E Table 2 lists per-task β values (HalfCheetah-ME=2.0, HC-M=0.1, etc.). The repo's default invocation (from README) uses `--beta 0.1` regardless.

Per "no hyperparameter tuning": **use β=0.1 for every task.** EDA's medium-expert numbers will likely be lower than paper. Tag with `tuning=none_default`. Acceptable.

If you want paper-faithful EDA numbers: this is hyperparameter tuning by the strict interpretation. Pick one and document.

### D8. Diffusion-QL resampling

D-QL evaluates by sampling 50 candidates and picking argmax-Q — part of the default invocation. Keep it. Tag `eval_mode=stoch_resample_50`.

Some baselines report a `D-QL@1` variant (no resampling). Per "default configs only," **do not add the @1 variant**. The repo's default is resampling.

## Risks

### R1. mujoco_py install for Option B

`mujoco_py==2.1.2.14` ships with `mujoco210` and requires `gcc`, `patchelf`, `libosmesa6-dev`. On Ubuntu 22.04+ it generally works. On macOS/M1 it does not — Option B needs a Linux host.

**Mitigation:** Use a Docker container based on `python:3.9-slim` with mujoco_py deps for the fetch step. Mount the host's `~/.d4rl/datasets/` dir. Or just use Option C for HC/Hop/Walker if compute pain outweighs paper-comparability.

### R2. Minari dataset version skew

`minari.load_dataset("mujoco/humanoid/medium-v0")` requires Minari version compatibility. If Farama bumps to `-v1`, the mapping breaks.

**Mitigation:** Pin Minari in `pyproject.toml`. Re-run the validation loop at the start of every matrix run.

### R3. PyTorch 2.6 weights_only default

`torch.load(weights_only=True)` is the new default; many baseline checkpoints fail.

**Mitigation:** Monkey-patch in `compat_utils/torch_load_patch.py`. Pin `torch<2.7` anyway.

### R4. LDCQ chunk-filter threshold

`utils/utils.py:49` (`if np.all(norms <= 0.8)`) is AntMaze-tuned. HalfCheetah obs deltas during fast running can exceed 0.8 → chunks discarded.

**Per the no-modifications rule:** do not touch the threshold. Monitor via a print in stage-1 training; tag MLflow run `data_loss_warning=ldcq_chunk_filter_high` if > 50% chunks discarded. The resulting numbers reflect "LDCQ as written, applied to locomotion without env-specific code." That's the honest answer.

If you want better LDCQ locomotion numbers, the threshold change is documented as a "logic change" and goes in a separate, clearly-labeled experiment outside this matrix.

### R5. LDCQ 1000-step episode assumption

`utils/utils.py` `elif episode_step == 1000-horizon` assumes 1000-step episodes. Gymnasium-v5 envs use 1000 by default for HalfCheetah/Hopper/Walker2d but Humanoid's default may differ.

**Mitigation:** Print episode lengths during data prep:
```python
import minari
for ep in minari.load_dataset("mujoco/humanoid/medium-v0").iterate_episodes():
    print(len(ep.actions))
    break
```
If Humanoid episodes are not 1000 steps, LDCQ on Humanoid may have subtle wraparound issues. Tag with `compat_caveat=ldcq_humanoid_episode_length_unverified`.

### R6. Decision Diffuser config rigidity

DD's `default_inv.py` hard-codes the task. Per "no modifications," we run DD only on its single default config (`hopper-medium-expert-v2`). That's 1 task out of 11. Document the gap.

### R7. EDA / QGPO sklearn import-time crash

EDA's `dataset.py` imports `sklearn.datasets` at top of file. With numpy 2.x there are dtype size mismatches.

**Mitigation:** Pin `numpy<2.0` in pyproject.toml.

### R8. CORL wandb integration

CORL natively logs to W&B. Our `wandb_stub` forwards `wandb.log` → `mlflow.log_metric`. The forwarding may miss metrics if CORL uses keys with special characters.

**Mitigation:** The stub replaces `/` with `.` in metric names. Verify post-smoke that all CORL metrics show up in MLflow.

### R9. Compute budget overrun

Estimated 500-700 GPU-hours for the full matrix at 3 seeds × 10 eval episodes (more than original estimate because Humanoid runs ~3x longer than other envs).

Budget cuts in priority order if compute is tight:
1. Drop expert-v2 splits (4 task-datasets fewer) → saves ~15%
2. Drop seeds 3→1 → saves 65% but eliminates error bars
3. Drop Humanoid → saves ~25% but defeats user's stated requirement
4. Run only the 3 headliner methods + BC baseline → saves ~70%

**Recommendation:** Decide budget before starting; do not let the matrix grow unbounded.

### R10. Repo bit-rot

Some repos were last updated 2 years ago. Subtle issues may surface only during training (NaN losses, deprecated CUDA ops).

**Mitigation:** Smoke tests catch most. For any baseline that smoke-passes but training-diverges, tag with `reproduction_health=trained_but_diverged` and move on. The goal is reproduction breadth, not perfection on every single baseline.

### R11. CORL Humanoid config "extension" temptation

It is very tempting to add a Humanoid YAML to CORL (decision D4). Doing so puts a foot over the "no modifications" line.

**Mitigation:** Strict interpretation. Document the absence clearly in `results.md` so the reader knows it's a constraint of the experiment design, not a missing run.

## What to do if all goes well

After Phase 7:

1. Push `results.md` to the project.
2. Spot-check 3 outlier numbers (methods significantly above/below their paper number). Brief explanation paragraph per outlier.
3. OOB-1, OOB-2: bump seeds + eval episodes for top performers.
4. Compare your algorithm's row to the matrix.

## What to do if something fundamental breaks

- **Compat layer doesn't work end-to-end:** Gymnasium API version mismatch. Pin `gymnasium==1.0.0` exactly.
- **Minari downloads keep failing:** clear `~/.minari/`, retry. Farama server has had outages.
- **One baseline consistently diverges:** tag `reproduction_health=consistently_diverges`, exclude from table.
- **MLflow runs not appearing:** `make mlflow_logs` to check container; common cause is the DB not initialized — `make mlflow_down && make mlflow_up`.
- **Out of disk:** check `~/.minari/` (datasets can be ~10 GB total) and `repos/<repo>/logs/` (per-run checkpoints add up fast). Configure auto-cleanup of intermediate stage checkpoints once final-stage checkpoint is saved.
