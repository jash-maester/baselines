# 01 — Baselines Catalog

Every baseline cited by LDCQ / EDA / QGPO for locomotion comparison, with verified repo URLs, language stack, and how it integrates with the matrix.

## Citation matrix

Which paper cites which baseline in their **locomotion result table** (✓ = appears with reported numbers):

| Baseline | LDCQ | EDA | QGPO |
|---|:---:|:---:|:---:|
| BC | ✓ | ✓ | – |
| BCQ | ✓ | ✓ | ✓ |
| CQL | ✓ | ✓ | ✓ |
| IQL | ✓ | ✓ | ✓ |
| TD3+BC | – | ✓ | – |
| DT | ✓ | ✓ | ✓ |
| Diffuser | ✓ | – | ✓ |
| Decision Diffuser | ✓ | – | ✓ |
| Diffusion-QL | – | ✓ | ✓ |
| SfBC | – | – | ✓ |
| IDQL | – | ✓ | – |
| QGPO | – | ✓ | (it is QGPO) |
| LDCQ | (it is LDCQ) | – | – |
| EDA | – | (it is EDA) | – |

## Method-by-method

### Headliner 1 — LDCQ
- **Paper:** Venkatraman et al. 2023, "Reasoning with Latent Diffusion in Offline Reinforcement Learning" (arXiv 2309.06599)
- **Repo:** https://github.com/ldcq/ldcq
- **Language:** PyTorch
- **Default config:** Hyperparameters in `training/*.py` argparse defaults. Locomotion `horizon=10`. **Do not override.**
- **Stages:** 3 sequential — `train_skills.py` → `train_diffusion.py` → `train_q_net.py`
- **Eval:** Repo ships eval **only for AntMaze/Kitchen/Maze2D**. We must write a locomotion eval script (see `05_execution_per_method.md`). That script calls LDCQ classes as a library, not modifying any LDCQ source.
- **Quirks:** `comet_ml` is imported and instantiated with empty creds in 3 scripts — fake `comet_ml` shim handles this. `utils/utils.py` line 49 has a chunk-filtering threshold (`np.all(norms <= 0.8)`) that was tuned for AntMaze; for locomotion it may discard a high fraction of trajectory chunks. **Per the no-modifications rule, do NOT change the threshold.** Tag the resulting runs with `data_loss_warning=ldcq_chunk_filter` if chunk loss exceeds 50%, and document in the final report.

### Headliner 2 — EDA
- **Paper:** Chen et al. 2024 NeurIPS, "Aligning Diffusion Behaviors with Q-functions for Efficient Continuous Control" (arXiv 2407.09024)
- **Repo:** https://github.com/thu-ml/Efficient-Diffusion-Alignment
- **Language:** PyTorch
- **Default config:** Hyperparameters in `argparse` defaults of each script. **Per-task β values** are part of the documented invocation (README example shows `--beta=0.1`); the **README's command itself uses default β=0.1**. We follow the README *exactly* — i.e. always pass `--beta=0.1` unless the paper's Table 2 specifies otherwise *and* the repo would crash without it. Default values stay default.
- **Stages:** 3 sequential — `train_behavior.py` → `train_critic.py` → `finetune_policy.py`
- **Quirks:** Clean repo. Only standard `import gym` / `import d4rl` shimming needed.

### Headliner 3 — QGPO (CEP)
- **Paper:** Lu et al. 2023 ICML, "Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning" (arXiv 2304.12824)
- **Repo:** https://github.com/thu-ml/CEP-energy-guided-diffusion
- **Code path:** `Offline_RL_2D/` subdirectory
- **Language:** PyTorch
- **Default config:** Hyperparameters in argparse defaults. README shows the exact command — use it verbatim with `--diffusion_steps 15 --alpha 3 --q_alpha 1 --method "CEP"`. These are not "tuning"; they're the README defaults.
- **Stages:** 2 sequential — `train_behavior.py` → `train_critic.py` (latter evaluates inline)

### Baseline 1 — BC (Behavior Cloning)
- **Source:** CORL — `algorithms/offline/any_percent_bc.py`
- **Repo:** https://github.com/corl-team/CORL
- **Default config:** `configs/offline/any_percent_bc/halfcheetah/medium_v2.yaml` (and analogous for hopper, walker2d). **No humanoid config ships.** See dataset note below.
- **Language:** PyTorch

### Baseline 2 — BCQ
- **Status:** **CORL does not ship BCQ.** Original repo is Fujimoto's: https://github.com/sfujim/BCQ
- **Decision:** BCQ is widely cited but the original repo is old (gym 0.21-era). Two choices:
  - (a) Run sfujim/BCQ through compat shim. Slightly more shim work because that repo uses some old gym APIs that CORL avoids.
  - (b) Skip BCQ. The three headliner papers' BCQ numbers are weak baselines anyway — never the SOTA.
- **Recommendation:** Skip in first pass. Add as OOB later if needed.

### Baseline 3 — CQL
- **Source:** CORL — `algorithms/offline/cql.py`
- **Default config:** `configs/offline/cql/halfcheetah/medium_v2.yaml`, etc.
- **Humanoid:** No config in CORL.

### Baseline 4 — IQL
- **Source:** CORL — `algorithms/offline/iql.py`
- **Default config:** `configs/offline/iql/halfcheetah/medium_v2.yaml`, etc.
- **Humanoid:** No config in CORL.

### Baseline 5 — TD3+BC
- **Source:** CORL — `algorithms/offline/td3_bc.py`
- **Default config:** `configs/offline/td3_bc/halfcheetah/medium_v2.yaml`, etc.
- **Humanoid:** No config in CORL.

### Baseline 6 — DT (Decision Transformer)
- **Source:** CORL — `algorithms/offline/dt.py`
- **Default config:** `configs/offline/dt/halfcheetah/medium_v2.yaml`, etc.
- **Humanoid:** No config in CORL.

### Baseline 7 — Diffuser
- **Repo:** https://github.com/jannerm/diffuser
- **Language:** PyTorch
- **Default config:** `config/locomotion.py` (Python file, not YAML). Loaded by `--config config.locomotion`.
- **Stages:** 3 — `train.py` (diffusion model) → `train_values.py` (value function) → `plan_guided.py` (eval).
- **Humanoid:** Not in default config. Skip Humanoid for Diffuser unless you add a config block (which is a modification — so don't).

### Baseline 8 — Decision Diffuser
- **Repo:** https://github.com/anuragajay/decision-diffuser
- **Code path:** `code/`
- **Default config:** `code/analysis/default_inv.py`
- **Humanoid:** Not in default config.

### Baseline 9 — Diffusion-QL
- **Repo:** https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL
- **Language:** PyTorch
- **Default config:** Hyperparameters hard-coded in `main.py` per task. The README explicitly says they're tuned. **Run as-is.**
- **Humanoid:** Not in default. Skip.

### Baseline 10 — SfBC
- **Repo:** https://github.com/ChenDRAG/SfBC
- **Status:** SfBC authors recommend QGPO instead — same authors, strict improvement. **Skip SfBC** unless your final write-up specifically requires its row. QGPO numbers strictly dominate.

### Baseline 11 — IDQL
- **Repo:** https://github.com/philippe-eecs/IDQL
- **Language:** **JAX**
- **Decision:** Cite reported numbers, do not reproduce. JAX setup is its own project. Tag the row in `results.md` as "from paper, not reproduced".

## CORL — the classical-baselines hub

**Repo:** https://github.com/corl-team/CORL

Provides clean single-file PyTorch implementations of: AWAC, BC, CQL, DT, EDAC, IQL, SAC-N, TD3+BC, LB-SAC, SPOT, Cal-QL, ReBRAC.

For us: **BC, CQL, IQL, TD3+BC, DT** — five baselines, one repo, one consistent installation.

CORL ships YAML configs under `configs/offline/<algo>/<env>/<split>.yaml`. **Use these verbatim.** Do not edit.

**CORL natively logs to W&B.** Our compat layer stubs `wandb` to a no-op so CORL doesn't crash, and we monkey-patch `wandb.log` to also call `mlflow.log_metric` (see `04_compat_layer.md`). This is a runtime monkey-patch in our launcher, not a source edit to CORL.

**Humanoid is NOT in CORL's locomotion config tree.** This is the critical gap discussed in `02_dataset_situation.md`.

## Quick repo URLs reference list

```
# Headliners
https://github.com/ldcq/ldcq
https://github.com/thu-ml/Efficient-Diffusion-Alignment
https://github.com/thu-ml/CEP-energy-guided-diffusion

# Trajectory / policy diffusion baselines
https://github.com/jannerm/diffuser
https://github.com/anuragajay/decision-diffuser
https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL

# Classical baselines library (BC, CQL, IQL, TD3+BC, DT)
https://github.com/corl-team/CORL

# Skipped (reasons documented above)
# https://github.com/sfujim/BCQ            # weak baseline, old repo
# https://github.com/ChenDRAG/SfBC         # dominated by QGPO
# https://github.com/philippe-eecs/IDQL    # JAX; cite reported numbers
```

## Final shortlist (active, will-be-run baselines)

Eight active methods + your method = **9 rows** in the comparison table:

1. **BC** (CORL)
2. **CQL** (CORL)
3. **IQL** (CORL)
4. **TD3+BC** (CORL)
5. **DT** (CORL)
6. **Diffuser** (jannerm)
7. **Decision Diffuser** (anuragajay)
8. **Diffusion-QL** (Zhendong-Wang)
9. **LDCQ** (ldcq)
10. **EDA** (thu-ml/Efficient-Diffusion-Alignment)
11. **QGPO** (thu-ml/CEP)
12. **Your method**

Plus three reference-only rows (numbers cited from papers): IDQL, SfBC, BCQ — these go in `results.md` with a "from paper" footnote.

## Humanoid availability summary

| Method | Has Humanoid default config? | Strategy |
|---|---|---|
| BC, CQL, IQL, TD3+BC, DT (CORL) | ❌ | See `02_dataset_situation.md` — copy halfcheetah config and change env name, document as "configuration extended from halfcheetah defaults" |
| Diffuser | ❌ | Skip Humanoid for Diffuser |
| Decision Diffuser | ❌ | Skip Humanoid |
| Diffusion-QL | ❌ | Skip Humanoid |
| LDCQ | ❌ (locomotion uses argparse defaults, env-agnostic) | Run; `--env_name humanoid-medium-v0` (custom string) → shim maps to `mujoco/humanoid/medium-v0` |
| EDA | ❌ | Same as LDCQ — env-name passed via argparse, no per-env config edit needed |
| QGPO | ❌ | Same |

**The cleanest path:** Run Humanoid for the 4 methods that take `--env` as a CLI flag (LDCQ, EDA, QGPO, your method). Skip Humanoid for the 7 methods that ship per-env YAML configs without a Humanoid entry. Document this gap in `results.md`.

For CORL, the alternative is to copy `configs/offline/cql/halfcheetah/medium_v2.yaml` to `configs/offline/cql/humanoid/medium_v0.yaml` and edit only the dataset name. Since the user said "use exact repo defaults," **this is borderline.** The strict interpretation is "don't add a new config file." Discuss this in `08_risks_and_decisions.md`.
