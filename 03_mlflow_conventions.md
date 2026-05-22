# 03 — MLflow Conventions

## Server

A locally-hosted MLflow Docker container at **`http://localhost:5555`**. The `docker/` directory in the project root contains `docker-compose.yml`; do not modify it.

Make targets (already present in the project):
- `make mlflow_up` — start the container (idempotent)
- `make mlflow_down` — stop
- `make mlflow_logs` — tail logs
- `make mlflow_status` — check whether running
- `make bootstrap` — `install` + `mlflow_up` combined

Run `make mlflow_status` before any training session. Training scripts assume the server is reachable and do not start it.

## Experiment naming

A **single MLflow experiment** named `latent_cep_baselines` for the long training matrix, plus `latent_cep_baselines_smoke` for autoresearch/smoke iteration. Both live on the same host MLflow at `http://localhost:5555`. The plan originally said `baselines`; renamed 2026-05-19 to avoid collision with 15 pre-existing trading-backtest runs in the `baselines` experiment on this host.

```python
mlflow.set_experiment("latent_cep_baselines")  # long matrix
# autoresearch loop sets MLFLOW_EXPERIMENT_NAME=latent_cep_baselines_smoke
```

Use **MLflow tags** for filtering. Tags scale; nested experiments do not.

## Run naming

```
{algo}_{env}-{dataset_variant}_seed{N}_{stage}
```

Where:
- `algo` ∈ `{bc, cql, iql, td3bc, dt, diffuser, dd, dql, qgpo, ldcq, eda, mymethod}` — lowercase, no spaces, no hyphens
- `env` ∈ `{halfcheetah, hopper, walker2d, humanoid}` — lowercase, no spaces
- `dataset_variant` ∈ `{medium, medium-replay, medium-expert, expert, simple}` — hyphenated within the variant
- `N` ∈ `{0, 1, 2}` — minimum 3 seeds per method/task
- `stage` ∈ `{pretrain, diffusion, critic, finetune, eval, full}` — multi-stage methods split into separate runs; single-stage methods use `full`

**Examples:**
```
cql_halfcheetah-medium_seed0_full
dt_hopper-medium-expert_seed2_full
ldcq_humanoid-medium_seed0_pretrain
ldcq_humanoid-medium_seed0_diffusion
ldcq_humanoid-medium_seed0_critic
ldcq_humanoid-medium_seed0_eval
eda_walker2d-medium-replay_seed1_pretrain
eda_walker2d-medium-replay_seed1_critic
eda_walker2d-medium-replay_seed1_finetune
qgpo_halfcheetah-medium-expert_seed0_pretrain
qgpo_halfcheetah-medium-expert_seed0_critic       # critic stage includes inline eval
mymethod_humanoid-expert_seed2_full
```

## Mandatory tags on every run

```python
mlflow.set_tags({
    "algo":             "ldcq",                                       # method name
    "env":              "humanoid",                                   # bare env
    "dataset":          "medium",                                     # variant
    "minari_id":        "mujoco/humanoid/medium-v0",                  # exact ID loaded
    "seed":             "0",
    "stage":            "diffusion",                                  # one of the values above
    "method_family":    "latent_diffusion",                           # see classification below
    "repo_url":         "https://github.com/ldcq/ldcq",
    "repo_sha":         "<git rev-parse HEAD inside repos/ldcq>",     # exact upstream commit
    "compat_sha":       "<git rev-parse HEAD of our project>",
    "device":           "cuda:0",
    "python":           "3.11",
    "torch":            "2.4.1",
    "dataset_option":   "C",                                          # B or C — see docs/02
    "eval_mode":        "stoch_argmax_q",                             # see docs/06
})
```

`method_family` classification:

| family | members |
|---|---|
| `behavior_cloning` | bc |
| `q_learning_offline` | cql, iql, td3bc |
| `sequence_modeling` | dt |
| `trajectory_diffusion` | diffuser, dd |
| `diffusion_policy` | dql, qgpo |
| `latent_diffusion` | ldcq |
| `diffusion_alignment` | eda |
| `mymethod` | (yours) |

## Parameters to log

For every run, log the full config dict (whatever the repo's argparse/yaml produced), flattened with dotted keys:

```python
mlflow.log_params({
    "env_name": "humanoid-medium-v0",
    "horizon": 10,                 # LDCQ
    "lr": 5e-5,
    "batch_size": 128,
    "diffusion.T": 200,
    "diffusion.beta_schedule": "linear",
    "vae.beta": 0.05,
    "vae.z_dim": 16,
    # ... whatever the upstream config dict contains
})
```

Important: **do not synthesize parameters** that aren't in the repo's own config. The point of "no tuning" is that the recorded params should be the repo's defaults verbatim.

## Metrics to log

### During training (every N gradient steps):
- `train.loss` — total loss
- Method-specific component losses (`vae.kl`, `vae.recon`, `diffusion.mse`, `q.bellman`, `policy.ce`)

### At eval boundaries (every K epochs or end-of-stage):
- `eval.return_mean` — raw cumulative return averaged over N eval episodes
- `eval.return_std`
- `eval.normalized_score_mean` — Minari `get_normalized_scores` output × 100 (or D4RL refs for Option B)
- `eval.normalized_score_std`
- `eval.episode_count` — N
- `eval.wall_time_sec`

### Final (logged once per run, at the end):
- `final.normalized_score_mean`
- `final.normalized_score_std`
- `final.raw_return_mean`
- `final.raw_return_std`

For Humanoid runs, `final.normalized_score_mean` is computed against Minari's reference scores (no paper baseline exists). For HalfCheetah/Hopper/Walker2d Option-B runs, against D4RL reference scores embedded in our local Minari dataset metadata.

## Artifacts to log

Per run:
- Full config JSON (whatever the repo produced, captured before training starts)
- The git diff of any edits to the cloned repo (should be empty — only compat shims, which live outside)
- Final checkpoint(s) only if total size < 200 MB; otherwise log local path under tag `checkpoint_path`
- Per-episode return CSV from the eval phase

## Aggregation query at the end

```python
import mlflow
client = mlflow.tracking.MlflowClient(tracking_uri="http://localhost:5555")
exp = client.get_experiment_by_name("latent_cep_baselines")
runs = client.search_runs(
    [exp.experiment_id],
    filter_string="tags.stage IN ('eval','full','critic','finetune') AND metrics.`final.normalized_score_mean` >= 0",
    max_results=10000,
)
# Build DataFrame: rows = algo, columns = env-dataset, cells = mean±std across seeds
```

`scripts/build_results_table.py` produces `results.md` (Markdown table) and `results.csv` from this query. See `06_evaluation_protocol.md`.

## Tag-based filtering examples (for the UI)

In the MLflow UI search box:

```
tags.env = "humanoid" AND tags.dataset = "medium"
tags.algo = "ldcq" AND tags.stage = "diffusion"
tags.method_family = "diffusion_policy"
tags.dataset_option = "B"
```

These should return clean per-method comparison plots when you click "Compare".

## Why a single experiment

Each method ends up with 1–3 stages × ~11 task-datasets × 3 seeds = 33–99 MLflow runs. Across ~11 methods, that's ~400–800 runs. Putting them all in one experiment makes cross-method comparison queries trivial; nested experiments would require manual joins.

If a single experiment becomes hard to navigate, MLflow's UI lets you save filter presets. Don't create more experiments.

## Sanity check on the dashboard

Open `http://localhost:5555`. You should see:
- Two experiments: `latent_cep_baselines` (matrix) and `latent_cep_baselines_smoke` (smoke loop).
- A run list with the naming scheme above.
- Tag filters working (try `tags.env = "humanoid"`).
- The "Compare" feature lets you select runs by tag and view metric overlays — this is the primary visualization tool.

If the dashboard is blank after a few runs, check `make mlflow_logs`. Most common cause: training scripts not having `MLFLOW_TRACKING_URI=http://localhost:5555` set in their env.
