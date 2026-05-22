# Baselines reproduction — new-machine onboarding

This is the `baselines` meta-repo: a Minari-based reproduction of 7 offline-RL
baseline codebases (CORL, Diffuser, Decision-Diffuser, Diffusion-Policies/DQL,
Efficient-Diffusion-Alignment, CEP/QGPO, LDCQ) wired as git submodules under
`repos/`. Training runs log to a shared MLflow. The goal is to compare
**our Minari-normalized scores** against **paper-reported D4RL scores**.

You're being set up on a second machine (same hardware class: RTX 5090 /
Blackwell sm_120) to run the long matrix. The first machine already debugged
all the compat fixes; this machine just needs to clone, install, and launch.

---

## Prompt for Claude Code (paste this on the new machine)

> I've cloned the `baselines` meta-repo. Get it ready to run the reproduction
> matrix on this machine. Specifically:
>
> 1. Run `bash scripts/bootstrap.sh` — it inits the 7 submodules, builds the
>    `.venv` (Python 3.11), installs torch 2.11.0+cu128 + the compat shim +
>    `requirements.lock.txt`, downloads the 8 Minari datasets, and runs an
>    import sanity check. If a step fails, diagnose and fix before continuing.
>
> 2. **Point this machine at the shared MLflow.** Edit the `Makefile` line
>    `MLFLOW_URI   ?= http://localhost:5555` to
>    `MLFLOW_URI   ?= http://10.21.186.205:5555` so every `make` target logs to
>    the shared container over LAN (unified logs with the other machine). This
>    is a LOCAL edit — do NOT commit it. Confirm reachability:
>    `curl -sf http://10.21.186.205:5555/health && echo OK`.
>
> 3. Smoke-test every method first: `make smoke` (each method < 5 min; halts on
>    first crash). Only if all smokes log an MLflow run should you proceed.
>
> 4. Launch the long matrix: `make matrix`. Then `make dashboard` for the live
>    view, including the Minari-vs-D4RL compare table.
>
> Notes / guardrails:
> - The MLflow at 10.21.186.205:5555 is a SHARED service — never restart or
>   bring it down. Only set the tracking URI.
> - Repo edits inside `repos/*` are allowed only if they preserve the
>   mathematical algorithm (helpers/utils/import-redirects/telemetry-silencing);
>   log every such edit. No hyperparameter changes, no methodology changes.
> - 4 envs only: HalfCheetah, Hopper, Walker2d, Humanoid; datasets medium +
>   expert. LDCQ + Decision-Diffuser are present as submodules but excluded
>   from the default matrix.
> - Discord failure notifications are wired (`scripts/notify.py`); matrix pings
>   on failure only.

---

## What the matrix runs

246 cells = {bc, cql, iql, td3bc, dt, diffuser, dql, eda, qgpo} × 4 envs ×
{medium, expert} × 3 seeds (EDA/QGPO have extra per-stage cells). Honest
wall-clock estimate at concurrency-12 / bin-packed: **~10–15 days**. The
orchestrator (`scripts/long_runner.py`) bin-packs by measured vRAM/RAM
footprint with easy-first ordering and OOM-safety floors; `make matrix` runs it
under `nohup` and writes `state/long_runner.pid`.

## Multi-machine work split (optional)

To split the matrix across both machines, use pools:
```
make matrix-reassign PATTERN=qgpo POOL=server2     # tag cells
make matrix-export   POOL=server2 OUT=state/matrix-server2.json
# scp the manifest to this machine, then:
python scripts/long_runner.py --execute --pool server2
```
Both machines logging to the same MLflow keeps results unified.

## Key files

- `Makefile` — operator shortcuts (`make help`)
- `scripts/bootstrap.sh` — this machine's setup
- `scripts/long_runner.py` — matrix orchestrator
- `scripts/progress_dashboard.py` — status + Minari-vs-D4RL compare table
- `reference_scores.json` — paper-reported D4RL numbers + per-algo units
- `compat/` — the gym/d4rl/wandb shim package (editable install)
- `RUN_LOG.md` — full patch/audit trail
- `logs/incidents.md` — matrix-run failures, root cause + fix per row
