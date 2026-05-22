# 04 — Compat Layer Design

## Goal

Make `import gym` and `import d4rl` work in 11 disparate baseline repositories, against Gymnasium + Minari + Mujoco 3.x + PyTorch 2.4+, **without editing any repo's source files** beyond:
- import-line redirects (done transparently by shadowing the package name on `PYTHONPATH`)
- stubbing dead telemetry calls (comet_ml, wandb when used with empty creds)

All real work happens in a single installable package at the project root: `compat/`.

## Directory layout

```
~/offline_rl_repro/
├── compat/                              # uv pip install -e ./compat
│   ├── pyproject.toml                   # name = "compat-shim"
│   └── src/
│       ├── gym/                         # FAKE gym package, shadows pip-installed gym
│       │   └── __init__.py              # re-exports Gymnasium with old-API contract
│       ├── d4rl/                        # FAKE d4rl package
│       │   └── __init__.py              # qlearning_dataset(), shimmed
│       ├── comet_ml/                    # FAKE comet_ml package
│       │   └── __init__.py              # no-op Experiment class
│       └── compat_utils/                # OUR helpers, importable as compat_utils
│           ├── __init__.py
│           ├── minari_mapping.py
│           ├── env_factory.py
│           ├── d4rl_dict_converter.py
│           ├── mlflow_helper.py
│           ├── wandb_stub.py
│           └── torch_load_patch.py
├── repos/                               # cloned baselines, UNTOUCHED
├── scripts/                             # orchestration (smoke, run_one, build_results_table, ...)
├── docker/                              # MLflow compose (pre-existing — do not modify)
├── pyproject.toml                       # uv project root
├── pixi.toml                            # (optional) pixi alternative
└── Makefile
```

The trick: by installing a package literally named `gym`, the line `import gym` inside any repo resolves to OUR fake gym. Same for `d4rl` and `comet_ml`. **No source edits required.**

⚠️ **Side effect:** any genuine pip-installed `gym` package will be shadowed. Use a dedicated venv where the real `gym` is NOT installed. **CORL's `requirements.txt` includes both `gym` and `d4rl`** — strip both before installing CORL's deps. Same for other repos that list them.

## `compat/pyproject.toml`

```toml
[project]
name = "compat-shim"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "gymnasium[mujoco]>=1.0",
    "minari>=0.5",
    "mujoco>=3.1",
    "numpy<2.0",
    "torch>=2.4,<2.7",
    "mlflow>=2.9",
    "h5py",
]

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

## `compat/src/gym/__init__.py` — the fake `gym`

```python
"""
Fake gym shim. When a repo does `import gym`, it gets this.

Re-exports Gymnasium's API but with the *old* gym contract for the operations
baseline repos use:
  - env.reset() returns a bare obs (not (obs, info))
  - env.step() returns (obs, reward, done, info) 4-tuple (not 5-tuple)
  - env.seed(seed) exists
  - gym.make(d4rl_name) routes through env_factory to a Gymnasium env
"""
import gymnasium as _gym
from gymnasium import spaces  # re-export

from compat_utils.env_factory import make_d4rl_compatible_env, OldGymEnvWrapper
from compat_utils.torch_load_patch import apply as _apply_torch_patch
_apply_torch_patch()

# Public API surface used by the baselines
Env = _gym.Env
Wrapper = _gym.Wrapper
Space = _gym.Space
ObservationWrapper = _gym.ObservationWrapper
ActionWrapper = _gym.ActionWrapper
RewardWrapper = _gym.RewardWrapper

# Many repos use these submodules
from gymnasium import error, logger, utils

def make(name, **kwargs):
    return make_d4rl_compatible_env(name, **kwargs)

class GoalEnv(_gym.Env):
    pass

__version__ = "0.21.0-shim"
```

## `compat/src/d4rl/__init__.py` — the fake `d4rl`

```python
"""
Fake d4rl shim. Implements only what the baselines actually call.
"""
from compat_utils.d4rl_dict_converter import (
    qlearning_dataset,
    get_dataset,
    get_normalized_score,
)

# Some repos try `import d4rl.kitchen_envs` etc. — register no-op submodules.
import sys, types
for sub in ("locomotion", "kitchen_envs", "infos"):
    mod = types.ModuleType(f"d4rl.{sub}")
    sys.modules[f"d4rl.{sub}"] = mod

def _register():
    pass
_register()
```

## `compat/src/comet_ml/__init__.py` — the fake `comet_ml`

```python
"""
LDCQ hard-codes `Experiment(api_key='', project_name='')` which crashes the real
comet_ml on empty credentials. This stub makes it a no-op.
"""
class Experiment:
    def __init__(self, *args, **kwargs):
        pass
    def log_parameters(self, params): pass
    def log_metric(self, key, value, step=None): pass
    def log_model(self, *args, **kwargs): pass
    def add_tag(self, tag): pass
    def end(self): pass
    def __getattr__(self, name):
        return lambda *a, **kw: None
```

## `compat_utils/env_factory.py`

The hardest piece. Maps D4RL-style env names → Gymnasium envs wrapped to the old API + attached metadata so `env.get_dataset()` and `env.get_normalized_score()` work.

```python
import gymnasium as gym
import numpy as np
from compat_utils.minari_mapping import D4RL_NAME_TO_GYM_ENV, D4RL_NAME_TO_MINARI_ID
from compat_utils.d4rl_dict_converter import _episodes_to_d4rl_dict


class OldGymEnvWrapper(gym.Wrapper):
    """Wraps a Gymnasium env to expose the old gym contract used by baselines."""

    def __init__(self, env, d4rl_name):
        super().__init__(env)
        self._d4rl_name = d4rl_name

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        info["terminated"] = bool(terminated)
        info["truncated"] = bool(truncated)
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        return [seed]

    def get_dataset(self):
        """Mimics d4rl's env.get_dataset() — flat dict of full dataset."""
        import minari
        minari_id = D4RL_NAME_TO_MINARI_ID[self._d4rl_name]
        ds = minari.load_dataset(minari_id, download=True)
        return _episodes_to_d4rl_dict(ds, include_next_obs=False)

    def get_normalized_score(self, raw_return):
        """Maps raw episode return to a normalized score using Minari refs."""
        import minari
        minari_id = D4RL_NAME_TO_MINARI_ID[self._d4rl_name]
        ds = minari.load_dataset(minari_id, download=True)
        ref_min = getattr(ds.spec, "reference_min_score", None) or 0.0
        ref_max = getattr(ds.spec, "reference_max_score", None) or 1.0
        if ref_max == ref_min:
            return 0.0
        return (raw_return - ref_min) / (ref_max - ref_min)

    @property
    def spec_d4rl_name(self):
        return self._d4rl_name


def make_d4rl_compatible_env(d4rl_name, **kwargs):
    """'halfcheetah-medium-v2' -> Gymnasium HalfCheetah-v5 wrapped + tagged."""
    if d4rl_name in D4RL_NAME_TO_GYM_ENV:
        gym_name = D4RL_NAME_TO_GYM_ENV[d4rl_name]
        env = gym.make(gym_name, **kwargs)
    else:
        # Direct Gymnasium env name (e.g. 'HalfCheetah-v5')
        env = gym.make(d4rl_name, **kwargs)
    return OldGymEnvWrapper(env, d4rl_name)
```

## `compat_utils/minari_mapping.py`

```python
import os

# D4RL string -> Gymnasium env name (-v5 for MuJoCo 3 bindings)
D4RL_NAME_TO_GYM_ENV = {
    # HalfCheetah (Option B, paper-named)
    'halfcheetah-medium-v2':        'HalfCheetah-v5',
    'halfcheetah-medium-replay-v2': 'HalfCheetah-v5',
    'halfcheetah-medium-expert-v2': 'HalfCheetah-v5',
    'halfcheetah-expert-v2':        'HalfCheetah-v5',
    # HalfCheetah (Option C, Minari-native)
    'halfcheetah-medium-v0':        'HalfCheetah-v5',
    'halfcheetah-expert-v0':        'HalfCheetah-v5',
    # Hopper
    'hopper-medium-v2':             'Hopper-v5',
    'hopper-medium-replay-v2':      'Hopper-v5',
    'hopper-medium-expert-v2':      'Hopper-v5',
    'hopper-expert-v2':             'Hopper-v5',
    'hopper-medium-v0':             'Hopper-v5',
    'hopper-expert-v0':             'Hopper-v5',
    # Walker2d
    'walker2d-medium-v2':           'Walker2d-v5',
    'walker2d-medium-replay-v2':    'Walker2d-v5',
    'walker2d-medium-expert-v2':    'Walker2d-v5',
    'walker2d-expert-v2':           'Walker2d-v5',
    'walker2d-medium-v0':           'Walker2d-v5',
    'walker2d-expert-v0':           'Walker2d-v5',
    # Humanoid (Option C only)
    'humanoid-medium-v0':           'Humanoid-v5',
    'humanoid-expert-v0':           'Humanoid-v5',
    'humanoid-simple-v0':           'Humanoid-v5',
}

# D4RL name -> Minari dataset ID, partitioned by dataset option
_OPTION_B = {
    'halfcheetah-medium-v2':        'localD4RL/halfcheetah/medium-v2',
    'halfcheetah-medium-replay-v2': 'localD4RL/halfcheetah/medium-replay-v2',
    'halfcheetah-medium-expert-v2': 'localD4RL/halfcheetah/medium-expert-v2',
    'halfcheetah-expert-v2':        'localD4RL/halfcheetah/expert-v2',
    'hopper-medium-v2':             'localD4RL/hopper/medium-v2',
    'hopper-medium-replay-v2':      'localD4RL/hopper/medium-replay-v2',
    'hopper-medium-expert-v2':      'localD4RL/hopper/medium-expert-v2',
    'hopper-expert-v2':             'localD4RL/hopper/expert-v2',
    'walker2d-medium-v2':           'localD4RL/walker2d/medium-v2',
    'walker2d-medium-replay-v2':    'localD4RL/walker2d/medium-replay-v2',
    'walker2d-medium-expert-v2':    'localD4RL/walker2d/medium-expert-v2',
    'walker2d-expert-v2':           'localD4RL/walker2d/expert-v2',
}

_OPTION_C = {
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

_OPTION = os.environ.get("DATASET_OPTION", "B").upper()

# Union both maps so any name works at runtime
D4RL_NAME_TO_MINARI_ID = {**_OPTION_B, **_OPTION_C}
# Plus an explicit "active" map for the current option (used by validation scripts)
ACTIVE_MAP = _OPTION_B if _OPTION == "B" else _OPTION_C
```

## `compat_utils/d4rl_dict_converter.py`

```python
import numpy as np
import minari
from compat_utils.minari_mapping import D4RL_NAME_TO_MINARI_ID


def _episodes_to_d4rl_dict(minari_dataset, include_next_obs=True):
    """Flatten Minari episodes into D4RL's flat-array dict."""
    obs_list, act_list, rew_list, nxt_list = [], [], [], []
    term_list, timo_list = [], []

    for ep in minari_dataset.iterate_episodes():
        n = len(ep.actions)
        obs_list.append(ep.observations[:n])
        if include_next_obs:
            nxt_list.append(ep.observations[1:n+1])
        act_list.append(ep.actions)
        rew_list.append(ep.rewards)
        t = np.zeros(n, dtype=bool)
        to = np.zeros(n, dtype=bool)
        # Minari stores termination/truncation per step; only last step is set.
        if hasattr(ep, "terminations") and len(ep.terminations) >= 1:
            t[-1] = bool(ep.terminations[-1])
        if hasattr(ep, "truncations") and len(ep.truncations) >= 1:
            to[-1] = bool(ep.truncations[-1])
        term_list.append(t)
        timo_list.append(to)

    out = dict(
        observations=np.concatenate(obs_list).astype(np.float32),
        actions=np.concatenate(act_list).astype(np.float32),
        rewards=np.concatenate(rew_list).astype(np.float32),
        terminals=np.concatenate(term_list),
        timeouts=np.concatenate(timo_list),
    )
    if include_next_obs:
        out['next_observations'] = np.concatenate(nxt_list).astype(np.float32)
    return out


def qlearning_dataset(env, **kwargs):
    """Replacement for d4rl.qlearning_dataset(env) — called by EDA, QGPO, CORL."""
    d4rl_name = env.spec_d4rl_name
    minari_id = D4RL_NAME_TO_MINARI_ID[d4rl_name]
    ds = minari.load_dataset(minari_id, download=True)
    return _episodes_to_d4rl_dict(ds, include_next_obs=True)


def get_dataset(env, **kwargs):
    """Replacement for env.get_dataset() — used by LDCQ."""
    d4rl_name = env.spec_d4rl_name if hasattr(env, "spec_d4rl_name") else env._d4rl_name
    minari_id = D4RL_NAME_TO_MINARI_ID[d4rl_name]
    ds = minari.load_dataset(minari_id, download=True)
    return _episodes_to_d4rl_dict(ds, include_next_obs=False)


def get_normalized_score(d4rl_name, raw_return):
    """Module-level normalized scoring (some repos call d4rl.get_normalized_score)."""
    minari_id = D4RL_NAME_TO_MINARI_ID[d4rl_name]
    ds = minari.load_dataset(minari_id, download=True)
    ref_min = getattr(ds.spec, "reference_min_score", 0.0) or 0.0
    ref_max = getattr(ds.spec, "reference_max_score", 1.0) or 1.0
    if ref_max == ref_min:
        return 0.0
    return (raw_return - ref_min) / (ref_max - ref_min)
```

## `compat_utils/torch_load_patch.py`

```python
"""PyTorch >= 2.6 made weights_only=True the default; many baseline checkpoints
serialize Python objects and fail to load. Monkey-patch to restore old behavior."""
import torch as _torch

_original_load = _torch.load
_PATCHED = False

def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

def apply():
    global _PATCHED
    if _PATCHED:
        return
    _torch.load = _patched_load
    _PATCHED = True
```

## `compat_utils/wandb_stub.py`

Imported before any repo that uses `wandb`. Replaces it with a no-op that *also* forwards `wandb.log()` calls to MLflow.

```python
"""Replace wandb with a no-op that forwards logs to MLflow."""
import sys
import types
import mlflow

_run_active = False

def init(*args, **kwargs):
    global _run_active
    _run_active = True
    return _StubRun()

class _StubRun:
    def __getattr__(self, name):
        return lambda *a, **kw: None
    def __enter__(self): return self
    def __exit__(self, *args): pass

def log(metrics, step=None, **kwargs):
    """Forward to MLflow if a run is active."""
    if not _run_active:
        return
    for k, v in metrics.items():
        try:
            mlflow.log_metric(k.replace("/", "."), float(v), step=step)
        except Exception:
            pass

def finish(*args, **kwargs):
    global _run_active
    _run_active = False

def login(*args, **kwargs):
    return True

# Install as `wandb` in sys.modules so `import wandb` resolves here.
def install():
    mod = types.ModuleType("wandb")
    mod.init = init
    mod.log = log
    mod.finish = finish
    mod.login = login
    mod.run = None
    sys.modules["wandb"] = mod
```

## `compat_utils/mlflow_helper.py`

```python
import os
import mlflow

DEFAULT_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555")
EXPERIMENT_NAME = "latent_cep_baselines"  # was "baselines"; renamed to avoid host-MLflow collision

def init():
    mlflow.set_tracking_uri(DEFAULT_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def run_name(algo, env, dataset, seed, stage):
    return f"{algo}_{env}-{dataset}_seed{seed}_{stage}"

def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out

def start_run(algo, env, dataset, seed, stage, config, extra_tags=None):
    init()
    name = run_name(algo, env, dataset, seed, stage)
    run = mlflow.start_run(run_name=name)
    tags = {
        "algo": algo, "env": env, "dataset": dataset,
        "seed": str(seed), "stage": stage,
        "minari_id": config.get("minari_id", ""),
        "repo_url": config.get("repo_url", ""),
        "method_family": config.get("method_family", ""),
        "dataset_option": os.environ.get("DATASET_OPTION", "B"),
    }
    if extra_tags:
        tags.update(extra_tags)
    mlflow.set_tags(tags)
    mlflow.log_params(_flatten(config))
    return run
```

## Installation flow

### Using uv (recommended; available on this server)

```bash
cd ~/offline_rl_repro

# 1. Create venv
uv venv --python 3.11 .venv
source .venv/bin/activate

# 2. Install compat shim FIRST (so its fake gym, d4rl, comet_ml register)
uv pip install -e ./compat

# 3. Install project dependencies (NO d4rl, NO mujoco_py, NO gym in this list)
uv pip install \
    "torch>=2.4,<2.7" \
    "gymnasium[mujoco]>=1.0" \
    "mujoco>=3.1" \
    "minari[hdf5]>=0.5" \
    "numpy<2.0" \
    "scipy" "tqdm" "mlflow>=2.9" "h5py" \
    "matplotlib" "scikit-learn" "tensorboard" \
    "pyrallis"   # CORL uses this

# 4. Clone repos (read-only after this)
mkdir -p repos && cd repos
git clone https://github.com/ldcq/ldcq.git
git clone https://github.com/thu-ml/Efficient-Diffusion-Alignment.git
git clone https://github.com/thu-ml/CEP-energy-guided-diffusion.git
git clone https://github.com/jannerm/diffuser.git
git clone https://github.com/anuragajay/decision-diffuser.git
git clone https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL.git
git clone https://github.com/corl-team/CORL.git
cd ..

# 5. Validate compat shims
python -c "import gym; e=gym.make('humanoid-medium-v0'); print(e); print(e.get_dataset()['observations'].shape)"
python -c "import d4rl, gym; print(d4rl.qlearning_dataset(gym.make('halfcheetah-medium-v0')).keys())"
```

### Using pixi (alternative, if installed)

```bash
cd ~/offline_rl_repro

# 1. Initialize pixi project (creates pixi.toml + pixi.lock)
pixi init --pyproject

# 2. Add deps from conda-forge
pixi add "python=3.11" "pytorch>=2.4,<2.7" "numpy<2.0" "scipy" "tqdm" \
         "mlflow>=2.9" "h5py" "matplotlib" "scikit-learn" "tensorboard"

# 3. Add pip-only deps inside the pixi env
pixi run pip install "gymnasium[mujoco]>=1.0" "mujoco>=3.1" "minari[hdf5]>=0.5" "pyrallis"

# 4. Install compat shim
pixi run pip install -e ./compat

# 5. Same git clones as above
# 6. Same validation calls, prefixed with `pixi run`
pixi run python -c "import gym; e=gym.make('humanoid-medium-v0'); print(e)"
```

Both produce the same project shape. Pick one, commit the lockfile (`uv.lock` or `pixi.lock`), do not mix.

## Verifying compat per repo

```bash
# CORL data path
cd repos/CORL
python -c "import d4rl, gym; env=gym.make('halfcheetah-medium-v0'); d=d4rl.qlearning_dataset(env); print('CORL data OK:', d['observations'].shape)"

# EDA data path
cd ../Efficient-Diffusion-Alignment
python -c "
import argparse
from dataset import D4RL_dataset
a = argparse.Namespace(env='halfcheetah-medium-v0', debug=True, device='cpu')
D4RL_dataset(a)
print('EDA: OK')
"

# QGPO data path
cd ../CEP-energy-guided-diffusion/Offline_RL_2D
# (similar smoke; depends on how QGPO loads data — see docs/05)
```

These exercise data loading without training anything. If any fails, fix compat before touching training scripts.
