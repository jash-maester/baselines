"""Maps D4RL-style env names to (Gymnasium-v5 env name, Minari dataset id).

`DATASET_OPTION` env var selects which Minari namespace is "active":
  - B: locally-converted D4RL hdf5 datasets under `localD4RL/...`
  - C: Minari-native `mujoco/...` datasets
"""
import os

# D4RL string -> Gymnasium env name (-v5 for MuJoCo 3 bindings)
D4RL_NAME_TO_GYM_ENV = {
    # HalfCheetah (Option B, paper-named)
    "halfcheetah-medium-v2":        "HalfCheetah-v5",
    "halfcheetah-medium-replay-v2": "HalfCheetah-v5",
    "halfcheetah-medium-expert-v2": "HalfCheetah-v5",
    "halfcheetah-expert-v2":        "HalfCheetah-v5",
    # HalfCheetah (Option C, Minari-native)
    "halfcheetah-medium-v0":        "HalfCheetah-v5",
    "halfcheetah-expert-v0":        "HalfCheetah-v5",
    "halfcheetah-simple-v0":        "HalfCheetah-v5",
    # Hopper
    "hopper-medium-v2":             "Hopper-v5",
    "hopper-medium-replay-v2":      "Hopper-v5",
    "hopper-medium-expert-v2":      "Hopper-v5",
    "hopper-expert-v2":             "Hopper-v5",
    "hopper-medium-v0":             "Hopper-v5",
    "hopper-expert-v0":             "Hopper-v5",
    "hopper-simple-v0":             "Hopper-v5",
    # Walker2d
    "walker2d-medium-v2":           "Walker2d-v5",
    "walker2d-medium-replay-v2":    "Walker2d-v5",
    "walker2d-medium-expert-v2":    "Walker2d-v5",
    "walker2d-expert-v2":           "Walker2d-v5",
    "walker2d-medium-v0":           "Walker2d-v5",
    "walker2d-expert-v0":           "Walker2d-v5",
    "walker2d-simple-v0":           "Walker2d-v5",
    # Humanoid (Option C only)
    "humanoid-medium-v0":           "Humanoid-v5",
    "humanoid-expert-v0":           "Humanoid-v5",
    "humanoid-simple-v0":           "Humanoid-v5",
}

# D4RL name -> Minari dataset ID, partitioned by dataset option
_OPTION_B = {
    "halfcheetah-medium-v2":        "localD4RL/halfcheetah/medium-v2",
    "halfcheetah-medium-replay-v2": "localD4RL/halfcheetah/medium-replay-v2",
    "halfcheetah-medium-expert-v2": "localD4RL/halfcheetah/medium-expert-v2",
    "halfcheetah-expert-v2":        "localD4RL/halfcheetah/expert-v2",
    "hopper-medium-v2":             "localD4RL/hopper/medium-v2",
    "hopper-medium-replay-v2":      "localD4RL/hopper/medium-replay-v2",
    "hopper-medium-expert-v2":      "localD4RL/hopper/medium-expert-v2",
    "hopper-expert-v2":             "localD4RL/hopper/expert-v2",
    "walker2d-medium-v2":           "localD4RL/walker2d/medium-v2",
    "walker2d-medium-replay-v2":    "localD4RL/walker2d/medium-replay-v2",
    "walker2d-medium-expert-v2":    "localD4RL/walker2d/medium-expert-v2",
    "walker2d-expert-v2":           "localD4RL/walker2d/expert-v2",
}

_OPTION_C = {
    "halfcheetah-medium-v0": "mujoco/halfcheetah/medium-v0",
    "halfcheetah-expert-v0": "mujoco/halfcheetah/expert-v0",
    "halfcheetah-simple-v0": "mujoco/halfcheetah/simple-v0",
    "hopper-medium-v0":      "mujoco/hopper/medium-v0",
    "hopper-expert-v0":      "mujoco/hopper/expert-v0",
    "hopper-simple-v0":      "mujoco/hopper/simple-v0",
    "walker2d-medium-v0":    "mujoco/walker2d/medium-v0",
    "walker2d-expert-v0":    "mujoco/walker2d/expert-v0",
    "walker2d-simple-v0":    "mujoco/walker2d/simple-v0",
    "humanoid-medium-v0":    "mujoco/humanoid/medium-v0",
    "humanoid-expert-v0":    "mujoco/humanoid/expert-v0",
    "humanoid-simple-v0":    "mujoco/humanoid/simple-v0",
}

_OPTION = os.environ.get("DATASET_OPTION", "C").upper()

# Union both maps so any name works at runtime (resolved by suffix).
D4RL_NAME_TO_MINARI_ID = {**_OPTION_B, **_OPTION_C}

# "Active" map for the current option — used by validation scripts.
ACTIVE_MAP = _OPTION_B if _OPTION == "B" else _OPTION_C


def get_minari_id(d4rl_name: str) -> str:
    if d4rl_name in D4RL_NAME_TO_MINARI_ID:
        return D4RL_NAME_TO_MINARI_ID[d4rl_name]
    raise KeyError(f"Unknown D4RL-style env name: {d4rl_name!r}")
