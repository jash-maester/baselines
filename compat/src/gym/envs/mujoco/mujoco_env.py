"""Shim for `gym.envs.mujoco.mujoco_env` exposing MujocoEnv.

Old gym-0.21 baselines do `from gym.envs.mujoco import mujoco_env` and subclass
`mujoco_env.MujocoEnv`. Gymnasium has a compatible MujocoEnv at
`gymnasium.envs.mujoco.mujoco_env`; we re-export it.
"""
try:
    from gymnasium.envs.mujoco.mujoco_env import MujocoEnv  # type: ignore
except Exception:  # pragma: no cover
    # Last resort — a no-op base class so `class Foo(MujocoEnv): ...` is at least
    # syntactically valid at import time.
    class MujocoEnv:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "gym.envs.mujoco.mujoco_env.MujocoEnv stub: not instantiable. "
                "Use compat `gym.make(<d4rl-name>)` instead."
            )
