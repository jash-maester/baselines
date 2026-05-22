"""Stub gym.envs.mujoco — exposes `mujoco_env.MujocoEnv` as gymnasium's MujocoEnv
so old-API mujoco env subclasses defined inside baseline repos remain importable.

These classes (e.g. Diffuser's `HalfCheetahFullObsEnv`) are never *instantiated*
in our path — actual envs come from `gym.make(...)` -> env_factory ->
Gymnasium's official mujoco envs. The classes only need to be importable so
that module loading does not fail.
"""
from . import mujoco_env  # noqa: F401
