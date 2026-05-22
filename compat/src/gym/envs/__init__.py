"""Stub gym.envs for old-API repos importing `from gym.envs.mujoco import mujoco_env`.

Real environment construction goes through the compat `gym.make()` -> env_factory
path; this submodule exists only so the repos' top-level imports do not fail.
"""
