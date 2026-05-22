"""Fake `gym` shim. Resolves `import gym` inside any baseline repo to this module.

Re-exports Gymnasium's API but with the *old* gym contract for the operations
baseline repos use:
  - env.reset() returns a bare obs (not (obs, info))
  - env.step() returns (obs, reward, done, info) 4-tuple (not 5-tuple)
  - env.seed(seed) exists
  - gym.make(d4rl_name) routes through env_factory to a Gymnasium env
"""
import gymnasium as _gym
from gymnasium import spaces  # re-export
from gymnasium import error, logger, utils  # noqa: F401
# Compat-patched wrappers module — adds gym-0.21-style 2-arg constructors on
# wrappers whose signature changed in Gymnasium. See gym/wrappers.py.
from . import wrappers  # noqa: F401

from compat_utils.env_factory import make_d4rl_compatible_env, OldGymEnvWrapper  # noqa: F401
from compat_utils.torch_load_patch import apply as _apply_torch_patch

_apply_torch_patch()

# Public API surface used by baselines
Env = _gym.Env
Wrapper = _gym.Wrapper
Space = _gym.Space
ObservationWrapper = _gym.ObservationWrapper
ActionWrapper = _gym.ActionWrapper
RewardWrapper = _gym.RewardWrapper


def make(name, **kwargs):
    return make_d4rl_compatible_env(name, **kwargs)


class GoalEnv(_gym.Env):
    pass


__version__ = "0.21.0-shim"
