"""Compat wrappers module — re-exports gymnasium.wrappers with shimmed
constructors for the wrappers whose signatures changed since gym 0.21.

Currently shimmed:
  - TransformObservation(env, func) — gymnasium now requires `observation_space`
    as a third positional arg. CORL's `any_percent_bc.py:94` uses the 2-arg form.
  - TransformReward(env, func) — same kind of change.

Anything not explicitly shimmed is re-exported as-is from gymnasium.wrappers.
Mathematical behavior is unchanged.
"""
from __future__ import annotations

import gymnasium.wrappers as _gw


class _AttrForwardMixin:
    """gym-0.21-style attribute forwarding to `self.env`.

    Gymnasium's Wrapper used to forward unknown attribute access to the wrapped
    env, but the modern API dropped `__getattr__`. CORL etc. still call
    `env.seed(seed)` on the outermost wrapper; we restore old-style forwarding
    so those calls land on our OldGymEnvWrapper's `seed`/`get_dataset`/etc.
    """

    def __getattr__(self, name):
        # `object.__getattribute__` raises AttributeError if `self.env` itself
        # is missing — defer to that, do not silently swallow.
        env = object.__getattribute__(self, "env")
        return getattr(env, name)


class TransformObservation(_AttrForwardMixin, _gw.TransformObservation):
    """gym-0.21-style 2-arg constructor; defaults observation_space to env.observation_space.

    Also re-implements reset/step so the wrapper plays nicely with our
    `OldGymEnvWrapper` underneath (which returns gym-0.21 bare-obs / 4-tuple).
    """

    def __init__(self, env, func, observation_space=None):
        if observation_space is None:
            observation_space = env.observation_space
        super().__init__(env, func, observation_space)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return self.func(obs), info
        return self.func(out)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, r, term, trunc, info = out
            return self.func(obs), r, term, trunc, info
        obs, r, done, info = out
        return self.func(obs), r, done, info


class TransformReward(_AttrForwardMixin, _gw.TransformReward):
    """Reward-transform wrapper. Bilingual reset/step for the same reason."""

    def __init__(self, env, func):
        super().__init__(env, func)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, r, term, trunc, info = out
            return obs, self.func(r), term, trunc, info
        obs, r, done, info = out
        return obs, self.func(r), done, info


# Re-export everything else from gymnasium.wrappers untouched.
def __getattr__(name):
    return getattr(_gw, name)


def __dir__():
    return sorted(set(dir(_gw)) | {"TransformObservation", "TransformReward"})
