"""Wrap a Gymnasium env to expose the old gym contract baseline repos expect."""
import gymnasium as gym

from compat_utils.minari_mapping import D4RL_NAME_TO_GYM_ENV, D4RL_NAME_TO_MINARI_ID
from compat_utils.d4rl_dict_converter import _episodes_to_d4rl_dict


# Minari-native ref-score derivation. Recreates the
# `env.get_normalized_score(score)` semantics our launchers (CORL et al.)
# call into, without depending on D4RL.
#
# Three paths, tried in order:
#   (1) If the loaded Minari dataset's on-disk metadata.json carries
#       `ref_min_score` / `ref_max_score`, use those (this is what Minari's
#       own `minari.get_normalized_score(ds, ...)` reads). Future-proof:
#       Minari may start populating these for the mujoco/* family.
#   (2) Otherwise derive ref_max as the mean episodic return of the
#       corresponding `expert` Minari dataset — the closest Minari-only
#       proxy for D4RL's "expert reference policy" anchor. ref_min = 0
#       (no random-policy episodes ship in mujoco/*-v0).
#   (3) If even (2) is unreachable (no expert dataset cached / download
#       disabled / unknown env), return (0.0, 1.0) — `get_normalized_score`
#       becomes the identity, eval still progresses.
#
# Path (2) gives DIFFERENT numbers than D4RL's canonical (random-policy,
# expert-mean) ref pair, by design — comparing our Minari-normalized
# numbers against paper-reported D4RL-normalized numbers is the point of
# the dashboard, and a non-zero delta is informative, not a bug.
_REF_SCORE_CACHE: dict = {}


def _minari_ref_scores(d4rl_name):
    """Return (ref_min, ref_max) derived purely from Minari for `d4rl_name`.

    Cached after first call per env (the expert-dataset return computation
    iterates 1000 episodes and is too slow to run on every eval rollout).
    """
    if d4rl_name in _REF_SCORE_CACHE:
        return _REF_SCORE_CACHE[d4rl_name]

    import minari, numpy as np

    # Path 1: metadata-supplied refs.
    try:
        minari_id = D4RL_NAME_TO_MINARI_ID[d4rl_name]
        ds = minari.load_dataset(minari_id, download=False)
        md = ds.storage.metadata
        ref_min = md.get("ref_min_score")
        ref_max = md.get("ref_max_score")
        if ref_min is not None and ref_max is not None:
            res = (float(ref_min), float(ref_max))
            _REF_SCORE_CACHE[d4rl_name] = res
            return res
    except Exception:
        pass

    # Path 2: derive ref_max as mean episodic return of the expert variant.
    # `<env>-medium-v0` → `<env>-expert-v0` → `mujoco/<env>/expert-v0`.
    try:
        env_stem = d4rl_name.split("-")[0]  # "halfcheetah-medium-v0" -> "halfcheetah"
        version  = d4rl_name.rsplit("-", 1)[1]  # "v0"
        expert_d4rl = f"{env_stem}-expert-{version}"
        expert_minari_id = D4RL_NAME_TO_MINARI_ID[expert_d4rl]
        expert_ds = minari.load_dataset(expert_minari_id, download=False)
        returns = np.array(
            [float(np.sum(ep.rewards)) for ep in expert_ds.iterate_episodes()]
        )
        ref_max = float(returns.mean())
        ref_min = 0.0
        res = (ref_min, ref_max)
        _REF_SCORE_CACHE[d4rl_name] = res
        return res
    except Exception:
        pass

    # Path 3: identity passthrough.
    _REF_SCORE_CACHE[d4rl_name] = (0.0, 1.0)
    return (0.0, 1.0)


class OldGymEnvWrapper(gym.Wrapper):
    """Wraps a Gymnasium env to expose the old gym contract used by baselines:
      - reset() returns bare obs
      - step() returns 4-tuple
      - seed(seed) exists and is a no-op spec sink
      - get_dataset(), get_normalized_score(), spec_d4rl_name tags
    """

    def __init__(self, env, d4rl_name):
        super().__init__(env)
        self._d4rl_name = d4rl_name

    def reset(self, **kwargs):
        # Old API returns only obs; new API returns (obs, info).
        seed = kwargs.pop("seed", None)
        if seed is not None:
            obs, _info = self.env.reset(seed=seed, **kwargs)
        else:
            obs, _info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        if isinstance(info, dict):
            info["terminated"] = bool(terminated)
            info["truncated"] = bool(truncated)
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is not None:
            try:
                self.action_space.seed(seed)
                self.observation_space.seed(seed)
            except Exception:
                pass
        return [seed]

    def get_dataset(self):
        """Mimics d4rl's env.get_dataset() — flat dict of full dataset."""
        import minari
        minari_id = D4RL_NAME_TO_MINARI_ID[self._d4rl_name]
        ds = minari.load_dataset(minari_id, download=True)
        return _episodes_to_d4rl_dict(ds, include_next_obs=False)

    def get_normalized_score(self, raw_return):
        """D4RL's `env.get_normalized_score(score)` — `(raw-ref_min)/(ref_max-ref_min)`.

        CORL et al. then multiply by 100 to recover the paper-canonical
        0–100 score. Proxies through Minari's metadata when available
        (`minari.get_normalized_score(ds, returns)`), and otherwise falls
        back to the canonical D4RL constants — see `_minari_ref_scores`.
        """
        ref_min, ref_max = _minari_ref_scores(self._d4rl_name)
        if ref_max == ref_min:
            return 0.0
        return (raw_return - ref_min) / (ref_max - ref_min)

    @property
    def spec_d4rl_name(self):
        return self._d4rl_name

    def __getattr__(self, name):
        """Old-gym-0.21-style attribute forwarding (Gymnasium ≥1.0 dropped it).

        Diffuser does `wrapped_env._max_episode_steps` expecting fallthrough
        to the inner TimeLimit wrapper. Mirror that.
        """
        # Avoid infinite recursion for our own attributes.
        if name in ("env", "_d4rl_name"):
            raise AttributeError(name)
        env = object.__getattribute__(self, "env")
        return getattr(env, name)


def _attach_d4rl_methods_to_inner(inner, wrapper):
    """Attach d4rl-style helpers to the *unwrapped* env instance too.

    Diffuser's `diffuser/datasets/d4rl.py` does
        wrapped_env = gym.make(name)
        env = wrapped_env.unwrapped
        ...
        dataset = env.get_dataset()
    which strips our OldGymEnvWrapper. Without this attach step, the inner
    Gymnasium env would AttributeError on get_dataset / get_normalized_score.
    Recorded as patch #3 in baselines/RUN_LOG.md (B1 fix).
    """
    inner._d4rl_name = wrapper._d4rl_name
    inner.spec_d4rl_name = wrapper._d4rl_name
    inner.get_dataset = wrapper.get_dataset
    inner.get_normalized_score = wrapper.get_normalized_score
    # Diffuser also does `env.seed(seed)` on the unwrapped env (sequence.py:23).
    # Gymnasium dropped this method, so attach the wrapper's old-API seed.
    inner.seed = wrapper.seed


# Diffuser's renderer (rendering.py:env_map) requests fully-observed env
# variants like "HalfCheetahFullObs-v2". Those classes live inside Diffuser
# and aren't installed via gym.register in our shim. Map the FullObs name to
# the corresponding base Gymnasium env so gym.make() resolves.
_FULLOBS_FALLBACK = {
    "HalfCheetahFullObs-v2": "HalfCheetah-v5",
    "HopperFullObs-v2":      "Hopper-v5",
    "Walker2dFullObs-v2":    "Walker2d-v5",
    "AntFullObs-v2":         "Ant-v5",
}


def make_d4rl_compatible_env(d4rl_name, **kwargs):
    """'halfcheetah-medium-v2' -> Gymnasium HalfCheetah-v5 wrapped + tagged.

    Accepts either a D4RL-style name (mapped via minari_mapping), a direct
    Gymnasium env id (e.g. 'HalfCheetah-v5'), or a Diffuser-style 'FullObs'
    variant (mapped to its base Gymnasium env via _FULLOBS_FALLBACK).
    """
    if d4rl_name in _FULLOBS_FALLBACK:
        gym_name = _FULLOBS_FALLBACK[d4rl_name]
        env = gym.make(gym_name, **kwargs)
    elif d4rl_name in D4RL_NAME_TO_GYM_ENV:
        gym_name = D4RL_NAME_TO_GYM_ENV[d4rl_name]
        env = gym.make(gym_name, **kwargs)
    else:
        env = gym.make(d4rl_name, **kwargs)
    wrapper = OldGymEnvWrapper(env, d4rl_name)
    _attach_d4rl_methods_to_inner(wrapper.unwrapped, wrapper)
    return wrapper
