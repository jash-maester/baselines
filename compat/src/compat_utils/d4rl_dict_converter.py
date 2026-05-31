"""Flatten Minari episodes into D4RL's flat-array dict, plus replacement helpers
for the d4rl module functions baselines call."""
import numpy as np
import minari

from compat_utils.minari_mapping import D4RL_NAME_TO_MINARI_ID


def _episodes_to_d4rl_dict(minari_dataset, include_next_obs=True):
    """Flatten Minari episodes into a D4RL-style flat-array dict."""
    obs_list, act_list, rew_list, nxt_list = [], [], [], []
    term_list, timo_list = [], []

    for ep in minari_dataset.iterate_episodes():
        n = len(ep.actions)
        if n == 0:
            continue
        obs_arr = np.asarray(ep.observations)
        # Minari stores n+1 observations per n-action episode.
        obs_list.append(obs_arr[:n])
        if include_next_obs:
            nxt_list.append(obs_arr[1 : n + 1])
        act_list.append(np.asarray(ep.actions))
        rew_list.append(np.asarray(ep.rewards))
        t = np.zeros(n, dtype=bool)
        to = np.zeros(n, dtype=bool)
        # Minari stores termination/truncation per step. Old-style D4RL only
        # uses the last-step flag; mark accordingly.
        if hasattr(ep, "terminations"):
            term_arr = np.asarray(ep.terminations)
            if term_arr.size >= 1:
                t[-1] = bool(term_arr[-1])
        if hasattr(ep, "truncations"):
            to_arr = np.asarray(ep.truncations)
            if to_arr.size >= 1:
                to[-1] = bool(to_arr[-1])
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
        out["next_observations"] = np.concatenate(nxt_list).astype(np.float32)
    return out


def _resolve_d4rl_name(env):
    if hasattr(env, "spec_d4rl_name"):
        return env.spec_d4rl_name
    if hasattr(env, "_d4rl_name"):
        return env._d4rl_name
    raise AttributeError(
        "env does not carry a D4RL-style name. Was it created via gym.make() from the compat shim?"
    )


def qlearning_dataset(env, **kwargs):
    """Replacement for d4rl.qlearning_dataset(env)."""
    d4rl_name = _resolve_d4rl_name(env)
    minari_id = D4RL_NAME_TO_MINARI_ID[d4rl_name]
    ds = minari.load_dataset(minari_id, download=True)
    return _episodes_to_d4rl_dict(ds, include_next_obs=True)


def get_dataset(env, **kwargs):
    """Replacement for env.get_dataset() — used by LDCQ."""
    d4rl_name = _resolve_d4rl_name(env)
    minari_id = D4RL_NAME_TO_MINARI_ID[d4rl_name]
    ds = minari.load_dataset(minari_id, download=True)
    return _episodes_to_d4rl_dict(ds, include_next_obs=False)


def get_normalized_score(d4rl_name, raw_return):
    """Module-level normalized scoring (some repos call d4rl.get_normalized_score).

    EDA and QGPO normalize via this module function; CORL/DQL normalize via
    OldGymEnvWrapper.get_normalized_score. The Minari datasets ship no
    `spec.reference_min/max_score`, so the old implementation here fell back to
    (0, 1) and returned the *raw* return — making EDA/QGPO scores meaningless
    and inconsistent with CORL. Delegate to the same Minari-derived reference
    scores (`_minari_ref_scores`) the env wrapper uses, so all methods share one
    normalization. Returns the 0-1 fraction; callers multiply by 100.
    Lazy import avoids a circular import (env_factory imports this module).
    """
    from compat_utils.env_factory import _minari_ref_scores
    ref_min, ref_max = _minari_ref_scores(d4rl_name)
    if ref_max == ref_min:
        return 0.0
    return (raw_return - ref_min) / (ref_max - ref_min)
