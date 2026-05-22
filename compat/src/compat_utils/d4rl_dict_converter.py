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
    """Module-level normalized scoring (some repos call d4rl.get_normalized_score)."""
    minari_id = D4RL_NAME_TO_MINARI_ID[d4rl_name]
    ds = minari.load_dataset(minari_id, download=True)
    ref_min = getattr(ds.spec, "reference_min_score", None)
    ref_max = getattr(ds.spec, "reference_max_score", None)
    if ref_min is None:
        ref_min = 0.0
    if ref_max is None or ref_max == ref_min:
        ref_max = ref_min + 1.0
    return (raw_return - ref_min) / (ref_max - ref_min)
