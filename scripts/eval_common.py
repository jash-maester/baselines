"""Shared evaluation helper used by per-method eval launchers.

Single source of truth for episode count, deterministic toggle, episode-length cap,
and normalized scoring. See docs/06_evaluation_protocol.md.
"""
import numpy as np
import torch

import gym  # compat shim


def evaluate_policy(
    policy_fn,
    d4rl_env_name,
    n_episodes: int = 10,
    seed_offset: int = 0,
    max_episode_steps: int = 1000,
    deterministic: bool = True,
):
    """Run `n_episodes`. Returns dict with raw_returns/normalized_scores/lengths."""
    env = gym.make(d4rl_env_name)
    raw_returns, lens, norm_scores = [], [], []

    for ep in range(n_episodes):
        ep_seed = seed_offset * 1000 + ep
        obs = env.reset(seed=ep_seed)
        ep_return = 0.0
        ep_len = 0
        done = False
        while not done and ep_len < max_episode_steps:
            with torch.no_grad():
                action = policy_fn(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
        raw_returns.append(ep_return)
        lens.append(ep_len)
        norm_scores.append(float(env.get_normalized_score(ep_return)) * 100.0)

    env.close()
    return dict(
        raw_returns=raw_returns,
        normalized_scores=norm_scores,
        episode_lengths=lens,
        raw_return_mean=float(np.mean(raw_returns)),
        raw_return_std=float(np.std(raw_returns)),
        normalized_score_mean=float(np.mean(norm_scores)),
        normalized_score_std=float(np.std(norm_scores)),
        episode_count=int(n_episodes),
    )
