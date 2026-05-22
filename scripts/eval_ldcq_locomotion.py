"""Locomotion eval for LDCQ.

LDCQ ships eval only for AntMaze/Kitchen/Maze2D under `eval/plan_skills_diffusion*.py`.
For our matrix we need a locomotion eval that does NOT modify LDCQ source.

We import LDCQ's classes as libraries and run the same Q-argmax-over-latents
inference loop they already use, but against locomotion envs constructed via
our compat shim.

Inputs:
    --env_name e.g. halfcheetah-medium-v0
    --skill_ckpt path to stage-1 ckpt
    --diffusion_ckpt path to stage-2 ckpt
    --q_ckpt path to stage-3 ckpt
    --num_eval_episodes (default 10)

This script logs into the currently-active MLflow run (started by the parent
launcher); if no run is active it creates one tagged `stage=eval`.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LDCQ_ROOT = ROOT / "repos" / "ldcq"
sys.path.insert(0, str(LDCQ_ROOT))
sys.path.insert(0, str(ROOT / "compat" / "src"))

import gym  # noqa: E402 (compat shim)
import mlflow  # noqa: E402

# Late-bound LDCQ imports; the upstream module side-effects are heavy.
def _load_ldcq_models(skill_ckpt: Path, diffusion_ckpt: Path, q_ckpt: Path,
                     state_dim: int, action_dim: int, device: str):
    from models.skill_model import SkillModel
    from models.diffusion_models import Model_Cond_Diffusion, Model_mlp_diff_embed
    from models.dqn import DQN

    skill_model = SkillModel(
        state_dim=state_dim, a_dim=action_dim, z_dim=16, h_dim=256,
        horizon=10, a_dist="normal", beta=0.05,
        fixed_sig=None, encoder_type="gru", state_decoder_type="mlp",
        policy_decoder_type="autoregressive", per_element_sigma=True,
        conditional_prior=True,
    ).to(device)
    ckpt = torch.load(str(skill_ckpt), map_location=device)
    skill_model.load_state_dict(ckpt["model_state_dict"])

    diff_net = Model_mlp_diff_embed(z_dim=16, h_dim=512, state_dim=state_dim,
                                    embed_dim=128, net_type="unet").to(device)
    diffusion_model = Model_Cond_Diffusion(diff_net, betas=(1e-4, 0.02),
                                           n_T=100, drop_prob=0.0,
                                           x_dim=16, y_dim=state_dim).to(device)
    diff_ckpt = torch.load(str(diffusion_ckpt), map_location=device)
    diffusion_model.load_state_dict(diff_ckpt["model_state_dict"])

    q_model = DQN(state_dim=state_dim, z_dim=16, h_dim=512, gamma=0.995,
                  alpha=0.6, net_type="unet").to(device)
    q_ckpt_state = torch.load(str(q_ckpt), map_location=device)
    q_model.load_state_dict(q_ckpt_state["model_state_dict"])

    skill_model.eval(); diffusion_model.eval(); q_model.eval()
    return skill_model, diffusion_model, q_model


def evaluate(env_name: str, skill_ckpt: Path, diffusion_ckpt: Path,
             q_ckpt: Path, num_eval_episodes: int = 10,
             total_prior_samples: int = 100, horizon: int = 10,
             seed_offset: int = 0, device: str = "cuda"):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    skill_model, diffusion_model, q_model = _load_ldcq_models(
        skill_ckpt, diffusion_ckpt, q_ckpt, state_dim, action_dim, device,
    )

    raw_returns, norm_scores, lens = [], [], []
    for ep in range(num_eval_episodes):
        seed = seed_offset * 1000 + ep
        obs = env.reset(seed=seed)
        ep_ret, ep_len, done = 0.0, 0, False
        z_t = None
        steps_in_skill = 0
        while not done and ep_len < 1000:
            if z_t is None or steps_in_skill >= horizon:
                # Sample candidate latents from the diffusion prior and pick best by Q.
                s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                s_batch = s_t.repeat(total_prior_samples, 1)
                with torch.no_grad():
                    z_cands = diffusion_model.sample(s_batch)
                    q_vals = q_model(s_batch, z_cands).squeeze(-1)
                z_t = z_cands[q_vals.argmax()].unsqueeze(0)
                steps_in_skill = 0
            s_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                act = skill_model.decoder.policy(s_t, z_t).cpu().numpy().squeeze(0)
            obs, reward, done, info = env.step(act)
            ep_ret += float(reward)
            ep_len += 1
            steps_in_skill += 1
        raw_returns.append(ep_ret)
        lens.append(ep_len)
        norm_scores.append(float(env.get_normalized_score(ep_ret)) * 100.0)
        print(f"  episode {ep}: ret={ep_ret:.1f} norm={norm_scores[-1]:.2f} len={ep_len}")

    env.close()
    return dict(
        raw_returns=raw_returns,
        normalized_scores=norm_scores,
        episode_lengths=lens,
        raw_return_mean=float(np.mean(raw_returns)),
        raw_return_std=float(np.std(raw_returns)),
        normalized_score_mean=float(np.mean(norm_scores)),
        normalized_score_std=float(np.std(norm_scores)),
        episode_count=int(num_eval_episodes),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", required=True)
    p.add_argument("--skill_ckpt", type=Path, required=True)
    p.add_argument("--diffusion_ckpt", type=Path, required=True)
    p.add_argument("--q_ckpt", type=Path, required=True)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--total_prior_samples", type=int, default=100)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    metrics = evaluate(
        args.env_name, args.skill_ckpt, args.diffusion_ckpt, args.q_ckpt,
        num_eval_episodes=args.num_eval_episodes,
        total_prior_samples=args.total_prior_samples,
        horizon=args.horizon, seed_offset=args.seed, device=args.device,
    )

    print()
    print(f"Eval over {args.num_eval_episodes} episodes:")
    print(f"  raw_return:     {metrics['raw_return_mean']:.1f} ± {metrics['raw_return_std']:.1f}")
    print(f"  normalized_x100: {metrics['normalized_score_mean']:.2f} ± {metrics['normalized_score_std']:.2f}")

    # Log into whatever run is active (or create one tagged eval).
    if mlflow.active_run() is None:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5555"))
        mlflow.set_experiment("baselines")
        mlflow.start_run(run_name=f"ldcq_{args.env_name}_seed{args.seed}_eval")

    mlflow.log_metric("final.raw_return_mean", metrics["raw_return_mean"])
    mlflow.log_metric("final.raw_return_std",  metrics["raw_return_std"])
    mlflow.log_metric("final.normalized_score_mean", metrics["normalized_score_mean"])
    mlflow.log_metric("final.normalized_score_std",  metrics["normalized_score_std"])
    mlflow.log_metric("final.episode_count", metrics["episode_count"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
