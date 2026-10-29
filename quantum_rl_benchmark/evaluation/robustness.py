"""
Robustness and generalization evaluations.
Tests policies across multiple independent evaluation seeds and observation perturbations.
"""

import os
import sys
import json
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from visualize_agent import load_trained_agent, find_default_checkpoint


def evaluate_with_noise(env_id: str, agent, noise_std: float = 0.0, n_episodes: int = 10, seed_start: int = 3000) -> Dict[str, float]:
    """Evaluate agent with Gaussian observation noise perturbation."""
    env = gym.make(env_id)
    returns = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        ep_ret = 0
        ep_len = 0
        while not done:
            # Perturb observation if noise_std > 0
            if noise_std > 0:
                perturbed_obs = obs + np.random.normal(0, noise_std, size=obs.shape).astype(obs.dtype)
            else:
                perturbed_obs = obs

            action_out = agent.select_action(perturbed_obs, deterministic=True)
            if isinstance(action_out, tuple):
                action = action_out[0]
            else:
                action = action_out

            if isinstance(action, (int, np.integer)):
                step_act = int(action)
            elif isinstance(action, np.ndarray):
                step_act = action
            else:
                step_act = np.array([action])

            obs, r, term, trunc, _ = env.step(step_act)
            ep_ret += float(r)
            ep_len += 1
            done = term or trunc

        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()

    return {
        "noise_std": noise_std,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "mean_length": float(np.mean(lengths)),
    }


def run_robustness_suite(save_path: str = None) -> Dict[str, Any]:
    """Test policy robustness across perturbation noise levels and multi-seed evaluations."""
    targets = [
        ("DQN", "CartPole-v1", "quantum_rl_benchmark/results/checkpoints/DQN_CartPole-v1_demonstration.pt", "demonstration"),
        ("Q-PPO", "CartPole-v1", None, "original"),
        ("SAC", "Pendulum-v1", None, "optimized"),
        ("Q-SAC", "Pendulum-v1", None, "optimized"),
    ]

    noise_levels = [0.0, 0.02, 0.05, 0.10]
    results = {}

    for algo, env_id, custom_ckpt, regime in targets:
        key = f"{algo}_{env_id}"
        if custom_ckpt and os.path.exists(custom_ckpt):
            ckpt_path = custom_ckpt
        else:
            ckpt_path = find_default_checkpoint(algo, env_id)

        print(f"Testing robustness for {algo} on {env_id} ({ckpt_path})...")
        agent = load_trained_agent(algo, ckpt_path)

        curve = []
        for noise in noise_levels:
            res = evaluate_with_noise(env_id, agent, noise_std=noise, n_episodes=10, seed_start=4000)
            curve.append(res)
            print(f"  Noise std {noise:4.2f} -> Mean Return: {res['mean_return']:7.2f} +/- {res['std_return']:5.2f}")

        results[key] = {
            "algorithm": algo,
            "environment": env_id,
            "regime": regime,
            "checkpoint": ckpt_path,
            "perturbation_curve": curve,
        }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Robustness results saved to: {save_path}")

    return results


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "..", "..", "quantum_rl_benchmark", "results", "data", "robustness_results.json")
    run_robustness_suite(out_file)

