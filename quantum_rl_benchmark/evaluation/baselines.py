"""
Baseline policies for all 5 Gymnasium environments.
Provides random and simple heuristic/zero baselines for rigorous comparison.
"""

import os
import json
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List


class RandomPolicy:
    """Selects uniformly random actions from the environment action space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Any:
        return self.action_space.sample()


class CartPoleHeuristicPolicy:
    """PD-style angle heuristic: push right if leaning right, left if leaning left."""
    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        x, x_dot, theta, theta_dot = obs
        # Anticipate angle using angular velocity
        predicted_theta = theta + 0.15 * theta_dot
        return 1 if predicted_theta > 0 else 0


class MountainCarHeuristicPolicy:
    """Energy-pumping heuristic: push in direction of velocity."""
    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        pos, vel = obs
        if vel > 0:
            return 2  # push right
        elif vel < 0:
            return 0  # push left
        return 2


class LunarLanderHeuristicPolicy:
    """Simple attitude and descent stabilization heuristic."""
    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        # obs: [x, y, vx, vy, theta, vtheta, left_leg_contact, right_leg_contact]
        x, y, vx, vy, theta, vtheta, l_contact, r_contact = obs
        target_angle = np.clip(x * 0.5 + vx * 1.0, -0.4, 0.4)
        angle_err = target_angle - theta

        if angle_err > 0.08:
            return 3  # Fire right engine to tilt left
        elif angle_err < -0.08:
            return 1  # Fire left engine to tilt right
        elif vy < -0.2:
            return 2  # Fire main engine to slow descent
        return 0  # Do nothing


class PendulumZeroPolicy:
    """Zero-torque passive policy baseline."""
    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


class PendulumHeuristicPolicy:
    """Damping heuristic: opposes angular velocity."""
    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # obs: [cos(theta), sin(theta), theta_dot]
        theta_dot = obs[2]
        return np.array([-np.clip(theta_dot * 0.5, -2.0, 2.0)], dtype=np.float32)


class BipedalWalkerHeuristicPolicy:
    """Periodic alternating sinusoidal rhythm heuristic."""
    def __init__(self):
        self.step_count = 0

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        self.step_count += 1
        phase = self.step_count * 0.1
        a1 = float(np.sin(phase))
        a2 = float(np.cos(phase))
        a3 = float(-np.sin(phase))
        a4 = float(-np.cos(phase))
        return np.array([a1, a2, a3, a4], dtype=np.float32)


def evaluate_policy(env_id: str, policy, n_episodes: int = 20, seed_start: int = 2000) -> Dict[str, Any]:
    """Deterministically evaluate a policy on an environment across fixed seeds."""
    env = gym.make(env_id)
    returns = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        if hasattr(policy, "step_count"):
            policy.step_count = 0
        done = False
        ep_ret = 0
        ep_len = 0
        while not done:
            act = policy.select_action(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(act)
            ep_ret += float(r)
            ep_len += 1
            done = term or trunc
        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "n_episodes": n_episodes,
        "episode_returns": returns,
        "episode_lengths": lengths,
    }


def run_all_baselines(save_path: str = None) -> Dict[str, Any]:
    """Run random and heuristic baselines across all 5 environments."""
    envs = ["CartPole-v1", "MountainCar-v0", "LunarLander-v3", "Pendulum-v1", "BipedalWalker-v3"]
    results = {}

    for env_id in envs:
        env = gym.make(env_id)
        rand_pol = RandomPolicy(env.action_space)
        results[f"Random_{env_id}"] = {
            "policy_type": "random",
            "environment": env_id,
            "metrics": evaluate_policy(env_id, rand_pol, n_episodes=20),
        }
        env.close()

    # Heuristic baselines
    results["Heuristic_CartPole-v1"] = {
        "policy_type": "heuristic",
        "environment": "CartPole-v1",
        "metrics": evaluate_policy("CartPole-v1", CartPoleHeuristicPolicy(), n_episodes=20),
    }
    results["Heuristic_MountainCar-v0"] = {
        "policy_type": "heuristic",
        "environment": "MountainCar-v0",
        "metrics": evaluate_policy("MountainCar-v0", MountainCarHeuristicPolicy(), n_episodes=20),
    }
    results["Heuristic_LunarLander-v3"] = {
        "policy_type": "heuristic",
        "environment": "LunarLander-v3",
        "metrics": evaluate_policy("LunarLander-v3", LunarLanderHeuristicPolicy(), n_episodes=20),
    }
    results["ZeroTorque_Pendulum-v1"] = {
        "policy_type": "zero_action",
        "environment": "Pendulum-v1",
        "metrics": evaluate_policy("Pendulum-v1", PendulumZeroPolicy(), n_episodes=20),
    }
    results["Heuristic_Pendulum-v1"] = {
        "policy_type": "heuristic",
        "environment": "Pendulum-v1",
        "metrics": evaluate_policy("Pendulum-v1", PendulumHeuristicPolicy(), n_episodes=20),
    }
    results["Heuristic_BipedalWalker-v3"] = {
        "policy_type": "heuristic",
        "environment": "BipedalWalker-v3",
        "metrics": evaluate_policy("BipedalWalker-v3", BipedalWalkerHeuristicPolicy(), n_episodes=20),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Baselines saved to: {save_path}")

    return results


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "..", "results", "data", "baselines_results.json")
    res = run_all_baselines(out_file)
    print("Baseline evaluation complete. Summary:")
    for k, v in res.items():
        m = v["metrics"]
        print(f"  {k:30s} -> Mean Return: {m['mean_return']:8.2f} +/- {m['std_return']:6.2f} (Median: {m['median_return']:8.2f})")
