"""
Train a reference demonstration classical DQN agent on CartPole-v1.
Achieves perfect/near-perfect performance (475 - 500 return)
with proper bootstrap handling for truncations and periodic evaluation.

Saves checkpoint to:
  quantum_rl_benchmark/results/checkpoints/DQN_CartPole-v1_demonstration.pt
and results to:
  quantum_rl_benchmark/results/data/DQN_CartPole-v1_demonstration_results.json
"""

import os
import sys
import json
import time
import torch
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_rl_benchmark.models.classical.dqn import DQNAgent, QNetwork
from quantum_rl_benchmark.config.settings import DQNConfig, get_default_config


def train_demonstration_dqn():
    print("=" * 70)
    print("TRAINING DEMONSTRATION CLASSICAL AGENT: DQN on CartPole-v1")
    print("=" * 70)

    env = gym.make("CartPole-v1")
    obs_dim = 4
    n_actions = 2
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    config = DQNConfig(
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_steps=6000,
        batch_size=64,
        replay_buffer_size=30000,
        target_update_freq=200,
        hidden_dim=128,
        train_freq=2,
    )

    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, config=config, seed=seed)

    max_steps = 45000
    step = 0
    episode = 0
    best_eval_mean = 0.0
    best_state_dict = None

    t0 = time.time()
    eval_history = []

    print(f"Target: achieve mean evaluation score >= 475.0 (Gymnasium solved criterion)")

    while step < max_steps:
        obs, _ = env.reset(seed=seed + episode)
        ep_ret = 0
        ep_len = 0
        done = False

        while not done:
            step += 1
            # Epsilon decay
            agent.total_steps = step
            agent.epsilon = max(
                config.epsilon_end,
                config.epsilon_start - (config.epsilon_start - config.epsilon_end) * step / config.epsilon_decay_steps
            )

            # Select action
            action = agent.select_action(obs, deterministic=False)

            # Step env
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition: use terminated (NOT truncated) for bootstrapping future value on timeout
            agent.store_transition(obs, action, reward, next_obs, float(terminated))

            # Update
            agent.update()

            obs = next_obs
            ep_ret += reward
            ep_len += 1

    # Periodic deterministic evaluation every 2,000 steps
            if step % 2000 == 0:
                eval_env = gym.make("CartPole-v1")
                eval_returns = []
                for eval_ep in range(10):
                    e_obs, _ = eval_env.reset(seed=5000 + eval_ep)
                    e_done = False
                    e_ret = 0
                    while not e_done:
                        e_act = agent.select_action(e_obs, deterministic=True)
                        e_obs, e_r, e_term, e_trunc, _ = eval_env.step(e_act)
                        e_ret += e_r
                        e_done = e_term or e_trunc
                    eval_returns.append(e_ret)
                eval_env.close()

                eval_mean = float(np.mean(eval_returns))
                eval_std = float(np.std(eval_returns))
                eval_history.append({"step": step, "eval_mean": eval_mean, "eval_std": eval_std})
                print(f"  [Step {step:05d}] Eval Mean: {eval_mean:6.2f} +/- {eval_std:5.2f} (Best: {best_eval_mean:6.2f})")

                if eval_mean > best_eval_mean:
                    best_eval_mean = eval_mean
                    best_state_dict = {
                        "q_network_state_dict": agent.q_network.state_dict(),
                        "target_network_state_dict": agent.target_network.state_dict(),
                        "total_steps": step,
                        "episode_count": episode,
                    }

                if eval_mean >= 495.0 and step >= 12000:
                    print(f"  >>> Solved CartPole-v1 with score {eval_mean:.2f}! Terminating early.")
                    break

        agent.record_episode(ep_ret, ep_len)
        episode += 1

        if best_eval_mean >= 495.0 and step >= 12000:
            break

    total_training_time = time.time() - t0

    # Load best weights into agent
    if best_state_dict is not None:
        agent.q_network.load_state_dict(best_state_dict["q_network_state_dict"])
        agent.target_network.load_state_dict(best_state_dict["target_network_state_dict"])

    # Final rigorous 20-episode evaluation
    final_returns = []
    final_lengths = []
    for eval_ep in range(20):
        e_obs, _ = env.reset(seed=1000 + eval_ep)
        e_done = False
        e_ret = 0
        e_len = 0
        while not e_done:
            e_act = agent.select_action(e_obs, deterministic=True)
            e_obs, e_r, e_term, e_trunc, _ = env.step(e_act)
            e_ret += e_r
            e_len += 1
            e_done = e_term or e_trunc
        final_returns.append(e_ret)
        final_lengths.append(e_len)

    env.close()

    final_mean = float(np.mean(final_returns))
    final_std = float(np.std(final_returns))
    final_median = float(np.median(final_returns))
    final_min = float(np.min(final_returns))
    final_max = float(np.max(final_returns))

    print("\n" + "=" * 70)
    print(f"DEMONSTRATION AGENT FINAL EVALUATION (20 episodes):")
    print(f"  Mean Return:   {final_mean:.2f} +/- {final_std:.2f}")
    print(f"  Median Return: {final_median:.2f}")
    print(f"  Min / Max:     {final_min:.1f} / {final_max:.1f}")
    print(f"  Training Time: {total_training_time:.1f}s across {step} steps")
    print("=" * 70)

    # Save demonstration checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "quantum_rl_benchmark", "results", "checkpoints")
    demo_ckpt_path = os.path.join(ckpt_dir, "DQN_CartPole-v1_demonstration.pt")
    agent.save_checkpoint(demo_ckpt_path)
    print(f"Saved demonstration checkpoint to: {demo_ckpt_path}")

    # Save demonstration results JSON
    data_dir = os.path.join(os.path.dirname(__file__), "..", "quantum_rl_benchmark", "results", "data")
    demo_res_path = os.path.join(data_dir, "DQN_CartPole-v1_demonstration_results.json")
    results = {
        "algorithm": "DQN",
        "environment": "CartPole-v1",
        "model_variant": "demonstration_reference",
        "status": "completed",
        "configuration_regime": "demonstration",
        "total_steps": step,
        "episodes": episode,
        "training_time": total_training_time,
        "eval_results": {
            "mean_return": final_mean,
            "std_return": final_std,
            "median_return": final_median,
            "min_return": final_min,
            "max_return": final_max,
            "mean_length": float(np.mean(final_lengths)),
            "n_episodes": 20,
            "episode_returns": final_returns,
            "episode_lengths": final_lengths,
        },
        "agent_info": agent.get_info(),
        "history": {
            "eval_history": eval_history,
            "episode_rewards": agent.history["episode_rewards"],
            "episode_lengths": agent.history["episode_lengths"],
        },
    }

    with open(demo_res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved demonstration results to: {demo_res_path}")


if __name__ == "__main__":
    train_demonstration_dqn()
