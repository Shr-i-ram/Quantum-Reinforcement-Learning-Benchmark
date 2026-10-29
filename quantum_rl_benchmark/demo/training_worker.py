"""
Background Live Training Worker for Streamlit Live Training Studio.
Executes lightweight training sessions asynchronously and logs real-time progress.
"""

import os
import sys
import time
import threading
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from quantum_rl_benchmark.config.settings import (
    get_default_config, ENVIRONMENTS, get_experiment_key
)
from quantum_rl_benchmark.training.trainer import Trainer


class LiveTrainingJob:
    """Manages a single background training session."""

    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.stop_requested = False
        self.is_running = False
        self.error: Optional[str] = None
        self.status = {
            "algorithm": None,
            "environment": None,
            "current_step": 0,
            "total_steps": 0,
            "episodes": 0,
            "latest_reward": 0.0,
            "avg_reward_10": 0.0,
            "elapsed_time_s": 0.0,
            "is_quantum": False,
            "saved_checkpoint_path": None,
            "history": {
                "steps": [],
                "episode_rewards": [],
            }
        }

    def start(
        self,
        algorithm: str,
        env_name: str,
        total_steps: int = 10000,
        learning_rate: float = 1e-3,
        seed: int = 42
    ):
        """Start background training."""
        if self.is_running:
            raise RuntimeError("A training job is already active!")

        self.stop_requested = False
        self.is_running = True
        self.error = None
        self.status["algorithm"] = algorithm
        self.status["environment"] = env_name
        self.status["current_step"] = 0
        self.status["total_steps"] = total_steps
        self.status["episodes"] = 0
        self.status["latest_reward"] = 0.0
        self.status["avg_reward_10"] = 0.0
        self.status["elapsed_time_s"] = 0.0
        self.status["is_quantum"] = algorithm.startswith("Q-")
        self.status["history"] = {"steps": [], "episode_rewards": []}

        self.thread = threading.Thread(
            target=self._run_training_loop,
            args=(algorithm, env_name, total_steps, learning_rate, seed),
            daemon=True
        )
        self.thread.start()

    def stop(self):
        """Signal the training job to stop gracefully."""
        self.stop_requested = True

    def _run_training_loop(self, algo: str, env_name: str, total_steps: int, lr: float, seed: int):
        t0 = time.time()
        try:
            config = get_default_config()
            config.seed = seed

            # Apply custom learning rate
            if algo in ["DQN", "Q-DQN"]:
                cfg_obj = config.dqn if algo == "DQN" else config.q_dqn
                cfg_obj.learning_rate = lr
                cfg_obj.train_freq = 2
            elif algo in ["PPO", "Q-PPO"]:
                cfg_obj = config.ppo if algo == "PPO" else config.q_ppo
                cfg_obj.learning_rate = lr

            trainer = Trainer(config)
            agent = trainer.create_agent(algo, env_name, ENVIRONMENTS)

            env = gym.make(ENVIRONMENTS[env_name]["gymnasium_id"])
            obs, _ = env.reset(seed=seed)
            step = 0
            episode = 0
            ep_ret = 0
            ep_len = 0
            recent_rets = []

            while step < total_steps and not self.stop_requested:
                step += 1
                self.status["current_step"] = step
                self.status["elapsed_time_s"] = time.time() - t0

                # Action selection
                if algo in ["PPO", "Q-PPO"]:
                    act, log_p, val = agent.select_action(obs, deterministic=False)
                    if ENVIRONMENTS[env_name]["action_type"] == "continuous":
                        step_act = act if isinstance(act, np.ndarray) else np.array([act])
                        next_obs, r, term, trunc, _ = env.step(step_act)
                        agent.store_transition_continuous(obs, act, r, term or trunc, log_p, val)
                    else:
                        step_act = int(act)
                        next_obs, r, term, trunc, _ = env.step(step_act)
                        agent.store_transition(obs, act, r, term or trunc, log_p, val)

                    if step % agent.config.rollout_length == 0:
                        agent.update(next_obs)

                elif algo in ["DQN", "Q-DQN"]:
                    act = agent.select_action(obs, deterministic=False)
                    step_act = int(act)
                    next_obs, r, term, trunc, _ = env.step(step_act)
                    agent.store_transition(obs, act, r, next_obs, float(term))
                    agent.update()

                elif algo in ["SAC", "Q-SAC"]:
                    act = agent.select_action(obs, deterministic=False)
                    step_act = act if isinstance(act, np.ndarray) else np.array([act])
                    next_obs, r, term, trunc, _ = env.step(step_act)
                    agent.store_transition(obs, act, r, next_obs, float(term))
                    agent.update()

                obs = next_obs
                ep_ret += float(r)
                ep_len += 1
                done = term or trunc

                if done:
                    episode += 1
                    recent_rets.append(ep_ret)
                    if len(recent_rets) > 10:
                        recent_rets.pop(0)

                    self.status["episodes"] = episode
                    self.status["latest_reward"] = ep_ret
                    self.status["avg_reward_10"] = float(np.mean(recent_rets))
                    self.status["history"]["steps"].append(step)
                    self.status["history"]["episode_rewards"].append(ep_ret)

                    ep_ret = 0
                    ep_len = 0
                    obs, _ = env.reset()

            env.close()

            # Save live-trained checkpoint
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ckpt_path = os.path.join(base_dir, "results", "checkpoints", f"{algo}_{env_name}_livetrained.pt")
            agent.save_checkpoint(ckpt_path)
            self.status["saved_checkpoint_path"] = ckpt_path

        except Exception as e:
            self.error = str(e)
            print(f"Live training error: {e}")
        finally:
            self.is_running = False


# Global singleton instance for live training
TRAINING_JOB = LiveTrainingJob()
