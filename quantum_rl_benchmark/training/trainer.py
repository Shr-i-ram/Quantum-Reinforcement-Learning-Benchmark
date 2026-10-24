"""
Core training infrastructure.

Handles the training loop, checkpointing, resumability, and logging
for all algorithm-environment combinations.
"""

import os
import json
import time
import signal
import sys
import numpy as np
from typing import Dict, Any, Optional
import torch

from quantum_rl_benchmark.config.settings import (
    ExperimentConfig, ENVIRONMENTS, EXPERIMENT_MATRIX, get_experiment_key,
)
from quantum_rl_benchmark.environments.manager import EnvironmentManager
from quantum_rl_benchmark.models.classical.ppo import PPOAgent
from quantum_rl_benchmark.models.classical.dqn import DQNAgent
from quantum_rl_benchmark.models.classical.sac import SACAgent
from quantum_rl_benchmark.models.quantum.q_ppo import QPPOAgent
from quantum_rl_benchmark.models.quantum.q_dqn import QDQNAgent
from quantum_rl_benchmark.models.quantum.q_sac import QSACAgent
from quantum_rl_benchmark.models.quantum.circuit import reset_circuit_eval_counter


class ExperimentLogger:
    """Logs experiment results to JSON files."""

    def __init__(self, log_dir: str, experiment_key: str):
        self.log_dir = log_dir
        self.experiment_key = experiment_key
        self.log_path = os.path.join(log_dir, f"{experiment_key}.json")
        self.data = {
            "experiment_key": experiment_key,
            "status": "running",
            "start_time": time.time(),
            "end_time": None,
            "total_steps": 0,
            "episodes": 0,
        }

    def update(self, **kwargs):
        self.data.update(kwargs)

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def finish(self, status: str = "completed"):
        self.data["status"] = status
        self.data["end_time"] = time.time()
        self.save()


class Trainer:
    """
    Trains a single algorithm on a single environment.

    Handles the full training lifecycle including:
    - Agent creation (or resumption from checkpoint)
    - Environment interaction loop
    - Regular checkpointing
    - Training logging
    - Interrupt handling
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.env_manager = EnvironmentManager(seed=config.seed)
        self._interrupted = False

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signal for graceful checkpoint saving."""
        print(f"\n[INTERRUPT] Signal {signum} received. Will save checkpoint and exit.")
        self._interrupted = True

    def get_checkpoint_path(self, algorithm: str, env_name: str) -> str:
        """Get the latest checkpoint path."""
        key = get_experiment_key(algorithm, env_name)
        return os.path.join(self.config.checkpoint_dir, f"{key}_latest.pt")

    def get_best_checkpoint_path(self, algorithm: str, env_name: str) -> str:
        """Get the best checkpoint path."""
        key = get_experiment_key(algorithm, env_name)
        return os.path.join(self.config.checkpoint_dir, f"{key}_best.pt")

    def get_periodic_checkpoint_path(self, algorithm: str, env_name: str, step: int) -> str:
        """Get a periodic checkpoint path."""
        key = get_experiment_key(algorithm, env_name)
        return os.path.join(self.config.checkpoint_dir, f"{key}_step{step}.pt")

    def find_latest_checkpoint(self, algorithm: str, env_name: str) -> Optional[str]:
        """Find the latest valid checkpoint for resumption."""
        latest = self.get_checkpoint_path(algorithm, env_name)
        if os.path.exists(latest):
            return latest
        return None

    def get_env_config(self, env_name: str) -> Dict:
        """Get environment-specific training config."""
        import dataclasses
        # Env-specific overrides start as base, then training config applies on top
        env_overrides = self.config.env_configs.get(env_name, {})
        base = dataclasses.asdict(self.config.training)
        # Apply env overrides first, then training config overrides env
        base.update(env_overrides)
        return base

    def create_agent(self, algorithm: str, env_name: str, env_specs: Dict):
        """Create a new agent for the given algorithm and environment."""
        spec = env_specs[env_name]
        env_cfg = self.get_env_config(env_name)

        if algorithm in ["PPO", "Q-PPO"]:
            if spec["action_type"] == "continuous":
                obs_dim = spec["obs_dim"]
                action_dim = spec["action_dim"]
                continuous = True
                action_bound = spec.get("action_bound", 1.0)
            else:
                obs_dim = spec["obs_dim"]
                action_dim = spec["n_actions"]
                continuous = False
                action_bound = 1.0

            ppo_config = self.config.ppo if algorithm == "PPO" else self.config.q_ppo

            if algorithm == "PPO":
                agent = PPOAgent(
                    obs_dim=obs_dim, action_dim=action_dim,
                    config=ppo_config, continuous=continuous,
                    action_bound=action_bound, seed=self.config.seed,
                )
            else:
                agent = QPPOAgent(
                    obs_dim=obs_dim, action_dim=action_dim,
                    config=ppo_config, quantum_config=self.config.quantum,
                    continuous=continuous, action_bound=action_bound,
                    seed=self.config.seed,
                )

        elif algorithm in ["DQN", "Q-DQN"]:
            obs_dim = spec["obs_dim"]
            n_actions = spec["n_actions"]
            dqn_config = self.config.dqn if algorithm == "DQN" else self.config.q_dqn

            if algorithm == "DQN":
                agent = DQNAgent(
                    obs_dim=obs_dim, n_actions=n_actions,
                    config=dqn_config, seed=self.config.seed,
                )
            else:
                agent = QDQNAgent(
                    obs_dim=obs_dim, n_actions=n_actions,
                    config=dqn_config, quantum_config=self.config.quantum,
                    seed=self.config.seed,
                )

        elif algorithm in ["SAC", "Q-SAC"]:
            obs_dim = spec["obs_dim"]
            action_dim = spec["action_dim"]
            action_bound = spec.get("action_bound", 1.0)
            sac_config = self.config.sac if algorithm == "SAC" else self.config.q_sac

            if algorithm == "SAC":
                agent = SACAgent(
                    obs_dim=obs_dim, action_dim=action_dim,
                    config=sac_config, action_bound=action_bound,
                    seed=self.config.seed,
                )
            else:
                agent = QSACAgent(
                    obs_dim=obs_dim, action_dim=action_dim,
                    config=sac_config, quantum_config=self.config.quantum,
                    action_bound=action_bound, seed=self.config.seed,
                )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        return agent

    def train(self, algorithm: str, env_name: str, env_specs: Dict) -> Dict[str, Any]:
        """
        Train an agent on an environment. Supports resumption from checkpoints.

        Returns a results dictionary.
        """
        key = get_experiment_key(algorithm, env_name)
        total_timesteps = self.config.get_experiment_budget(algorithm, env_name)
        eval_episodes = self.config.evaluation.n_episodes
        checkpoint_freq = self.config.training.checkpoint_freq
        log_interval = self.config.training.log_interval

        # Logger
        logger = ExperimentLogger(self.config.log_dir, key)

        # Check for existing checkpoint
        checkpoint_path = self.find_latest_checkpoint(algorithm, env_name)
        resume = False
        if checkpoint_path:
            try:
                ckpt_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                ckpt_steps = ckpt_data.get("total_steps", 0)
                if ckpt_steps > 0:
                    agent = self._load_agent(algorithm, checkpoint_path)
                    resume = True
                    start_step = agent.total_steps
                    print(f"  [RESUME] Found checkpoint: {checkpoint_path}")
                    print(f"  [RESUME] Resuming from step {start_step}")
                else:
                    print(f"  [NEW] Checkpoint has 0 steps. Creating fresh agent for {algorithm} on {env_name}")
                    agent = self.create_agent(algorithm, env_name, env_specs)
                    start_step = 0
            except Exception as e:
                print(f"  [WARN] Failed to load checkpoint {checkpoint_path} ({e}). Starting fresh.")
                agent = self.create_agent(algorithm, env_name, env_specs)
                start_step = 0
        else:
            print(f"  [NEW] Creating new agent for {algorithm} on {env_name}")
            agent = self.create_agent(algorithm, env_name, env_specs)
            start_step = 0

        logger.update(
            algorithm=algorithm,
            environment=env_name,
            status="running",
            resumed=resume,
            start_step=start_step,
            total_timesteps=total_timesteps,
            seed=self.config.seed,
            quantum_config=self.config.quantum.to_dict() if hasattr(self.config.quantum, 'to_dict') else {},
        )
        logger.save()

        # Create environment
        env = self.env_manager.create_and_seed(env_name, env_specs, seed=self.config.seed)

        # Training loop
        episode_reward = 0
        episode_length = 0
        obs, _ = env.reset(seed=self.config.seed)
        best_eval_reward = -float("inf")
        step = start_step
        max_steps_per_episode = env_specs[env_name]["max_episode_steps"]

        # Load best reward from checkpoint if resuming
        if resume and hasattr(agent, 'history') and agent.history.get("episode_rewards"):
            recent_rewards = agent.history["episode_rewards"][-100:]
            if recent_rewards:
                best_eval_reward = max(recent_rewards)

        print(f"  [TRAIN] Training {algorithm} on {env_name} ({total_timesteps} steps)")

        try:
            while step < total_timesteps:
                if self._interrupted:
                    raise KeyboardInterrupt("Training interrupted")

                # Select action and step environment based on algorithm family
                if algorithm in ["PPO", "Q-PPO"]:
                    action, log_prob, value = agent.select_action(obs)
                    if env_specs[env_name]["action_type"] == "continuous":
                        next_obs, reward, terminated, truncated, info = env.step(action if isinstance(action, np.ndarray) else np.array([action]))
                        done = terminated or truncated
                        agent.store_transition_continuous(obs, action, reward, done, log_prob, value)
                    else:
                        next_obs, reward, terminated, truncated, info = env.step(int(action) if not isinstance(action, (int, np.integer)) else action)
                        done = terminated or truncated
                        agent.store_transition(obs, action, reward, done, log_prob, value)
                elif algorithm in ["SAC", "Q-SAC"]:
                    action = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action if isinstance(action, np.ndarray) else np.array([action]))
                    done = terminated or truncated
                    agent.store_transition(obs, action, reward, next_obs, done)
                elif algorithm in ["DQN", "Q-DQN"]:
                    action = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(int(action) if not isinstance(action, (int, np.integer)) else action)
                    done = terminated or truncated
                    agent.store_transition(obs, action, reward, next_obs, done)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")

                episode_reward += float(reward)
                episode_length += 1
                agent.total_steps += 1
                step += 1

                # Update for on-policy algorithms at rollout boundaries
                if algorithm in ["PPO", "Q-PPO"]:
                    if step % agent.config.rollout_length == 0 and step > 0:
                        agent.update(next_obs)

                # Update for off-policy algorithms
                if algorithm in ["DQN", "Q-DQN", "SAC", "Q-SAC"]:
                    agent.update()

                obs = next_obs

                # Episode done
                if done:
                    agent.record_episode(float(episode_reward), episode_length)
                    episode_reward = 0
                    episode_length = 0
                    obs, _ = env.reset()

                # Log progress
                if step % log_interval == 0 and step > 0:
                    recent_rewards = agent.history["episode_rewards"][-100:]
                    avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                    print(f"    Step {step}/{total_timesteps} | "
                          f"Episodes: {agent.episode_count} | "
                          f"Avg reward (last 100): {avg_reward:.2f} | "
                          f"Time: {agent.total_training_time:.1f}s")

                    logger.update(
                        total_steps=step,
                        episodes=agent.episode_count,
                        avg_reward_recent=float(avg_reward),
                        training_time=agent.total_training_time,
                    )
                    logger.save()

                # Checkpointing
                if step % checkpoint_freq == 0 and step > 0:
                    agent.save_checkpoint(self.get_checkpoint_path(algorithm, env_name))
                    # Also save periodic checkpoint (keep limited)
                    periodic_path = self.get_periodic_checkpoint_path(algorithm, env_name, step)
                    agent.save_checkpoint(periodic_path)
                    # Clean old periodic checkpoints (keep last 3)
                    self._clean_periodic_checkpoints(algorithm, env_name, keep=3)

                    logger.update(checkpoint_step=step)
                    logger.save()

        except KeyboardInterrupt:
            print(f"  [INTERRUPT] Saving checkpoint at step {step}...")
            agent.save_checkpoint(self.get_checkpoint_path(algorithm, env_name))
            agent.training_step = step
            logger.finish(status="interrupted")
            env.close()
            raise

        # Final save
        agent.save_checkpoint(self.get_checkpoint_path(algorithm, env_name))
        agent.training_step = step

        # Evaluation
        print(f"  [EVAL] Evaluating {algorithm} on {env_name}...")
        eval_results = self._evaluate(agent, env_name, env_specs[env_name], env_specs, eval_episodes)

        # Save final checkpoint and best
        agent.save_checkpoint(self.get_checkpoint_path(algorithm, env_name))
        if eval_results["mean_return"] > best_eval_reward:
            agent.save_checkpoint(self.get_best_checkpoint_path(algorithm, env_name))
            best_eval_reward = eval_results["mean_return"]

        logger.update(
            status="completed",
            total_steps=step,
            episodes=agent.episode_count,
            training_time=agent.total_training_time,
            eval_results=eval_results,
            best_eval_return=float(best_eval_reward),
        )
        logger.finish(status="completed")

        env.close()

        # Compile full results
        agent_info = agent.get_info()
        is_optimized = key in [
            "Q-DQN_LunarLander-v3", "SAC_Pendulum-v1", "Q-SAC_Pendulum-v1",
            "SAC_BipedalWalker-v3", "Q-SAC_BipedalWalker-v3"
        ]
        regime = "optimized" if is_optimized else "original"
        results = {
            "algorithm": algorithm,
            "environment": env_name,
            "status": "completed",
            "configuration_regime": regime,
            "resumed": resume,
            "total_steps": step,
            "episodes": agent.episode_count,
            "training_time": agent.total_training_time,
            "eval_results": eval_results,
            "agent_info": agent_info,
            "history": agent.history,
            "config": {
                "seed": self.config.seed,
                "total_timesteps": total_timesteps,
                "configuration_regime": regime,
                "batch_size": getattr(agent.config, "batch_size", None),
                "train_freq": getattr(agent.config, "train_freq", None),
                "hidden_dim": getattr(agent.config, "hidden_dim", None),
            },
        }

        # Save results
        results_path = os.path.join(self.config.data_dir, f"{key}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"  [DONE] {algorithm} on {env_name}: "
              f"Mean return = {eval_results['mean_return']:.2f}, "
              f"Steps = {step}")

        return results

    def _evaluate(self, agent, env_name: str, env_spec: Dict, all_env_specs: Dict,
                  n_episodes: int = 20) -> Dict[str, Any]:
        """Evaluate a trained agent."""
        env = self.env_manager.create_and_seed(
            env_name, all_env_specs, seed=self.config.seed + 1000
        )

        returns = []
        lengths = []
        eval_start = time.time()

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=self.config.seed + 1000 + ep)
            ep_return = 0
            ep_length = 0
            done = False

            while not done:
                action_out = agent.select_action(obs, deterministic=True)
                if isinstance(action_out, tuple):
                    action = action_out[0]
                else:
                    action = action_out

                if agent.continuous if hasattr(agent, 'continuous') else env_spec["action_type"] == "continuous":
                    obs, reward, terminated, truncated, _ = env.step(action if isinstance(action, np.ndarray) else np.array([action]))
                else:
                    obs, reward, terminated, truncated, _ = env.step(int(action) if not isinstance(action, (int, np.integer)) else action)
                done = terminated or truncated
                ep_return += float(reward)
                ep_length += 1

            returns.append(ep_return)
            lengths.append(ep_length)

        eval_time = time.time() - eval_start
        env.close()

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "median_return": float(np.median(returns)),
            "mean_length": float(np.mean(lengths)),
            "eval_time": eval_time,
            "n_episodes": n_episodes,
            "episode_returns": [float(r) for r in returns],
            "episode_lengths": [int(l) for l in lengths],
        }

    def _load_agent(self, algorithm: str, checkpoint_path: str):
        """Load an agent from checkpoint."""
        if algorithm == "PPO":
            return PPOAgent.load_checkpoint(checkpoint_path, config=self.config.ppo)
        elif algorithm == "Q-PPO":
            return QPPOAgent.load_checkpoint(checkpoint_path, config=self.config.q_ppo,
                                              quantum_config=self.config.quantum)
        elif algorithm == "DQN":
            return DQNAgent.load_checkpoint(checkpoint_path, config=self.config.dqn)
        elif algorithm == "Q-DQN":
            return QDQNAgent.load_checkpoint(checkpoint_path, config=self.config.q_dqn,
                                               quantum_config=self.config.quantum)
        elif algorithm == "SAC":
            return SACAgent.load_checkpoint(checkpoint_path, config=self.config.sac)
        elif algorithm == "Q-SAC":
            return QSACAgent.load_checkpoint(checkpoint_path, config=self.config.q_sac,
                                              quantum_config=self.config.quantum)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _clean_periodic_checkpoints(self, algorithm: str, env_name: str, keep: int = 3):
        """Remove old periodic checkpoints, keeping only the most recent ones."""
        key = get_experiment_key(algorithm, env_name)
        pattern = f"{key}_step"
        files = []
        for f in os.listdir(self.config.checkpoint_dir):
            if f.startswith(pattern) and f.endswith(".pt"):
                try:
                    step_num = int(f.replace(pattern, "").replace(".pt", ""))
                    files.append((step_num, f))
                except ValueError:
                    continue

        files.sort(key=lambda x: x[0])
        if len(files) > keep:
            for _, fname in files[:-keep]:
                os.remove(os.path.join(self.config.checkpoint_dir, fname))
