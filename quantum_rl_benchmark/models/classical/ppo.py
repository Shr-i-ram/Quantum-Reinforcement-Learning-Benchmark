"""
Classical PPO (Proximal Policy Optimization) agent.

Supports both discrete and continuous action spaces.
Implements standard PPO with GAE, clipping, and separate policy/value networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import time
import os
import json


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class RolloutBuffer:
    """Experience buffer for PPO rollouts."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []

    def add(self, obs, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, next_value: float, gamma: float, lam: float):
        """Compute Generalized Advantage Estimation."""
        values = self.values + [next_value]
        gae = 0
        self.advantages = [0] * len(self.rewards)
        self.returns = [0] * len(self.rewards)

        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t])
                     - values[t])
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + values[t]

    def get_batches(self, batch_size: int):
        """Generate mini-batches for PPO update."""
        n = len(self.observations)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            obs_batch = torch.stack([self.observations[i] for i in batch_idx])
            action_batch = torch.stack([self.actions[i] for i in batch_idx])
            old_log_probs = torch.stack([self.log_probs[i] for i in batch_idx])
            advantages = torch.tensor(
                [self.advantages[i] for i in batch_idx], dtype=torch.float32
            )
            returns = torch.tensor(
                [self.returns[i] for i in batch_idx], dtype=torch.float32
            )

            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            yield {
                "obs": obs_batch,
                "actions": action_batch,
                "old_log_probs": old_log_probs,
                "advantages": advantages,
                "returns": returns,
            }

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.observations)


class PolicyNetwork(nn.Module):
    """Combined policy-value network for PPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 continuous: bool = False, action_bound: float = 1.0):
        super().__init__()
        self.continuous = continuous
        self.action_bound = action_bound

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        shared_out = self.shared(obs)

        if self.continuous:
            mean = self.actor_mean(shared_out) * self.action_bound
            log_std = self.actor_log_std.expand_as(mean)
            return mean, log_std, self.critic(shared_out)
        else:
            logits = self.actor(shared_out)
            return logits, self.critic(shared_out)

    def get_action_and_value(self, obs):
        if self.continuous:
            mean, log_std, value = self.forward(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value.squeeze(-1)
        else:
            logits, value = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, obs, actions):
        if self.continuous:
            mean, log_std, value = self.forward(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, value.squeeze(-1), entropy
        else:
            logits, value = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions), value.squeeze(-1), dist.entropy()

    def get_value(self, obs):
        if self.continuous:
            _, _, value = self.forward(obs)
        else:
            _, value = self.forward(obs)
        return value.squeeze(-1)


class PPOAgent:
    """
    Classical PPO agent for discrete and continuous action spaces.

    Args:
        obs_dim: Observation space dimensionality
        action_dim: Action space dimensionality
        config: PPOConfig or dict with hyperparameters
        continuous: Whether the action space is continuous
        action_bound: Action bound for continuous spaces
        seed: Random seed
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config=None,
        continuous: bool = False,
        action_bound: float = 1.0,
        seed: int = 42,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.action_bound = action_bound
        self.seed = seed

        set_seed(seed)

        # Config
        if config is None:
            from quantum_rl_benchmark.config.settings import PPOConfig
            config = PPOConfig()
        self.config = config

        self.hidden_dim = config.hidden_dim
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_epsilon = config.clip_epsilon
        self.entropy_coeff = config.entropy_coeff
        self.value_loss_coeff = config.value_loss_coeff
        self.max_grad_norm = config.max_grad_norm
        self.n_epochs = config.n_epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate

        # Network
        self.policy = PolicyNetwork(
            obs_dim, action_dim, hidden_dim=self.hidden_dim,
            continuous=continuous, action_bound=action_bound,
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Buffer
        self.buffer = RolloutBuffer()

        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.training_step = 0

        # Training history
        self.history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "approx_kl": [],
            "total_steps": [],
        }

        # Timing
        self.total_training_time = 0.0
        self.quantum_time = 0.0  # 0 for classical

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple:
        """Select action given observation."""
        obs_t = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            if self.continuous:
                mean, log_std, value = self.policy(obs_t)
                if deterministic:
                    action = mean
                    log_prob = torch.tensor(0.0)
                else:
                    std = log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                return action.squeeze(0).numpy(), log_prob.item(), value.item()
            else:
                logits, value = self.policy(obs_t)
                dist = torch.distributions.Categorical(logits=logits)
                if deterministic:
                    action = logits.argmax(dim=-1)
                    log_prob = dist.log_prob(action)
                else:
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                return action.item(), log_prob.item(), value.item()

    def store_transition(self, obs, action, reward, done, log_prob, value):
        """Store a transition in the rollout buffer."""
        self.buffer.add(
            torch.FloatTensor(obs),
            torch.LongTensor([action]) if not self.continuous else torch.FloatTensor(action),
            reward, done,
            torch.FloatTensor([log_prob]),
            torch.FloatTensor([value]),
        )

    def store_transition_continuous(self, obs, action, reward, done, log_prob, value):
        """Store a transition for continuous action spaces."""
        self.buffer.add(
            torch.FloatTensor(obs),
            torch.FloatTensor(action),
            reward, done,
            torch.FloatTensor([log_prob]),
            torch.FloatTensor([value]),
        )

    def update(self, next_obs: np.ndarray) -> Dict[str, float]:
        """Perform PPO update after collecting a rollout."""
        start_time = time.time()

        # Compute GAE
        next_value = self.policy.get_value(
            torch.FloatTensor(next_obs).unsqueeze(0)
        ).item()
        self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)

        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_frac = 0
        total_kl = 0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Evaluate current policy
                new_log_probs, values, entropy = self.policy.evaluate(obs, actions)

                # Policy loss (clipped)
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Stats
                with torch.no_grad():
                    approx_kl = (
                        (old_log_probs - new_log_probs).mean().item()
                    )
                    clip_frac = (
                        (torch.abs(ratio - 1) > self.clip_epsilon).float().mean().item()
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_clip_frac += clip_frac
                total_kl += approx_kl
                n_updates += 1

        # Clear buffer
        self.buffer.clear()

        update_time = time.time() - start_time
        self.total_training_time += update_time

        n_updates = max(n_updates, 1)
        stats = {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "clip_fraction": total_clip_frac / n_updates,
            "approx_kl": total_kl / n_updates,
        }

        self.history["policy_loss"].append(stats["policy_loss"])
        self.history["value_loss"].append(stats["value_loss"])
        self.history["entropy"].append(stats["entropy"])
        self.history["clip_fraction"].append(stats["clip_fraction"])
        self.history["approx_kl"].append(stats["approx_kl"])

        return stats

    def record_episode(self, reward: float, length: int):
        """Record a completed episode."""
        self.episode_count += 1
        self.history["episode_rewards"].append(float(reward))
        self.history["episode_lengths"].append(int(length))
        self.history["total_steps"].append(int(self.total_steps))

    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        n_params = sum(p.numel() for p in self.policy.parameters())
        return {
            "algorithm": "PPO",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "continuous": self.continuous,
            "hidden_dim": self.hidden_dim,
            "n_parameters": n_params,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "total_training_time": self.total_training_time,
            "quantum_time": 0.0,
            "is_quantum": False,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "entropy_coeff": self.entropy_coeff,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
            },
        }

    def save_checkpoint(self, path: str):
        """Save full checkpoint for resumability."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "training_step": self.training_step,
            "history": self.history,
            "total_training_time": self.total_training_time,
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "continuous": self.continuous,
                "action_bound": self.action_bound,
                "hidden_dim": self.hidden_dim,
                "seed": self.seed,
            },
            "algorithm": "PPO",
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, config=None) -> "PPOAgent":
        """Load from checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        info = checkpoint["config"]

        agent = cls(
            obs_dim=info["obs_dim"],
            action_dim=info["action_dim"],
            config=config,
            continuous=info["continuous"],
            action_bound=info.get("action_bound", 1.0),
            seed=info["seed"],
        )
        agent.policy.load_state_dict(checkpoint["model_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.total_steps = checkpoint["total_steps"]
        agent.episode_count = checkpoint["episode_count"]
        agent.training_step = checkpoint.get("training_step", 0)
        agent.history = checkpoint["history"]
        agent.total_training_time = checkpoint.get("total_training_time", 0.0)
        return agent
