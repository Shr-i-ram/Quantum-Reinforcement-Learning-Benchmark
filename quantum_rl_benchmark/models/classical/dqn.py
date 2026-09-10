"""
Classical DQN (Deep Q-Network) agent.

Includes experience replay, target network, epsilon-greedy exploration,
and standard DQN training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
import random
import time
import os


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network for DQN."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """
    Classical DQN agent for discrete action spaces.

    Args:
        obs_dim: Observation dimensionality
        n_actions: Number of discrete actions
        config: DQNConfig or dict with hyperparameters
        seed: Random seed
    """

    def __init__(self, obs_dim: int, n_actions: int, config=None, seed: int = 42):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.seed = seed

        set_seed(seed)
        random.seed(seed)

        if config is None:
            from quantum_rl_benchmark.config.settings import DQNConfig
            config = DQNConfig()
        self.config = config

        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_steps = config.epsilon_decay_steps
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.train_freq = config.train_freq
        self.learning_rate = config.learning_rate

        # Networks
        self.q_network = QNetwork(obs_dim, n_actions, hidden_dim=config.hidden_dim)
        self.target_network = QNetwork(obs_dim, n_actions, hidden_dim=config.hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        # Epsilon
        self.epsilon = self.epsilon_start

        # Training state
        self.total_steps = 0
        self.episode_count = 0
        self.update_count = 0
        self.training_step = 0

        # History
        self.history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "q_loss": [],
            "mean_q_value": [],
            "td_error": [],
            "epsilon": [],
            "total_steps": [],
        }

        # Timing
        self.total_training_time = 0.0
        self.quantum_time = 0.0

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_t)
        return q_values.argmax(dim=-1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, float(done))

    def update(self) -> Optional[Dict[str, float]]:
        """Perform DQN update if enough samples."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        if self.total_steps % self.train_freq != 0:
            return None

        start_time = time.time()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        td_errors = q_values - target_q
        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Update epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end)
            * self.total_steps / self.epsilon_decay_steps,
        )

        update_time = time.time() - start_time
        self.total_training_time += update_time

        stats = {
            "q_loss": loss.item(),
            "mean_q_value": q_values.mean().item(),
            "td_error": td_errors.abs().mean().item(),
            "epsilon": self.epsilon,
        }

        self.history["q_loss"].append(stats["q_loss"])
        self.history["mean_q_value"].append(stats["mean_q_value"])
        self.history["td_error"].append(stats["td_error"])
        self.history["epsilon"].append(stats["epsilon"])

        return stats

    def record_episode(self, reward: float, length: int):
        self.episode_count += 1
        self.history["episode_rewards"].append(float(reward))
        self.history["episode_lengths"].append(int(length))
        self.history["total_steps"].append(int(self.total_steps))

    def get_info(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.q_network.parameters())
        return {
            "algorithm": "DQN",
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_dim": self.config.hidden_dim,
            "n_parameters": n_params,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "total_training_time": self.total_training_time,
            "quantum_time": 0.0,
            "is_quantum": False,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "batch_size": self.batch_size,
                "replay_buffer_size": self.config.replay_buffer_size,
                "target_update_freq": self.target_update_freq,
            },
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "update_count": self.update_count,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "history": self.history,
            "total_training_time": self.total_training_time,
            "config": {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "hidden_dim": self.config.hidden_dim,
                "seed": self.seed,
            },
            "algorithm": "DQN",
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, config=None) -> "DQNAgent":
        checkpoint = torch.load(path, weights_only=False)
        info = checkpoint["config"]
        agent = cls(
            obs_dim=info["obs_dim"],
            n_actions=info["n_actions"],
            config=config,
            seed=info["seed"],
        )
        agent.q_network.load_state_dict(checkpoint["q_network_state"])
        agent.target_network.load_state_dict(checkpoint["target_network_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
        agent.total_steps = checkpoint["total_steps"]
        agent.episode_count = checkpoint["episode_count"]
        agent.update_count = checkpoint["update_count"]
        agent.training_step = checkpoint.get("training_step", 0)
        agent.epsilon = checkpoint["epsilon"]
        agent.history = checkpoint["history"]
        agent.total_training_time = checkpoint.get("total_training_time", 0.0)
        return agent
