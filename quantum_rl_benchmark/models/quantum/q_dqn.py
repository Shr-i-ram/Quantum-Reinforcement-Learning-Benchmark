"""
Hybrid Quantum DQN (Q-DQN) agent.

Replaces the Q-network with a quantum circuit for action-value estimation.
Uses the same replay buffer, target network, and epsilon-greedy as classical DQN.
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

from quantum_rl_benchmark.models.quantum.circuit import QuantumCircuitLayer, QuantumFeatureProjection, get_circuit_info
from quantum_rl_benchmark.config.settings import QuantumConfig, DQNConfig


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
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


class QuantumQNetwork(nn.Module):
    """Quantum Q-network: state -> quantum circuit -> Q-values for each action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128,
                 quantum_config: Optional[QuantumConfig] = None):
        super().__init__()
        self.n_actions = n_actions

        if quantum_config is None:
            quantum_config = QuantumConfig()

        self.preprocess = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_proj = QuantumFeatureProjection(hidden_dim, output_dim=4, hidden_dim=16)
        self.quantum = QuantumCircuitLayer(
            n_qubits=quantum_config.n_qubits,
            depth=quantum_config.circuit_depth,
            init_scale=quantum_config.init_scale,
            shots=quantum_config.shots,
            differentiation_method=quantum_config.differentiation_method,
        )
        self.output = nn.Linear(quantum_config.n_qubits, n_actions)

    def forward(self, x):
        h = self.preprocess(x)
        features = self.feature_proj(h)
        quantum_out = self.quantum(features)
        return self.output(quantum_out)

    def get_quantum_params(self):
        return self.quantum.params


class QDQNAgent:
    """Hybrid Quantum DQN agent for discrete action spaces."""

    def __init__(self, obs_dim: int, n_actions: int, config=None,
                 quantum_config: Optional[QuantumConfig] = None, seed: int = 42):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.seed = seed

        set_seed(seed)
        random.seed(seed)

        if config is None:
            config = DQNConfig()
        self.config = config
        if quantum_config is None:
            quantum_config = QuantumConfig()
        self.quantum_config = quantum_config

        self.gamma = config.gamma
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_steps = config.epsilon_decay_steps
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.train_freq = config.train_freq
        self.learning_rate = config.learning_rate

        self.q_network = QuantumQNetwork(obs_dim, n_actions, hidden_dim=config.hidden_dim,
                                          quantum_config=quantum_config)
        self.target_network = QuantumQNetwork(obs_dim, n_actions, hidden_dim=config.hidden_dim,
                                               quantum_config=quantum_config)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)
        self.epsilon = self.epsilon_start

        self.total_steps = 0
        self.episode_count = 0
        self.update_count = 0
        self.training_step = 0

        self.history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "q_loss": [],
            "mean_q_value": [],
            "td_error": [],
            "epsilon": [],
            "quantum_gradient_norm": [],
            "quantum_gradient_mean": [],
            "quantum_gradient_var": [],
            "total_steps": [],
        }

        self.total_training_time = 0.0
        self.quantum_time = 0.0
        self.quantum_eval_count = 0

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(obs_t)
        return q_values.argmax(dim=-1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, float(done))

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        if self.total_steps % self.train_freq != 0:
            return None

        start_time = time.time()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        td_errors = q_values - target_q
        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # Track quantum gradients
        with torch.no_grad():
            q_params = self.q_network.get_quantum_params()
            if q_params.grad is not None:
                grad_norm = q_params.grad.norm().item()
                grad_mean = q_params.grad.mean().item()
                grad_var = q_params.grad.var().item()
            else:
                grad_norm = 0.0
                grad_mean = 0.0
                grad_var = 0.0

        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        self.quantum_eval_count += states.shape[0] * 2  # forward + target

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

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
            "quantum_gradient_norm": grad_norm,
            "quantum_gradient_mean": grad_mean,
            "quantum_gradient_var": grad_var,
        }

        for key in ["q_loss", "mean_q_value", "td_error", "epsilon",
                     "quantum_gradient_norm", "quantum_gradient_mean", "quantum_gradient_var"]:
            self.history[key].append(stats[key])

        return stats

    def record_episode(self, reward: float, length: int):
        self.episode_count += 1
        self.history["episode_rewards"].append(float(reward))
        self.history["episode_lengths"].append(int(length))
        self.history["total_steps"].append(int(self.total_steps))

    def get_info(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.q_network.parameters())
        quantum_info = self.q_network.quantum.get_circuit_info()
        return {
            "algorithm": "Q-DQN",
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "hidden_dim": self.config.hidden_dim,
            "n_parameters": n_params,
            "n_quantum_parameters": quantum_info["n_params"],
            "n_classical_parameters": n_params - quantum_info["n_params"],
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "total_training_time": self.total_training_time,
            "quantum_time": self.quantum_time,
            "is_quantum": True,
            "quantum_info": quantum_info,
            "quantum_evaluations": self.quantum_eval_count,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end,
                "batch_size": self.batch_size,
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
            "quantum_eval_count": self.quantum_eval_count,
            "config": {
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "hidden_dim": self.config.hidden_dim,
                "seed": self.seed,
            },
            "quantum_config": {
                "n_qubits": self.quantum_config.n_qubits,
                "circuit_depth": self.quantum_config.circuit_depth,
                "init_scale": self.quantum_config.init_scale,
                "shots": self.quantum_config.shots,
                "differentiation_method": self.quantum_config.differentiation_method,
            },
            "algorithm": "Q-DQN",
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, config=None, quantum_config=None) -> "QDQNAgent":
        checkpoint = torch.load(path, weights_only=False)
        info = checkpoint["config"]
        q_cfg_dict = checkpoint.get("quantum_config", {})
        q_config = quantum_config or (QuantumConfig(**q_cfg_dict) if q_cfg_dict else quantum_config)
        agent = cls(
            obs_dim=info["obs_dim"],
            n_actions=info["n_actions"],
            config=config,
            quantum_config=q_config,
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
        agent.quantum_eval_count = checkpoint.get("quantum_eval_count", 0)
        return agent
