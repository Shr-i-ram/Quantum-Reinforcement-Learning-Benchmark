"""
Hybrid Quantum SAC (Q-SAC) agent.

Uses a quantum circuit in the actor for policy representation.
The twin critics remain classical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque
import random
import time
import os
import math

from quantum_rl_benchmark.models.quantum.circuit import QuantumCircuitLayer, QuantumFeatureProjection, get_circuit_info
from quantum_rl_benchmark.config.settings import QuantumConfig, SACConfig


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPSILON = 1e-6


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
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class QuantumActor(nn.Module):
    """Quantum SAC actor: uses quantum circuit for policy representation."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_bound: float = 1.0, quantum_config: Optional[QuantumConfig] = None):
        super().__init__()
        self.action_bound = action_bound

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
        self.mean = nn.Linear(quantum_config.n_qubits, action_dim)
        self.log_std = nn.Linear(quantum_config.n_qubits, action_dim)

    def forward(self, obs):
        h = self.preprocess(obs)
        features = self.feature_proj(h)
        quantum_out = self.quantum(features)
        mean = self.mean(quantum_out)
        log_std = self.log_std(quantum_out)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        tanh_x = torch.tanh(x_t)
        action = tanh_x * self.action_bound
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_bound * (1 - tanh_x.pow(2)) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def get_quantum_params(self):
        return self.quantum.params


class Critic(nn.Module):
    """Classical critic for Q-SAC."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class QSACAgent:
    """Hybrid Quantum SAC agent for continuous action spaces."""

    def __init__(self, obs_dim: int, action_dim: int, config=None,
                 quantum_config: Optional[QuantumConfig] = None,
                 action_bound: float = 1.0, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.seed = seed

        set_seed(seed)
        random.seed(seed)

        if config is None:
            config = SACConfig()
        self.config = config
        if quantum_config is None:
            quantum_config = QuantumConfig()
        self.quantum_config = quantum_config

        self.gamma = config.gamma
        self.tau = config.tau
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.warmup_steps = config.warmup_steps
        self.train_freq = config.train_freq

        hidden = config.hidden_dim
        self.actor = QuantumActor(obs_dim, action_dim, hidden_dim=hidden,
                                   action_bound=action_bound, quantum_config=quantum_config)
        self.critic1 = Critic(obs_dim, action_dim, hidden_dim=hidden)
        self.critic2 = Critic(obs_dim, action_dim, hidden_dim=hidden)
        self.critic1_target = Critic(obs_dim, action_dim, hidden_dim=hidden)
        self.critic2_target = Critic(obs_dim, action_dim, hidden_dim=hidden)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.learning_rate)

        self.auto_alpha = config.auto_alpha
        if self.auto_alpha:
            self.target_entropy = config.target_entropy or -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
        else:
            self.log_alpha = torch.tensor(math.log(config.alpha))
            self.target_entropy = None

        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

        self.total_steps = 0
        self.episode_count = 0
        self.training_step = 0

        self.history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "actor_loss": [],
            "critic1_loss": [],
            "critic2_loss": [],
            "alpha": [],
            "mean_q_value": [],
            "quantum_gradient_norm": [],
            "quantum_gradient_mean": [],
            "quantum_gradient_var": [],
            "total_steps": [],
        }

        self.total_training_time = 0.0
        self.quantum_time = 0.0
        self.quantum_eval_count = 0

    @property
    def alpha(self):
        if self.auto_alpha:
            return self.log_alpha.exp().item()
        return self.config.alpha

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(obs_t)
                action = torch.tanh(mean) * self.action_bound
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.squeeze(0).numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, float(done))

    def update(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None
        if self.total_steps % self.train_freq != 0:
            return None

        start_time = time.time()
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        alpha = self.alpha

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * q_next

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_actions, log_probs, _ = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Track quantum gradients
        with torch.no_grad():
            q_params = self.actor.get_quantum_params()
            if q_params.grad is not None:
                grad_norm = q_params.grad.norm().item()
                grad_mean = q_params.grad.mean().item()
                grad_var = q_params.grad.var().item()
            else:
                grad_norm = 0.0
                grad_mean = 0.0
                grad_var = 0.0

        self.actor_optimizer.step()

        self.quantum_eval_count += states.shape[0]

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        update_time = time.time() - start_time
        self.total_training_time += update_time

        stats = {
            "actor_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "alpha": alpha,
            "mean_q_value": q_new.mean().item(),
            "quantum_gradient_norm": grad_norm,
            "quantum_gradient_mean": grad_mean,
            "quantum_gradient_var": grad_var,
        }

        for key in ["actor_loss", "critic1_loss", "critic2_loss", "alpha", "mean_q_value",
                     "quantum_gradient_norm", "quantum_gradient_mean", "quantum_gradient_var"]:
            self.history[key].append(stats[key])

        return stats

    def record_episode(self, reward: float, length: int):
        self.episode_count += 1
        self.history["episode_rewards"].append(float(reward))
        self.history["episode_lengths"].append(int(length))
        self.history["total_steps"].append(int(self.total_steps))

    def get_info(self) -> Dict[str, Any]:
        actor_params = sum(p.numel() for p in self.actor.parameters())
        critic_params = sum(p.numel() for p in self.critic1.parameters()) * 2
        quantum_info = self.actor.quantum.get_circuit_info()
        return {
            "algorithm": "Q-SAC",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "action_bound": self.action_bound,
            "hidden_dim": self.config.hidden_dim,
            "n_parameters": actor_params + critic_params,
            "n_quantum_parameters": quantum_info["n_params"],
            "n_classical_parameters": actor_params + critic_params - quantum_info["n_params"],
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
                "tau": self.tau,
                "batch_size": self.batch_size,
                "auto_alpha": self.auto_alpha,
            },
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "actor_state": self.actor.state_dict(),
            "critic1_state": self.critic1.state_dict(),
            "critic2_state": self.critic2.state_dict(),
            "critic1_target_state": self.critic1_target.state_dict(),
            "critic2_target_state": self.critic2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "training_step": self.training_step,
            "history": self.history,
            "total_training_time": self.total_training_time,
            "quantum_eval_count": self.quantum_eval_count,
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "action_bound": self.action_bound,
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
            "algorithm": "Q-SAC",
        }
        if self.auto_alpha:
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, config=None, quantum_config=None) -> "QSACAgent":
        checkpoint = torch.load(path, weights_only=False)
        info = checkpoint["config"]
        q_cfg_dict = checkpoint.get("quantum_config", {})
        q_config = quantum_config or (QuantumConfig(**q_cfg_dict) if q_cfg_dict else quantum_config)
        agent = cls(
            obs_dim=info["obs_dim"],
            action_dim=info["action_dim"],
            config=config,
            quantum_config=q_config,
            action_bound=info.get("action_bound", 1.0),
            seed=info["seed"],
        )
        agent.actor.load_state_dict(checkpoint["actor_state"])
        agent.critic1.load_state_dict(checkpoint["critic1_state"])
        agent.critic2.load_state_dict(checkpoint["critic2_state"])
        agent.critic1_target.load_state_dict(checkpoint["critic1_target_state"])
        agent.critic2_target.load_state_dict(checkpoint["critic2_target_state"])
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        agent.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
        agent.log_alpha = checkpoint["log_alpha"]
        if "alpha_optimizer" in checkpoint and agent.auto_alpha:
            agent.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        agent.total_steps = checkpoint["total_steps"]
        agent.episode_count = checkpoint["episode_count"]
        agent.training_step = checkpoint.get("training_step", 0)
        agent.history = checkpoint["history"]
        agent.total_training_time = checkpoint.get("total_training_time", 0.0)
        agent.quantum_eval_count = checkpoint.get("quantum_eval_count", 0)
        return agent
