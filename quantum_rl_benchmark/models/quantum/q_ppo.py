"""
Hybrid Quantum PPO (Q-PPO) agent.

Replaces the policy network's core with a quantum circuit layer.
The value function remains classical.

Architecture:
    Observation -> Classical preprocessing -> Feature projection to 4D
    -> 4-qubit quantum circuit -> Expectation values -> Policy logits/mean
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple
import time
import os
import json

from quantum_rl_benchmark.models.quantum.circuit import QuantumCircuitLayer, QuantumFeatureProjection, get_circuit_info
from quantum_rl_benchmark.config.settings import QuantumConfig, PPOConfig


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class QPPOBuffer:
    """Rollout buffer for Q-PPO."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.quantum_gradients = []  # track quantum gradient norms

    def add(self, obs, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_gae(self, next_value, gamma, lam):
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

    def get_batches(self, batch_size):
        n = len(self.observations)
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            obs_batch = torch.stack([self.observations[i] for i in batch_idx])
            action_batch = torch.stack([self.actions[i] for i in batch_idx])
            old_log_probs = torch.stack([self.log_probs[i] for i in batch_idx])
            advantages = torch.tensor([self.advantages[i] for i in batch_idx], dtype=torch.float32)
            returns = torch.tensor([self.returns[i] for i in batch_idx], dtype=torch.float32)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            yield {
                "obs": obs_batch, "actions": action_batch,
                "old_log_probs": old_log_probs, "advantages": advantages,
                "returns": returns,
            }

    def clear(self):
        self.__init__()


class QPPOPolicy(nn.Module):
    """
    Q-PPO policy: quantum circuit for policy representation, classical value head.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 continuous: bool = False, action_bound: float = 1.0,
                 quantum_config: Optional[QuantumConfig] = None):
        super().__init__()
        self.continuous = continuous
        self.action_bound = action_bound
        self.obs_dim = obs_dim

        if quantum_config is None:
            quantum_config = QuantumConfig()

        # Classical preprocessing
        self.preprocess = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Feature projection to 4D
        self.feature_proj = QuantumFeatureProjection(hidden_dim, output_dim=4, hidden_dim=16)

        # Quantum circuit
        self.quantum = QuantumCircuitLayer(
            n_qubits=quantum_config.n_qubits,
            depth=quantum_config.circuit_depth,
            init_scale=quantum_config.init_scale,
            shots=quantum_config.shots,
            differentiation_method=quantum_config.differentiation_method,
        )

        # Classical output from quantum measurements (4 expectation values)
        if continuous:
            self.actor_mean = nn.Linear(quantum_config.n_qubits, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(quantum_config.n_qubits, action_dim)

        # Classical value network (separate from quantum path)
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        # Classical value
        value = self.value_net(obs)

        # Quantum policy path
        h = self.preprocess(obs)
        features = self.feature_proj(h)
        quantum_out = self.quantum(features)

        if self.continuous:
            mean = self.actor_mean(quantum_out) * self.action_bound
            log_std = self.actor_log_std.expand_as(mean)
            return mean, log_std, value.squeeze(-1)
        else:
            logits = self.actor(quantum_out)
            return logits, value.squeeze(-1)

    def get_action_and_value(self, obs):
        if self.continuous:
            mean, log_std, value = self.forward(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action, log_prob, value
        else:
            logits, value = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action), value

    def evaluate(self, obs, actions):
        if self.continuous:
            mean, log_std, value = self.forward(obs)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return log_prob, value, entropy
        else:
            logits, value = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions), value, dist.entropy()

    def get_value(self, obs):
        return self.value_net(obs).squeeze(-1)

    def get_quantum_params(self):
        return self.quantum.params

    def get_quantum_info(self):
        return self.quantum.get_circuit_info()


class QPPOAgent:
    """
    Quantum PPO agent with hybrid quantum-classical policy.
    """

    def __init__(self, obs_dim: int, action_dim: int, config=None,
                 quantum_config: Optional[QuantumConfig] = None,
                 continuous: bool = False, action_bound: float = 1.0, seed: int = 42):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.action_bound = action_bound
        self.seed = seed

        set_seed(seed)

        if config is None:
            config = PPOConfig()
        self.config = config

        if quantum_config is None:
            quantum_config = QuantumConfig()
        self.quantum_config = quantum_config

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

        self.policy = QPPOPolicy(
            obs_dim, action_dim, hidden_dim=self.hidden_dim,
            continuous=continuous, action_bound=action_bound,
            quantum_config=quantum_config,
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.buffer = QPPOBuffer()

        self.total_steps = 0
        self.episode_count = 0
        self.training_step = 0

        self.history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "clip_fraction": [],
            "approx_kl": [],
            "quantum_gradient_norm": [],
            "quantum_gradient_mean": [],
            "quantum_gradient_var": [],
            "total_steps": [],
        }

        self.total_training_time = 0.0
        self.quantum_time = 0.0
        self.quantum_eval_count = 0

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
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
        self.buffer.add(
            torch.FloatTensor(obs),
            torch.LongTensor([action]) if not self.continuous else torch.FloatTensor(action),
            reward, done,
            torch.FloatTensor([log_prob]),
            torch.FloatTensor([value]),
        )

    def store_transition_continuous(self, obs, action, reward, done, log_prob, value):
        self.buffer.add(
            torch.FloatTensor(obs),
            torch.FloatTensor(action),
            reward, done,
            torch.FloatTensor([log_prob]),
            torch.FloatTensor([value]),
        )

    def update(self, next_obs: np.ndarray) -> Dict[str, float]:
        start_time = time.time()

        next_value = self.policy.get_value(
            torch.FloatTensor(next_obs).unsqueeze(0)
        ).item()
        self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_frac = 0
        total_kl = 0
        total_q_grad_norm = 0
        total_q_grad_mean = 0
        total_q_grad_var = 0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                new_log_probs, values, entropy = self.policy.evaluate(obs, actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()

                # Track quantum gradients
                with torch.no_grad():
                    q_params = self.policy.get_quantum_params()
                    if q_params.grad is not None:
                        grad_norm = q_params.grad.norm().item()
                        grad_mean = q_params.grad.mean().item()
                        grad_var = q_params.grad.var().item()
                    else:
                        grad_norm = 0.0
                        grad_mean = 0.0
                        grad_var = 0.0

                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.quantum_eval_count += obs.shape[0]

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean().item()
                    clip_frac = (torch.abs(ratio - 1) > self.clip_epsilon).float().mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_clip_frac += clip_frac
                total_kl += approx_kl
                total_q_grad_norm += grad_norm
                total_q_grad_mean += grad_mean
                total_q_grad_var += grad_var
                n_updates += 1

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
            "quantum_gradient_norm": total_q_grad_norm / n_updates,
            "quantum_gradient_mean": total_q_grad_mean / n_updates,
            "quantum_gradient_var": total_q_grad_var / n_updates,
        }

        for key in ["policy_loss", "value_loss", "entropy", "clip_fraction", "approx_kl",
                     "quantum_gradient_norm", "quantum_gradient_mean", "quantum_gradient_var"]:
            self.history[key].append(stats[key])

        return stats

    def record_episode(self, reward: float, length: int):
        self.episode_count += 1
        self.history["episode_rewards"].append(float(reward))
        self.history["episode_lengths"].append(int(length))
        self.history["total_steps"].append(int(self.total_steps))

    def get_info(self) -> Dict[str, Any]:
        n_params = sum(p.numel() for p in self.policy.parameters())
        quantum_info = self.policy.get_quantum_info()
        return {
            "algorithm": "Q-PPO",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "continuous": self.continuous,
            "hidden_dim": self.hidden_dim,
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
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
            },
        }

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "training_step": self.training_step,
            "history": self.history,
            "total_training_time": self.total_training_time,
            "quantum_time": self.quantum_time,
            "quantum_eval_count": self.quantum_eval_count,
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "continuous": self.continuous,
                "action_bound": self.action_bound,
                "hidden_dim": self.hidden_dim,
                "seed": self.seed,
            },
            "quantum_config": {
                "n_qubits": self.quantum_config.n_qubits,
                "circuit_depth": self.quantum_config.circuit_depth,
                "init_scale": self.quantum_config.init_scale,
                "shots": self.quantum_config.shots,
                "differentiation_method": self.quantum_config.differentiation_method,
            },
            "algorithm": "Q-PPO",
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, config=None, quantum_config=None) -> "QPPOAgent":
        checkpoint = torch.load(path, weights_only=False)
        info = checkpoint["config"]

        q_cfg_dict = checkpoint.get("quantum_config", {})
        q_config = quantum_config or QuantumConfig(**q_cfg_dict) if q_cfg_dict else quantum_config

        agent = cls(
            obs_dim=info["obs_dim"],
            action_dim=info["action_dim"],
            config=config,
            quantum_config=q_config,
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
        agent.quantum_time = checkpoint.get("quantum_time", 0.0)
        agent.quantum_eval_count = checkpoint.get("quantum_eval_count", 0)
        return agent
