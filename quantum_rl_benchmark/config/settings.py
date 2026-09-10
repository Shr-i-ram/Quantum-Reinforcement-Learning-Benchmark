"""
Central configuration for the Quantum RL Benchmark.

All hyperparameters, environment definitions, algorithm settings,
and experiment parameters are defined here.
"""

import os
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QuantumConfig:
    """Quantum circuit configuration."""
    n_qubits: int = 4
    circuit_depth: int = 2
    encoding: str = "angle"  # angle encoding
    init_scale: float = 0.1
    shots: Optional[int] = None  # analytic mode
    differentiation_method: str = "adjoint"
    device_name: str = "lightning.qubit"
    max_qubits: int = 4  # hard cap


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    rollout_length: int = 2048
    hidden_dim: int = 64


@dataclass
class DQNConfig:
    """DQN algorithm hyperparameters."""
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10000
    batch_size: int = 64
    replay_buffer_size: int = 100000
    target_update_freq: int = 1000
    hidden_dim: int = 128
    train_freq: int = 1


@dataclass
class SACConfig:
    """SAC algorithm hyperparameters."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: Optional[float] = None  # auto-set per env
    batch_size: int = 256
    replay_buffer_size: int = 1000000
    hidden_dim: int = 256
    warmup_steps: int = 1000
    train_freq: int = 1


@dataclass
class TrainingConfig:
    """Training budget and checkpoint configuration."""
    total_timesteps: int = 200_000
    eval_freq: int = 5000
    eval_episodes: int = 10
    checkpoint_freq: int = 10000
    max_checkpoint_count: int = 5  # keep last N checkpoints
    log_interval: int = 1000
    save_best: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    n_episodes: int = 20
    max_episode_steps: int = 1000
    record_video: bool = False


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    # Seed
    seed: int = 42

    # Directories
    base_dir: str = ""
    checkpoint_dir: str = ""
    log_dir: str = ""
    figure_dir: str = ""
    data_dir: str = ""

    # Algorithm configs
    ppo: PPOConfig = field(default_factory=PPOConfig)
    q_ppo: PPOConfig = field(default_factory=PPOConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    q_dqn: DQNConfig = field(default_factory=DQNConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    q_sac: SACConfig = field(default_factory=SACConfig)

    # Quantum config
    quantum: QuantumConfig = field(default_factory=QuantumConfig)

    # Training config
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Eval config
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # Environment configs - per environment overrides
    env_configs: dict = field(default_factory=dict)

    # Performance thresholds for sample efficiency
    # (env_name -> threshold for "solved")
    performance_thresholds: dict = field(default_factory=lambda: {
        "CartPole-v1": 475.0,
        "MountainCar-v0": -110.0,
        "LunarLander-v3": 200.0,
        "Pendulum-v1": -200.0,
        "BipedalWalker-v3": 300.0,
    })

    def __post_init__(self):
        """Set up directory paths based on base_dir."""
        if not self.base_dir:
            self.base_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results"
            )
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join(self.base_dir, "checkpoints")
        if not self.log_dir:
            self.log_dir = os.path.join(self.base_dir, "logs")
        if not self.figure_dir:
            self.figure_dir = os.path.join(self.base_dir, "figures")
        if not self.data_dir:
            self.data_dir = os.path.join(self.base_dir, "data")
        # Create directories
        for d in [self.checkpoint_dir, self.log_dir, self.figure_dir, self.data_dir]:
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return dataclasses.asdict(self)

    def save(self, path: str):
        """Save configuration to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# Environment definitions with complexity characteristics
ENVIRONMENTS = {
    "CartPole-v1": {
        "gymnasium_id": "CartPole-v1",
        "complexity_level": 1,
        "action_type": "discrete",
        "obs_dim": 4,
        "n_actions": 2,
        "max_episode_steps": 500,
        "reward_range": (0, 500),
        "complexity_chars": {
            "obs_dim": 4,
            "act_dim": 1,
            "discrete": True,
            "sparse_reward": False,
            "horizon": 500,
            "dynamics_difficulty": "low",
            "exploration_difficulty": "low",
            "optimization_difficulty": "low",
        },
    },
    "MountainCar-v0": {
        "gymnasium_id": "MountainCar-v0",
        "complexity_level": 2,
        "action_type": "discrete",
        "obs_dim": 2,
        "n_actions": 3,
        "max_episode_steps": 200,
        "reward_range": (-1, 0),
        "complexity_chars": {
            "obs_dim": 2,
            "act_dim": 1,
            "discrete": True,
            "sparse_reward": True,
            "horizon": 200,
            "dynamics_difficulty": "low",
            "exploration_difficulty": "high",
            "optimization_difficulty": "medium",
        },
    },
    "LunarLander-v3": {
        "gymnasium_id": "LunarLander-v3",
        "complexity_level": 3,
        "action_type": "discrete",
        "obs_dim": 8,
        "n_actions": 4,
        "max_episode_steps": 1000,
        "reward_range": (-200, 300),
        "complexity_chars": {
            "obs_dim": 8,
            "act_dim": 1,
            "discrete": True,
            "sparse_reward": False,
            "horizon": 1000,
            "dynamics_difficulty": "medium",
            "exploration_difficulty": "medium",
            "optimization_difficulty": "medium",
        },
    },
    "Pendulum-v1": {
        "gymnasium_id": "Pendulum-v1",
        "complexity_level": 4,
        "action_type": "continuous",
        "obs_dim": 3,
        "action_dim": 1,
        "action_bound": 2.0,
        "max_episode_steps": 200,
        "reward_range": (-1600, 0),
        "complexity_chars": {
            "obs_dim": 3,
            "act_dim": 1,
            "discrete": False,
            "sparse_reward": False,
            "horizon": 200,
            "dynamics_difficulty": "medium",
            "exploration_difficulty": "medium",
            "optimization_difficulty": "medium",
        },
    },
    "BipedalWalker-v3": {
        "gymnasium_id": "BipedalWalker-v3",
        "complexity_level": 5,
        "action_type": "continuous",
        "obs_dim": 24,
        "action_dim": 4,
        "action_bound": 1.0,
        "max_episode_steps": 2000,
        "reward_range": (-100, 300),
        "complexity_chars": {
            "obs_dim": 24,
            "act_dim": 4,
            "discrete": False,
            "sparse_reward": False,
            "horizon": 2000,
            "dynamics_difficulty": "high",
            "exploration_difficulty": "high",
            "optimization_difficulty": "high",
        },
    },
}

# Experimental matrix: which algorithms run on which environments
EXPERIMENT_MATRIX = {
    "CartPole-v1": ["PPO", "Q-PPO", "DQN", "Q-DQN"],
    "MountainCar-v0": ["PPO", "Q-PPO", "DQN", "Q-DQN"],
    "LunarLander-v3": ["PPO", "Q-PPO", "DQN", "Q-DQN"],
    "Pendulum-v1": ["PPO", "Q-PPO", "SAC", "Q-SAC"],
    "BipedalWalker-v3": ["PPO", "Q-PPO", "SAC", "Q-SAC"],
}


def get_experiment_key(algorithm: str, env_name: str) -> str:
    """Create a unique key for an experiment."""
    return f"{algorithm}_{env_name}"


def get_default_config() -> ExperimentConfig:
    """Get the default experiment configuration."""
    config = ExperimentConfig()
    # Reduce BipedalWalker budget for practicality
    config.env_configs["BipedalWalker-v3"] = {
        "total_timesteps": 150_000,
    }
    config.env_configs["CartPole-v1"] = {
        "total_timesteps": 100_000,
    }
    config.env_configs["MountainCar-v0"] = {
        "total_timesteps": 100_000,
    }
    config.env_configs["Pendulum-v1"] = {
        "total_timesteps": 100_000,
    }
    config.env_configs["LunarLander-v3"] = {
        "total_timesteps": 150_000,
    }
    return config
