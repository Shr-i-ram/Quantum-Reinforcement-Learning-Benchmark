"""
Environment complexity characterization and normalization.

Provides metrics and normalization for comparing across environments.
"""

import numpy as np
from typing import Dict, List, Tuple


class EnvironmentComplexity:
    """
    Characterizes and normalizes environment complexity.

    Uses a multi-dimensional complexity representation:
    - Observation dimensionality
    - Action dimensionality (discrete count or continuous dim)
    - Discrete vs continuous control
    - Reward structure (sparse/dense, range)
    - Episode horizon
    - Dynamics difficulty (qualitative, mapped to numeric)
    - Exploration difficulty
    - Optimization difficulty
    """

    DIFFICULTY_MAP = {
        "low": 0.25,
        "medium": 0.5,
        "high": 0.75,
        "very_high": 1.0,
    }

    def __init__(self, env_configs: Dict):
        self.env_configs = env_configs
        self._complexity_vectors = {}
        self._normalized_vectors = {}
        self._compute_all()

    def _difficulty_to_float(self, val) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        return self.DIFFICULTY_MAP.get(str(val).lower(), 0.5)

    def _compute_complexity_vector(self, env_name: str) -> np.ndarray:
        """Compute a numeric complexity vector for an environment."""
        chars = self.env_configs[env_name]["complexity_chars"]
        vec = np.array([
            chars["obs_dim"] / 24.0,  # normalize by max obs dim
            chars["act_dim"] / 4.0,   # normalize by max act dim
            0.0 if chars["discrete"] else 1.0,
            1.0 if chars["sparse_reward"] else 0.0,
            chars["horizon"] / 2000.0,  # normalize by max horizon
            self._difficulty_to_float(chars["dynamics_difficulty"]),
            self._difficulty_to_float(chars["exploration_difficulty"]),
            self._difficulty_to_float(chars["optimization_difficulty"]),
        ], dtype=np.float64)
        return vec

    def _compute_all(self):
        """Compute and normalize complexity vectors for all environments."""
        for env_name in self.env_configs:
            self._complexity_vectors[env_name] = self._compute_complexity_vector(env_name)

        # Stack all vectors for normalization
        all_vecs = np.stack(list(self._complexity_vectors.values()))
        mins = all_vecs.min(axis=0)
        maxs = all_vecs.max(axis=0)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0  # avoid division by zero

        for env_name, vec in self._complexity_vectors.items():
            self._normalized_vectors[env_name] = (vec - mins) / ranges

    def get_complexity_vector(self, env_name: str, normalized: bool = True) -> np.ndarray:
        """Get complexity vector for an environment."""
        if normalized:
            return self._normalized_vectors[env_name].copy()
        return self._complexity_vectors[env_name].copy()

    def get_complexity_score(self, env_name: str) -> float:
        """Get a single scalar complexity score (mean of normalized vector)."""
        return float(np.mean(self._normalized_vectors[env_name]))

    def get_complexity_ranking(self) -> List[Tuple[str, float]]:
        """Get environments ranked by complexity score."""
        scores = [(name, self.get_complexity_score(name)) for name in self.env_configs]
        return sorted(scores, key=lambda x: x[1])

    def get_discrete_continuous_boundary(self) -> int:
        """Get the complexity level index where discrete->continuous transition occurs."""
        for name, cfg in sorted(self.env_configs.items(), key=lambda x: x[1]["complexity_level"]):
            if cfg["action_type"] == "continuous":
                return cfg["complexity_level"] - 1
        return 0

    def normalize_rewards(self, rewards: Dict[str, float],
                          env_names: List[str]) -> Dict[str, float]:
        """
        Normalize rewards across environments for cross-environment comparison.
        Uses min-max normalization per environment based on reward ranges.
        """
        normalized = {}
        for env_name in env_names:
            cfg = self.env_configs[env_name]
            r_min, r_max = cfg["reward_range"]
            r_range = r_max - r_min
            if r_range == 0:
                normalized[env_name] = 0.0
            else:
                raw = rewards.get(env_name, 0.0)
                normalized[env_name] = (raw - r_min) / r_range
        return normalized

    def complexity_description(self, env_name: str) -> str:
        """Human-readable complexity description."""
        chars = self.env_configs[env_name]["complexity_chars"]
        ctrl = "discrete" if chars["discrete"] else "continuous"
        sparse = "sparse" if chars["sparse_reward"] else "dense"
        return (
            f"Obs dim={chars['obs_dim']}, Act dim={chars['act_dim']} ({ctrl}), "
            f"Reward={sparse}, Horizon={chars['horizon']}, "
            f"Dynamics={chars['dynamics_difficulty']}, "
            f"Exploration={chars['exploration_difficulty']}, "
            f"Optimization={chars['optimization_difficulty']}"
        )
