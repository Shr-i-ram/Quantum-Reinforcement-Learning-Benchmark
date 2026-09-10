"""
Environment creation and management.

Handles creation, validation, and seeding of all Gymnasium environments.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any


class EnvironmentManager:
    """Creates and manages Gymnasium environments with proper seeding."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._available_envs = {}

    def validate_all(self, env_specs: Dict) -> Dict[str, bool]:
        """Validate that all required environments are available."""
        results = {}
        for env_name, spec in env_specs.items():
            gym_id = spec["gymnasium_id"]
            try:
                env = gym.make(gym_id)
                obs, info = env.reset(seed=self.seed)
                results[env_name] = True
                env.close()
            except Exception as e:
                results[env_name] = False
                print(f"  [FAIL] {env_name} ({gym_id}): {e}")
        return results

    def create_env(self, env_name: str, env_specs: Dict,
                   seed: Optional[int] = None,
                   max_episode_steps: Optional[int] = None,
                   render_mode: Optional[str] = None) -> gym.Env:
        """Create a Gymnasium environment with proper configuration."""
        if env_name not in env_specs:
            raise ValueError(f"Unknown environment: {env_name}")

        spec = env_specs[env_name]
        gym_id = spec["gymnasium_id"]

        use_seed = seed if seed is not None else self.seed

        env = gym.make(gym_id, render_mode=render_mode)

        # Apply episode time limit if specified
        if max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

        return env

    def create_and_seed(self, env_name: str, env_specs: Dict,
                        seed: Optional[int] = None) -> gym.Env:
        """Create, seed, and return an environment."""
        use_seed = seed if seed is not None else self.seed
        env = self.create_env(env_name, env_specs, seed=use_seed)
        env.reset(seed=use_seed)
        # Seed action and observation spaces
        env.action_space.seed(use_seed)
        env.observation_space.seed(use_seed)
        return env

    def get_env_info(self, env_name: str, env_specs: Dict) -> Dict[str, Any]:
        """Get comprehensive environment information."""
        env = self.create_env(env_name, env_specs)
        info = {
            "name": env_name,
            "gymnasium_id": env_specs[env_name]["gymnasium_id"],
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "obs_shape": env.observation_space.shape,
            "action_type": env_specs[env_name]["action_type"],
        }
        if hasattr(env.action_space, "n"):
            info["n_actions"] = env.action_space.n
        if hasattr(env.action_space, "shape"):
            info["action_shape"] = env.action_space.shape
            info["action_low"] = float(env.action_space.low.min())
            info["action_high"] = float(env.action_space.high.max())
        env.close()
        return info

    def validate_all_comprehensive(self, env_specs: Dict) -> Dict[str, Dict]:
        """Comprehensive validation of all environments."""
        results = {}
        for env_name, spec in env_specs.items():
            try:
                env_info = self.get_env_info(env_name, env_specs)
                env_info["available"] = True
                results[env_name] = env_info
            except Exception as e:
                results[env_name] = {
                    "available": False,
                    "error": str(e),
                }
        return results
