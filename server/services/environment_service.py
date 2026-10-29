"""
Environment Service for Quantum RL Arena.
Provides metadata, specifications, and control schemes for Gymnasium environments.
"""

from typing import Dict, Any, List

ENVIRONMENTS_METADATA: Dict[str, Dict[str, Any]] = {
    "CartPole-v1": {
        "id": "CartPole-v1",
        "name": "CartPole Inverted Pendulum",
        "category": "Classic Control",
        "description": "Balance a rigid pole on a moving cart by applying horizontal impulse forces. A benchmark standard for reinforcement learning stability and control.",
        "difficulty": "Introductory",
        "difficulty_level": 1,
        "obs_dim": 4,
        "action_type": "discrete",
        "action_dim": 2,
        "action_labels": ["0: Push Left (←)", "1: Push Right (→)"],
        "obs_labels": [
            "Cart Position (x)",
            "Cart Velocity (v)",
            "Pole Angle (θ)",
            "Pole Angular Velocity (ω)"
        ],
        "obs_ranges": [
            [-4.8, 4.8],
            [-5.0, 5.0],
            [-0.418, 0.418],
            [-5.0, 5.0]
        ],
        "reward_metric": "Steps alive (1.0 per step, max 500)",
        "solved_threshold": 475.0,
        "max_steps": 500,
        "human_keys": {
            "ArrowLeft": 0,
            "KeyA": 0,
            "ArrowRight": 1,
            "KeyD": 1,
        },
        "key_guide": "Press Left/A to push left, Right/D to push right.",
        "compatible_algorithms": [
            "DQN (Demonstration Solved)",
            "DQN",
            "Q-DQN",
            "PPO",
            "Q-PPO",
            "Heuristic",
            "Random Baseline"
        ]
    },
    "MountainCar-v0": {
        "id": "MountainCar-v0",
        "name": "Mountain Car Underactuated Climb",
        "category": "Classic Control",
        "description": "An underpowered car stuck in a valley must build kinetic momentum by oscillating back and forth to reach the target flag atop the mountain.",
        "difficulty": "Sparse Reward",
        "difficulty_level": 2,
        "obs_dim": 2,
        "action_type": "discrete",
        "action_dim": 3,
        "action_labels": ["0: Accelerate Left (←)", "1: Don't Accelerate (·)", "2: Accelerate Right (→)"],
        "obs_labels": [
            "Car Position (x)",
            "Car Velocity (v)"
        ],
        "obs_ranges": [
            [-1.2, 0.6],
            [-0.07, 0.07]
        ],
        "reward_metric": "-1.0 per step until goal reached (max ~ -80)",
        "solved_threshold": -110.0,
        "max_steps": 200,
        "human_keys": {
            "ArrowLeft": 0,
            "KeyA": 0,
            "ArrowDown": 1,
            "KeyS": 1,
            "ArrowRight": 2,
            "KeyD": 2,
        },
        "key_guide": "Left/A: Push Left, Right/D: Push Right, Down/S: Coast.",
        "compatible_algorithms": [
            "DQN",
            "Q-DQN",
            "PPO",
            "Q-PPO",
            "Heuristic",
            "Random Baseline"
        ]
    },
    "LunarLander-v3": {
        "id": "LunarLander-v3",
        "name": "Lunar Lander Soft Touchdown",
        "category": "Box2D Simulation",
        "description": "Guide a lunar lander safely down between the landing flags without crashing. Requires multi-thruster coordination, fuel conservation, and ground contact.",
        "difficulty": "Complex Multi-Thruster",
        "difficulty_level": 3,
        "obs_dim": 8,
        "action_type": "discrete",
        "action_dim": 4,
        "action_labels": [
            "0: Do Nothing",
            "1: Fire Left Thruster",
            "2: Fire Main Thruster",
            "3: Fire Right Thruster"
        ],
        "obs_labels": [
            "X Position",
            "Y Position",
            "X Velocity",
            "Y Velocity",
            "Angle",
            "Angular Velocity",
            "Left Leg Contact",
            "Right Leg Contact"
        ],
        "obs_ranges": [
            [-1.5, 1.5],
            [-0.2, 1.5],
            [-2.0, 2.0],
            [-2.0, 2.0],
            [-3.14, 3.14],
            [-5.0, 5.0],
            [0.0, 1.0],
            [0.0, 1.0]
        ],
        "reward_metric": "+100 to +140 for landing, -100 for crash, points for proximity & zero velocity",
        "solved_threshold": 200.0,
        "max_steps": 1000,
        "human_keys": {
            "ArrowDown": 0,
            "KeyS": 0,
            "ArrowLeft": 1,
            "KeyA": 1,
            "ArrowUp": 2,
            "KeyW": 2,
            "Space": 2,
            "ArrowRight": 3,
            "KeyD": 3,
        },
        "key_guide": "W/Up/Space: Main Thruster, A/Left: Left Engine, D/Right: Right Engine.",
        "compatible_algorithms": [
            "DQN",
            "Q-DQN",
            "PPO",
            "Q-PPO",
            "Heuristic",
            "Random Baseline"
        ]
    },
    "Pendulum-v1": {
        "id": "Pendulum-v1",
        "name": "Inverted Pendulum Continuous Torque",
        "category": "Continuous Control",
        "description": "Swing up a frictionless inverted pendulum from the hanging position and balance it upright using continuous joint torque.",
        "difficulty": "Continuous Dynamics",
        "difficulty_level": 3,
        "obs_dim": 3,
        "action_type": "continuous",
        "action_dim": 1,
        "action_labels": ["Joint Torque ∈ [-2.0, +2.0] Nm"],
        "obs_labels": [
            "Cos(θ) Angle",
            "Sin(θ) Angle",
            "Angular Velocity (ω)"
        ],
        "obs_ranges": [
            [-1.0, 1.0],
            [-1.0, 1.0],
            [-8.0, 8.0]
        ],
        "reward_metric": "Negative quadratic penalty on angle and angular velocity (best ~ 0.0)",
        "solved_threshold": -150.0,
        "max_steps": 200,
        "human_keys": {
            "ArrowLeft": -2.0,
            "KeyA": -2.0,
            "ArrowRight": 2.0,
            "KeyD": 2.0,
            "ArrowDown": 0.0,
            "KeyS": 0.0,
        },
        "key_guide": "A/Left: Max Counter-Clockwise Torque (-2.0), D/Right: Max Clockwise (+2.0), S/Down: Zero.",
        "compatible_algorithms": [
            "PPO",
            "Q-PPO",
            "SAC",
            "Q-SAC",
            "Heuristic",
            "Random Baseline"
        ]
    },
    "BipedalWalker-v3": {
        "id": "BipedalWalker-v3",
        "name": "Bipedal Walker Locomotion",
        "category": "Box2D Continuous Locomotion",
        "description": "Coordinate a 4-joint bipedal robot across rugged terrain toward the far right boundary. High dimensional continuous control with strict balance constraints.",
        "difficulty": "High-Dimensional Frontier",
        "difficulty_level": 5,
        "obs_dim": 24,
        "action_type": "continuous",
        "action_dim": 4,
        "action_labels": [
            "Hip 1 Torque",
            "Knee 1 Torque",
            "Hip 2 Torque",
            "Knee 2 Torque"
        ],
        "obs_labels": [
            "Hull Angle", "Hull Angular Vel", "X Velocity", "Y Velocity",
            "Hip 1 Angle", "Hip 1 Speed", "Knee 1 Angle", "Knee 1 Speed", "Leg 1 Ground Contact",
            "Hip 2 Angle", "Hip 2 Speed", "Knee 2 Angle", "Knee 2 Speed", "Leg 2 Ground Contact",
            "Lidar 0", "Lidar 1", "Lidar 2", "Lidar 3", "Lidar 4", "Lidar 5", "Lidar 6", "Lidar 7", "Lidar 8", "Lidar 9"
        ],
        "obs_ranges": [[-3.14, 3.14]] * 24,
        "reward_metric": "+300 for traversing the course, penalties for motor torque and falling (-100)",
        "solved_threshold": 300.0,
        "max_steps": 1600,
        "human_keys": {
            "ArrowUp": [0.5, 0.5, 0.5, 0.5],
            "ArrowDown": [-0.5, -0.5, -0.5, -0.5],
            "ArrowLeft": [-0.8, 0.4, 0.8, -0.4],
            "ArrowRight": [0.8, -0.4, -0.8, 0.4],
        },
        "key_guide": "Arrow keys apply alternating joint torques to hips and knees.",
        "compatible_algorithms": [
            "PPO",
            "Q-PPO",
            "SAC",
            "Q-SAC",
            "Random Baseline"
        ]
    }
}


def get_all_environments() -> List[Dict[str, Any]]:
    """Return a list of metadata for all supported environments."""
    return list(ENVIRONMENTS_METADATA.values())


def get_environment_info(env_id: str) -> Dict[str, Any]:
    """Retrieve metadata for a specific environment."""
    if env_id not in ENVIRONMENTS_METADATA:
        raise ValueError(f"Unknown environment: {env_id}")
    return ENVIRONMENTS_METADATA[env_id]
