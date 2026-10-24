"""
Visualizes and records a trained Quantum or Classical RL agent in action.

Usage:
    # Run default (Q-PPO on CartPole-v1) and save animation
    python visualize_agent.py

    # Run Classical PPO for side-by-side comparison
    python visualize_agent.py --algo PPO --env CartPole-v1

    # Run DQN on MountainCar
    python visualize_agent.py --algo DQN --env MountainCar-v0

    # Watch in a live interactive window (requires GUI display)
    python visualize_agent.py --algo Q-PPO --env CartPole-v1 --live
"""

import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_rl_benchmark.config.settings import (
    get_default_config,
    ENVIRONMENTS,
    get_experiment_key,
)
from quantum_rl_benchmark.models.classical.ppo import PPOAgent
from quantum_rl_benchmark.models.classical.dqn import DQNAgent
from quantum_rl_benchmark.models.classical.sac import SACAgent
from quantum_rl_benchmark.models.quantum.q_ppo import QPPOAgent
from quantum_rl_benchmark.models.quantum.q_dqn import QDQNAgent
from quantum_rl_benchmark.models.quantum.q_sac import QSACAgent


def load_trained_agent(algo: str, checkpoint_path: str, config=None):
    """Load a trained agent from a checkpoint file."""
    if config is None:
        config = get_default_config()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    print(f"Loading {algo} agent from: {checkpoint_path}")

    if algo == "PPO":
        return PPOAgent.load_checkpoint(checkpoint_path, config=config.ppo)
    elif algo == "Q-PPO":
        return QPPOAgent.load_checkpoint(
            checkpoint_path,
            config=config.q_ppo,
            quantum_config=config.quantum,
        )
    elif algo == "DQN":
        return DQNAgent.load_checkpoint(checkpoint_path, config=config.dqn)
    elif algo == "Q-DQN":
        return QDQNAgent.load_checkpoint(
            checkpoint_path,
            config=config.q_dqn,
            quantum_config=config.quantum,
        )
    elif algo == "SAC":
        return SACAgent.load_checkpoint(checkpoint_path, config=config.sac)
    elif algo == "Q-SAC":
        return QSACAgent.load_checkpoint(
            checkpoint_path,
            config=config.q_sac,
            quantum_config=config.quantum,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")


def is_algo_viable(algo: str, env_name: str) -> bool:
    """Check if the given algorithm is compatible with the environment action space."""
    env_spec = ENVIRONMENTS.get(env_name, {})
    action_type = env_spec.get("action_type", "discrete")
    if algo in ["DQN", "Q-DQN"]:
        return action_type == "discrete"
    elif algo in ["SAC", "Q-SAC"]:
        return action_type == "continuous"
    elif algo in ["PPO", "Q-PPO"]:
        return True
    return False


def find_default_checkpoint(algo: str, env_name: str) -> str:
    """Find the best or latest checkpoint for the given algo and env."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(base_dir, "quantum_rl_benchmark", "results", "checkpoints")
    key = get_experiment_key(algo, env_name)

    candidates = [
        os.path.join(checkpoints_dir, f"{key}_best.pt"),
        os.path.join(checkpoints_dir, f"{key}_latest.pt"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No checkpoint found for {algo} on {env_name} in {checkpoints_dir}"
    )


def overlay_hud(
    frame: np.ndarray,
    algo: str,
    env_name: str,
    step: int,
    action: str,
    reward: float,
    total_reward: float,
) -> np.ndarray:
    """Draw a clean HUD overlay with telemetry onto the simulation frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Semi-transparent dark banner at top
    banner_height = 42
    banner = Image.new("RGBA", (img.width, banner_height), (15, 23, 42, 200))
    img.paste(banner, (0, 0), banner)

    # Text strings
    header = f"{algo} on {env_name}"
    stats = f"Step: {step:03d} | Action: {action} | R: {reward:+.1f} | Return: {total_reward:+.1f}"

    # Draw text (defaults to PIL basic font if custom font not available)
    draw.text((10, 4), header, fill=(56, 189, 248))  # Cyan/blue
    draw.text((10, 22), stats, fill=(241, 245, 249))  # White

    return np.array(img)


VIABLE_EXPERIMENTS = [
    # CartPole-v1 (Discrete)
    ("PPO", "CartPole-v1"),
    ("Q-PPO", "CartPole-v1"),
    ("DQN", "CartPole-v1"),
    ("Q-DQN", "CartPole-v1"),

    # MountainCar-v0 (Discrete)
    ("PPO", "MountainCar-v0"),
    ("Q-PPO", "MountainCar-v0"),
    ("DQN", "MountainCar-v0"),
    ("Q-DQN", "MountainCar-v0"),

    # LunarLander-v3 (Discrete)
    ("PPO", "LunarLander-v3"),
    ("Q-PPO", "LunarLander-v3"),
    ("DQN", "LunarLander-v3"),
    ("Q-DQN", "LunarLander-v3"),

    # Pendulum-v1 (Continuous)
    ("PPO", "Pendulum-v1"),
    ("Q-PPO", "Pendulum-v1"),
    ("SAC", "Pendulum-v1"),
    ("Q-SAC", "Pendulum-v1"),

    # BipedalWalker-v3 (Continuous)
    ("PPO", "BipedalWalker-v3"),
    ("Q-PPO", "BipedalWalker-v3"),
    ("SAC", "BipedalWalker-v3"),
    ("Q-SAC", "BipedalWalker-v3"),
]


def run_simulation(
    algo: str = "Q-PPO",
    env_name: str = "CartPole-v1",
    checkpoint_path: str = None,
    output_path: str = None,
    max_steps: int = 500,
    seed: int = 42,
    fps: int = 30,
    live: bool = False,
):
    """
    Run a single simulation episode of a trained agent and record an animated video.
    """
    if not is_algo_viable(algo, env_name):
        raise ValueError(
            f"Algorithm '{algo}' is not compatible with environment '{env_name}'! "
            f"(DQN/Q-DQN are discrete-only, SAC/Q-SAC are continuous-only)"
        )

    config = get_default_config()

    if checkpoint_path is None:
        checkpoint_path = find_default_checkpoint(algo, env_name)

    # Load agent
    agent = load_trained_agent(algo, checkpoint_path, config=config)

    # Determine render mode
    render_mode = "human" if live else "rgb_array"
    env_spec = ENVIRONMENTS.get(env_name, {})
    gym_id = env_spec.get("gymnasium_id", env_name)
    is_continuous = env_spec.get("action_type") == "continuous"

    print(f"\nInitializing {gym_id} (render_mode='{render_mode}')...")
    env = gym.make(gym_id, render_mode=render_mode)
    obs, _ = env.reset(seed=seed)

    frames = []
    total_reward = 0.0
    step = 0
    done = False

    print(f"Running simulation for {algo} on {env_name}...")

    # Record initial frame
    if not live:
        init_frame = env.render()
        if init_frame is not None:
            annotated = overlay_hud(init_frame, algo, env_name, 0, "N/A", 0.0, 0.0)
            frames.append(Image.fromarray(annotated))

    while not done and step < max_steps:
        # Select action deterministically for evaluation
        action_out = agent.select_action(obs, deterministic=True)
        if isinstance(action_out, tuple):
            action = action_out[0]
        else:
            action = action_out

        # Format action for environment
        if is_continuous:
            env_action = action if isinstance(action, np.ndarray) else np.array([action])
            if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                env_action = np.clip(env_action, env.action_space.low, env.action_space.high)
        else:
            env_action = int(action) if not isinstance(action, (int, np.integer)) else action

        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated
        total_reward += float(reward)
        step += 1

        # Capture rendered frame
        if not live:
            frame = env.render()
            if frame is not None:
                # Format action string for HUD
                if isinstance(action, np.ndarray):
                    if action.size == 1:
                        act_str = f"{action.item():+.2f}"
                    else:
                        act_str = "[" + ", ".join(f"{x:+.2f}" for x in action) + "]"
                elif isinstance(action, (float, np.floating)):
                    act_str = f"{action:+.2f}"
                else:
                    act_str = str(action)
                annotated = overlay_hud(frame, algo, env_name, step, act_str, float(reward), total_reward)
                frames.append(Image.fromarray(annotated))

        obs = next_obs

    env.close()

    print(f"\n--- Simulation Finished ---")
    print(f"Algorithm:    {algo}")
    print(f"Environment:  {env_name}")
    print(f"Total Steps:  {step}")
    print(f"Total Return: {total_reward:.2f}")

    if not live and frames:
        if output_path is None:
            videos_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "quantum_rl_benchmark",
                "results",
                "videos",
            )
            os.makedirs(videos_dir, exist_ok=True)
            output_path = os.path.join(videos_dir, f"{algo}_{env_name}_simulation.gif")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        print(f"Saving {len(frames)} frames to: {output_path} ...")
        # Save as animated GIF (universally viewable in all browsers and platforms)
        duration_ms = int(1000 / fps)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        print(f"[SUCCESS] Video saved to: {output_path}")

    return total_reward, step, output_path


def run_all_simulations(max_steps: int = 500, fps: int = 30, seed: int = 42):
    """Run simulations and generate videos for all 20 viable algorithm-environment pairs."""
    print("=" * 80)
    print(f"=== GENERATING SIMULATION VIDEOS FOR ALL {len(VIABLE_EXPERIMENTS)} VIABLE EXPERIMENTS ===")
    print("=" * 80)

    summary = []
    videos_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "quantum_rl_benchmark",
        "results",
        "videos",
    )
    os.makedirs(videos_dir, exist_ok=True)

    for idx, (algo, env_name) in enumerate(VIABLE_EXPERIMENTS, 1):
        print(f"\n[{idx:02d}/{len(VIABLE_EXPERIMENTS)}] Simulating {algo} on {env_name}...")
        try:
            total_reward, step, out_path = run_simulation(
                algo=algo,
                env_name=env_name,
                max_steps=max_steps,
                fps=fps,
                seed=seed,
            )
            summary.append({
                "algo": algo,
                "env": env_name,
                "status": "SUCCESS",
                "steps": step,
                "return": total_reward,
                "path": out_path,
            })
        except Exception as e:
            print(f"[ERROR] Failed to simulate {algo} on {env_name}: {e}")
            summary.append({
                "algo": algo,
                "env": env_name,
                "status": "FAILED",
                "steps": 0,
                "return": float("nan"),
                "error": str(e),
            })

    print("\n" + "=" * 80)
    print("=== VIDEO GENERATION SUMMARY REPORT ===")
    print("=" * 80)
    print(f"{'Idx':<4} | {'Algorithm':<8} | {'Environment':<16} | {'Status':<8} | {'Steps':<6} | {'Return':<10} | {'File'}")
    print("-" * 80)
    for idx, row in enumerate(summary, 1):
        ret_str = f"{row['return']:+.1f}" if row['status'] == "SUCCESS" else "N/A"
        filename = os.path.basename(row.get('path', '')) if row['status'] == "SUCCESS" else row.get('error', '')
        print(f"{idx:<4} | {row['algo']:<8} | {row['env']:<16} | {row['status']:<8} | {row['steps']:<6} | {ret_str:<10} | {filename}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Showcase a trained Quantum or Classical RL agent in simulation"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate simulation videos for all 20 viable algorithm-environment benchmark pairs",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="Q-PPO",
        choices=["Q-PPO", "PPO", "Q-DQN", "DQN", "Q-SAC", "SAC"],
        help="Algorithm to run (default: Q-PPO)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        choices=["CartPole-v1", "MountainCar-v0", "LunarLander-v3", "Pendulum-v1", "BipedalWalker-v3"],
        help="Environment to simulate (default: CartPole-v1)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Custom path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file path for video (.gif)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum simulation steps (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for simulation (default: 42)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for output animation (default: 30)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Render live simulation window instead of saving video",
    )

    args = parser.parse_args()

    if args.all:
        run_all_simulations(max_steps=args.max_steps, fps=args.fps, seed=args.seed)
    else:
        run_simulation(
            algo=args.algo,
            env_name=args.env,
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            max_steps=args.max_steps,
            seed=args.seed,
            fps=args.fps,
            live=args.live,
        )


if __name__ == "__main__":
    main()

