"""
Interactive Simulation Engine for Streamlit Live Demo.
Handles model loading, environment stepping, telemetry extraction,
and internal activation/quantum measurement inspection.
"""

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Tuple, Optional, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from quantum_rl_benchmark.config.settings import (
    get_default_config, ENVIRONMENTS, EXPERIMENT_MATRIX, get_experiment_key
)
from visualize_agent import load_trained_agent, find_default_checkpoint, is_algo_viable
from quantum_rl_benchmark.evaluation.baselines import (
    RandomPolicy, CartPoleHeuristicPolicy, MountainCarHeuristicPolicy,
    LunarLanderHeuristicPolicy, PendulumZeroPolicy, PendulumHeuristicPolicy,
    BipedalWalkerHeuristicPolicy
)

# Global in-memory cache for loaded agents
_AGENT_CACHE: Dict[str, Any] = {}


def get_cached_agent(algo: str, env_name: str, model_type: str = "benchmark"):
    """Load and cache agent instance."""
    cache_key = f"{algo}_{env_name}_{model_type}"
    if cache_key in _AGENT_CACHE:
        return _AGENT_CACHE[cache_key]

    config = get_default_config()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base_dir, "results", "checkpoints")

    if model_type == "demonstration":
        ckpt_path = os.path.join(ckpt_dir, f"{algo}_{env_name}_demonstration.pt")
    else:
        ckpt_path = find_default_checkpoint(algo, env_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    agent = load_trained_agent(algo, ckpt_path, config)
    _AGENT_CACHE[cache_key] = agent
    return agent


def get_compatible_models(env_name: str) -> List[Dict[str, str]]:
    """Return all available models, demonstrations, and baselines for an environment."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(base_dir, "results", "checkpoints")
    models = []

    # 1. Demonstration models if available
    demo_ckpt = os.path.join(ckpt_dir, f"DQN_{env_name}_demonstration.pt")
    if os.path.exists(demo_ckpt):
        models.append({
            "id": f"Demonstration_DQN",
            "name": "DQN (Demonstration Solved Model - 500 Return)",
            "algo": "DQN",
            "type": "demonstration",
            "is_quantum": False,
        })

    # 2. Benchmark models
    algos = EXPERIMENT_MATRIX.get(env_name, [])
    for algo in algos:
        key = get_experiment_key(algo, env_name)
        has_ckpt = os.path.exists(os.path.join(ckpt_dir, f"{key}_best.pt")) or os.path.exists(os.path.join(ckpt_dir, f"{key}_latest.pt"))
        if has_ckpt:
            is_q = algo.startswith("Q-")
            models.append({
                "id": f"Benchmark_{algo}",
                "name": f"{algo} (Primary Benchmark)",
                "algo": algo,
                "type": "benchmark",
                "is_quantum": is_q,
            })

    # 3. Baselines
    models.append({
        "id": "Baseline_Random",
        "name": "Random Policy Baseline",
        "algo": "Random",
        "type": "baseline_random",
        "is_quantum": False,
    })

    if env_name == "CartPole-v1":
        models.append({
            "id": "Baseline_Heuristic",
            "name": "Angle-Velocity PD Heuristic",
            "algo": "Heuristic",
            "type": "baseline_heuristic",
            "is_quantum": False,
        })
    elif env_name == "MountainCar-v0":
        models.append({
            "id": "Baseline_Heuristic",
            "name": "Energy-Pumping Velocity Heuristic",
            "algo": "Heuristic",
            "type": "baseline_heuristic",
            "is_quantum": False,
        })
    elif env_name == "LunarLander-v3":
        models.append({
            "id": "Baseline_Heuristic",
            "name": "PD Descent & Tilt Stabilization Heuristic",
            "algo": "Heuristic",
            "type": "baseline_heuristic",
            "is_quantum": False,
        })
    elif env_name == "Pendulum-v1":
        models.append({
            "id": "Baseline_ZeroTorque",
            "name": "Zero-Torque Passive Baseline",
            "algo": "ZeroTorque",
            "type": "baseline_zero",
            "is_quantum": False,
        })
        models.append({
            "id": "Baseline_Heuristic",
            "name": "Angular Damping Heuristic",
            "algo": "Heuristic",
            "type": "baseline_heuristic",
            "is_quantum": False,
        })
    elif env_name == "BipedalWalker-v3":
        models.append({
            "id": "Baseline_Heuristic",
            "name": "Periodic Sinusoidal Gait Heuristic",
            "algo": "Heuristic",
            "type": "baseline_heuristic",
            "is_quantum": False,
        })

    return models


def step_policy_with_inspection(
    env_name: str,
    model_info: Dict[str, str],
    obs: np.ndarray,
    env: gym.Env,
    deterministic: bool = True,
    human_action: Optional[Any] = None
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Step policy and extract deep internal inspections:
    - Discrete Q-values or Action probabilities
    - Continuous Gaussian mean and standard deviation
    - Quantum Pauli expectation values <Z_i>
    """
    spec = ENVIRONMENTS[env_name]
    action_type = spec["action_type"]
    inspection: Dict[str, Any] = {
        "model_id": model_info["id"],
        "is_quantum": model_info.get("is_quantum", False),
        "quantum_expectations": None,
        "action_distribution": None,
        "q_values": None,
        "selected_action": None,
    }

    # Case 1: Human Control
    if human_action is not None:
        action = human_action
        inspection["selected_action"] = action
        inspection["action_source"] = "Human User"
        step_act = action if action_type == "continuous" else int(action)
        return action, step_act, inspection

    m_type = model_info["type"]
    algo = model_info["algo"]

    # Case 2: Baselines
    if m_type == "baseline_random":
        action = env.action_space.sample()
        inspection["selected_action"] = action
        inspection["action_source"] = "Uniform Random Distribution"
        step_act = action
        return action, step_act, inspection

    if m_type == "baseline_zero":
        action = np.array([0.0], dtype=np.float32)
        inspection["selected_action"] = action
        step_act = action
        return action, step_act, inspection

    if m_type == "baseline_heuristic":
        if env_name == "CartPole-v1":
            p = CartPoleHeuristicPolicy()
        elif env_name == "MountainCar-v0":
            p = MountainCarHeuristicPolicy()
        elif env_name == "LunarLander-v3":
            p = LunarLanderHeuristicPolicy()
        elif env_name == "Pendulum-v1":
            p = PendulumHeuristicPolicy()
        elif env_name == "BipedalWalker-v3":
            p = BipedalWalkerHeuristicPolicy()
        action = p.select_action(obs, deterministic=True)
        inspection["selected_action"] = action
        inspection["action_source"] = "Heuristic Expert Rule"
        step_act = action
        return action, step_act, inspection

    # Case 3: Trained Agents (Classical & Hybrid Quantum)
    agent = get_cached_agent(algo, env_name, model_type=m_type)
    obs_t = torch.FloatTensor(obs).unsqueeze(0)

    with torch.no_grad():
        # --- DQN / Q-DQN ---
        if algo == "DQN":
            q_vals = agent.q_network(obs_t)[0].cpu().numpy()
            inspection["q_values"] = q_vals.tolist()
            if not deterministic and np.random.rand() < agent.epsilon:
                action = np.random.randint(0, len(q_vals))
            else:
                action = int(np.argmax(q_vals))
            step_act = action

        elif algo == "Q-DQN":
            # Extract quantum intermediate representation
            h = agent.q_network.preprocess(obs_t)
            features = agent.q_network.feature_proj(h)
            q_expvals = agent.q_network.quantum(features)[0].cpu().numpy()
            q_vals = agent.q_network.output(agent.q_network.quantum(features))[0].cpu().numpy()

            inspection["quantum_expectations"] = q_expvals.tolist()
            inspection["q_values"] = q_vals.tolist()
            if not deterministic and np.random.rand() < agent.epsilon:
                action = np.random.randint(0, len(q_vals))
            else:
                action = int(np.argmax(q_vals))
            step_act = action

        # --- PPO / Q-PPO ---
        elif algo == "PPO":
            if action_type == "continuous":
                mean, log_std, val = agent.policy(obs_t)
                std = log_std.exp()
                inspection["action_distribution"] = {
                    "mean": mean[0].cpu().numpy().tolist(),
                    "std": std[0].cpu().numpy().tolist(),
                }
                action = mean[0].cpu().numpy() if deterministic else torch.distributions.Normal(mean, std).sample()[0].cpu().numpy()
                step_act = np.clip(action, -spec.get("action_bound", 1.0), spec.get("action_bound", 1.0))
            else:
                logits, val = agent.policy(obs_t)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                inspection["action_distribution"] = {"probs": probs.tolist()}
                action = int(np.argmax(probs)) if deterministic else int(torch.distributions.Categorical(logits=logits).sample().item())
                step_act = action

        elif algo == "Q-PPO":
            # Extract quantum expectations
            h = agent.policy.preprocess(obs_t)
            features = agent.policy.feature_proj(h)
            q_expvals = agent.policy.quantum(features)[0].cpu().numpy()
            inspection["quantum_expectations"] = q_expvals.tolist()

            if action_type == "continuous":
                mean = (agent.policy.actor_mean(agent.policy.quantum(features)) * agent.policy.action_bound)[0].cpu().numpy()
                std = agent.policy.actor_log_std.exp().cpu().numpy()
                inspection["action_distribution"] = {
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                }
                action = mean if deterministic else np.random.normal(mean, std)
                step_act = np.clip(action, -spec.get("action_bound", 1.0), spec.get("action_bound", 1.0))
            else:
                logits = agent.policy.actor(agent.policy.quantum(features))
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                inspection["action_distribution"] = {"probs": probs.tolist()}
                action = int(np.argmax(probs)) if deterministic else int(torch.distributions.Categorical(logits=logits).sample().item())
                step_act = action

        # --- SAC / Q-SAC ---
        elif algo == "SAC":
            mean, log_std = agent.actor(obs_t)
            inspection["action_distribution"] = {
                "mean": mean[0].cpu().numpy().tolist(),
                "std": log_std.exp()[0].cpu().numpy().tolist(),
            }
            if deterministic:
                action = (torch.tanh(mean) * agent.action_bound)[0].cpu().numpy()
            else:
                act_t, _, _ = agent.actor.sample(obs_t)
                action = act_t[0].cpu().numpy()
            step_act = action

        elif algo == "Q-SAC":
            h = agent.actor.preprocess(obs_t)
            features = agent.actor.feature_proj(h)
            q_expvals = agent.actor.quantum(features)[0].cpu().numpy()
            inspection["quantum_expectations"] = q_expvals.tolist()

            mean, log_std = agent.actor(obs_t)
            inspection["action_distribution"] = {
                "mean": mean[0].cpu().numpy().tolist(),
                "std": log_std.exp()[0].cpu().numpy().tolist(),
            }
            if deterministic:
                action = (torch.tanh(mean) * agent.action_bound)[0].cpu().numpy()
            else:
                act_t, _, _ = agent.actor.sample(obs_t)
                action = act_t[0].cpu().numpy()
            step_act = action

        else:
            action = env.action_space.sample()
            step_act = action

    inspection["selected_action"] = action.tolist() if isinstance(action, np.ndarray) else action
    return action, step_act, inspection


def render_annotated_frame(
    frame: np.ndarray,
    env_name: str,
    model_name: str,
    step: int,
    action: Any,
    reward: float,
    cum_reward: float,
    quantum_expvals: Optional[List[float]] = None
) -> Image.Image:
    """Render high-resolution frame with telemetry overlay and optional quantum gauge banner."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Top Telemetry Banner
    banner_height = 44
    banner = Image.new("RGBA", (img.width, banner_height), (15, 23, 42, 215))
    img.paste(banner, (0, 0), banner)

    title_text = f"{model_name} on {env_name}"
    stats_text = f"Step: {step:03d} | Action: {str(action)[:14]} | r: {reward:+.2f} | Return: {cum_reward:+.1f}"

    draw.text((12, 5), title_text, fill=(56, 189, 248))      # Cyan
    draw.text((12, 23), stats_text, fill=(241, 245, 249))    # White

    # If quantum model, render subtle 4-qubit Pauli-Z expectations bar at bottom
    if quantum_expvals and len(quantum_expvals) == 4:
        q_banner_height = 24
        q_banner = Image.new("RGBA", (img.width, q_banner_height), (30, 27, 75, 220))
        img.paste(q_banner, (0, img.height - q_banner_height), q_banner)
        q_text = f"⚛ 4-Qubit <Z_i>: q0={quantum_expvals[0]:+.2f} | q1={quantum_expvals[1]:+.2f} | q2={quantum_expvals[2]:+.2f} | q3={quantum_expvals[3]:+.2f}"
        draw.text((12, img.height - q_banner_height + 4), q_text, fill=(192, 132, 252))

    return img
