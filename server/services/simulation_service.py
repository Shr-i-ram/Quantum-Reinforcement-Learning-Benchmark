"""
Simulation Service for Quantum RL Arena.
Handles interactive environment stepping, agent loading, frame rendering,
deep model telemetry extraction (Pauli-Z expectations, Q-values), and batch duels.
"""

import os
import sys
import uuid
import base64
import io
import time
import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, Tuple, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from quantum_rl_benchmark.config.settings import get_default_config, ENVIRONMENTS, get_experiment_key
from quantum_rl_benchmark.models.classical.ppo import PPOAgent
from quantum_rl_benchmark.models.classical.dqn import DQNAgent
from quantum_rl_benchmark.models.classical.sac import SACAgent
from quantum_rl_benchmark.models.quantum.q_ppo import QPPOAgent
from quantum_rl_benchmark.models.quantum.q_dqn import QDQNAgent
from quantum_rl_benchmark.models.quantum.q_sac import QSACAgent
from server.services.environment_service import ENVIRONMENTS_METADATA

# In-memory session store
_ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}
_AGENT_CACHE: Dict[str, Any] = {}


def _get_checkpoint_path(algo: str, env_id: str) -> Optional[str]:
    """Find the checkpoint path for a given algorithm and environment."""
    checkpoints_dir = os.path.join(PROJECT_ROOT, "quantum_rl_benchmark", "results", "checkpoints")
    
    if algo == "DQN (Demonstration Solved)" and env_id == "CartPole-v1":
        path = os.path.join(checkpoints_dir, "DQN_CartPole-v1_demonstration.pt")
        if os.path.exists(path):
            return path

    base_algo = algo.replace(" (Demonstration Solved)", "")
    key = get_experiment_key(base_algo, env_id)
    
    candidates = [
        os.path.join(checkpoints_dir, f"{key}_best.pt"),
        os.path.join(checkpoints_dir, f"{key}_latest.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_cached_agent(algo: str, env_id: str):
    """Load and cache agent instances to avoid reloading disk checkpoints on every step."""
    cache_key = f"{algo}::{env_id}"
    if cache_key in _AGENT_CACHE:
        return _AGENT_CACHE[cache_key]

    ckpt_path = _get_checkpoint_path(algo, env_id)
    if not ckpt_path:
        return None

    config = get_default_config()
    clean_algo = algo.replace(" (Demonstration Solved)", "")

    try:
        if clean_algo == "PPO":
            agent = PPOAgent.load_checkpoint(ckpt_path, config=config.ppo)
        elif clean_algo == "Q-PPO":
            agent = QPPOAgent.load_checkpoint(ckpt_path, config=config.q_ppo, quantum_config=config.quantum)
        elif clean_algo == "DQN":
            agent = DQNAgent.load_checkpoint(ckpt_path, config=config.dqn)
        elif clean_algo == "Q-DQN":
            agent = QDQNAgent.load_checkpoint(ckpt_path, config=config.q_dqn, quantum_config=config.quantum)
        elif clean_algo == "SAC":
            agent = SACAgent.load_checkpoint(ckpt_path, config=config.sac)
        elif clean_algo == "Q-SAC":
            agent = QSACAgent.load_checkpoint(ckpt_path, config=config.q_sac, quantum_config=config.quantum)
        else:
            return None
        _AGENT_CACHE[cache_key] = agent
        return agent
    except Exception as e:
        print(f"Warning: Failed to load checkpoint {ckpt_path}: {e}")
        return None


def _render_frame_to_base64(env: gym.Env, hud_text: Optional[Dict[str, str]] = None) -> str:
    """Render the gymnasium environment frame, overlay HUD telemetry, and encode to base64 JPEG."""
    try:
        frame = env.render()
        if frame is None or not isinstance(frame, np.ndarray):
            # Fallback synthetic frame if environment rendering fails
            frame = np.zeros((400, 600, 3), dtype=np.uint8)
        
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Draw HUD bar if telemetry is supplied
        if hud_text:
            banner_height = 36
            banner = Image.new("RGBA", (img.width, banner_height), (15, 23, 42, 220))
            img.paste(banner, (0, 0), banner)
            
            line1 = f"{hud_text.get('title', 'Quantum RL Arena')} | Step: {hud_text.get('step', '0')}"
            line2 = f"Action: {hud_text.get('action', '-')} | R: {hud_text.get('reward', '0.0')} | Return: {hud_text.get('return', '0.0')}"
            
            draw.text((10, 3), line1, fill=(56, 189, 248))   # Sky blue
            draw.text((10, 18), line2, fill=(241, 245, 249))  # White

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        # Return simple transparent placeholder on error
        return ""


def _heuristic_action(env_id: str, state: np.ndarray) -> Any:
    """Compute a simple domain-specific heuristic action for benchmark comparison."""
    if env_id == "CartPole-v1":
        # Heuristic: push in direction pole is falling (pole angle and angular velocity)
        pole_angle = state[2]
        pole_vel = state[3]
        return 1 if (pole_angle + 0.1 * pole_vel) > 0 else 0
    elif env_id == "MountainCar-v0":
        # Heuristic: push in direction of velocity
        velocity = state[1]
        if velocity > 0:
            return 2  # Right
        elif velocity < 0:
            return 0  # Left
        return 1      # Coast
    elif env_id == "Pendulum-v1":
        # Heuristic: counter torque against angular velocity and angle
        cos_th, sin_th, th_dot = state[0], state[1], state[2]
        torque = -2.0 * sin_th - 0.5 * th_dot
        return np.array([np.clip(torque, -2.0, 2.0)], dtype=np.float32)
    elif env_id == "LunarLander-v3":
        # Heuristic: fire main engine if falling fast
        y_vel = state[3]
        angle = state[4]
        if y_vel < -0.3:
            return 2  # Main engine
        elif angle > 0.1:
            return 1  # Left engine
        elif angle < -0.1:
            return 3  # Right engine
        return 0
    return 0


def create_session(env_id: str, algo: str = "Human Manual", seed: Optional[int] = 42) -> Dict[str, Any]:
    """Create a new simulation session."""
    if env_id not in ENVIRONMENTS_METADATA:
        raise ValueError(f"Unknown environment: {env_id}")

    session_id = str(uuid.uuid4())
    env = gym.make(env_id, render_mode="rgb_array")
    state, info = env.reset(seed=seed)

    agent = None
    if algo not in ["Human Manual", "Random Baseline", "Heuristic"]:
        agent = _load_cached_agent(algo, env_id)

    hud_data = {
        "title": f"{algo} on {env_id}",
        "step": "0",
        "action": "INIT",
        "reward": "0.0",
        "return": "0.0",
    }
    frame_base64 = _render_frame_to_base64(env, hud_data)

    session_data = {
        "session_id": session_id,
        "env": env,
        "env_id": env_id,
        "algo": algo,
        "agent": agent,
        "state": state,
        "step_count": 0,
        "cumulative_reward": 0.0,
        "done": False,
        "termination_reason": "",
        "seed": seed,
        "history": []
    }
    _ACTIVE_SESSIONS[session_id] = session_data

    return {
        "session_id": session_id,
        "env_id": env_id,
        "algo": algo,
        "state": state.tolist(),
        "step_count": 0,
        "cumulative_reward": 0.0,
        "done": False,
        "frame_base64": frame_base64,
        "telemetry": {}
    }


def step_session(session_id: str, human_action: Optional[Any] = None) -> Dict[str, Any]:
    """Execute a single simulation step via agent inference or manual human input."""
    if session_id not in _ACTIVE_SESSIONS:
        raise ValueError(f"Session {session_id} not found")

    session = _ACTIVE_SESSIONS[session_id]
    if session["done"]:
        return {
            "session_id": session_id,
            "done": True,
            "step_count": session["step_count"],
            "cumulative_reward": session["cumulative_reward"],
            "state": session["state"].tolist(),
            "termination_reason": session["termination_reason"],
            "frame_base64": "",
            "telemetry": {}
        }

    env = session["env"]
    env_id = session["env_id"]
    algo = session["algo"]
    agent = session["agent"]
    state = session["state"]

    action = None
    telemetry: Dict[str, Any] = {
        "is_quantum": False,
        "expectation_values": [],
        "projected_features": [],
        "q_values": [],
        "action_probabilities": [],
    }

    # Determine action
    if human_action is not None:
        action = human_action
        if isinstance(action, (int, float)) and ENVIRONMENTS_METADATA[env_id]["action_type"] == "continuous":
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, list) and ENVIRONMENTS_METADATA[env_id]["action_type"] == "continuous":
            action = np.array(action, dtype=np.float32)
    elif algo == "Random Baseline":
        action = env.action_space.sample()
    elif algo == "Heuristic":
        action = _heuristic_action(env_id, state)
    elif agent is not None:
        # Deep telemetry extraction
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            clean_algo = algo.replace(" (Demonstration Solved)", "")
            if clean_algo == "Q-DQN":
                telemetry["is_quantum"] = True
                h = agent.q_network.preprocess(state_tensor)
                features = agent.q_network.feature_proj(h)
                telemetry["projected_features"] = features[0].tolist()
                quantum_out = agent.q_network.quantum(features)
                telemetry["expectation_values"] = quantum_out[0].tolist()
                q_vals = agent.q_network.output(quantum_out)
                telemetry["q_values"] = q_vals[0].tolist()
                action = int(torch.argmax(q_vals, dim=-1).item())
            elif clean_algo == "DQN":
                q_vals = agent.q_network(state_tensor)
                telemetry["q_values"] = q_vals[0].tolist()
                action = int(torch.argmax(q_vals, dim=-1).item())
            elif clean_algo == "Q-PPO":
                telemetry["is_quantum"] = True
                h = agent.actor.preprocess(state_tensor)
                features = agent.actor.feature_proj(h)
                telemetry["projected_features"] = features[0].tolist()
                quantum_out = agent.actor.quantum(features)
                telemetry["expectation_values"] = quantum_out[0].tolist()
                if agent.is_continuous:
                    mean = agent.actor.mean_linear(quantum_out)
                    action = mean.squeeze(0).numpy()
                    telemetry["continuous_mean"] = action.tolist()
                else:
                    logits = agent.actor.output(quantum_out)
                    probs = torch.softmax(logits, dim=-1)
                    telemetry["action_probabilities"] = probs[0].tolist()
                    action = int(torch.argmax(probs, dim=-1).item())
            elif clean_algo == "PPO":
                if agent.is_continuous:
                    mean, _ = agent.actor(state_tensor)
                    action = mean.squeeze(0).numpy()
                    telemetry["continuous_mean"] = action.tolist()
                else:
                    logits = agent.actor(state_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    telemetry["action_probabilities"] = probs[0].tolist()
                    action = int(torch.argmax(probs, dim=-1).item())
            elif clean_algo == "Q-SAC":
                telemetry["is_quantum"] = True
                h = agent.actor.preprocess(state_tensor)
                features = agent.actor.feature_proj(h)
                telemetry["projected_features"] = features[0].tolist()
                quantum_out = agent.actor.quantum(features)
                telemetry["expectation_values"] = quantum_out[0].tolist()
                action = agent.select_action(state, evaluate=True)
            elif clean_algo == "SAC":
                action = agent.select_action(state, evaluate=True)
            else:
                action = env.action_space.sample()
    else:
        action = env.action_space.sample()

    # Step environment
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    session["step_count"] += 1
    session["cumulative_reward"] += float(reward)
    session["state"] = next_state
    session["done"] = done

    if terminated:
        session["termination_reason"] = "Terminal State Reached (Goal or Failure)"
    elif truncated:
        session["termination_reason"] = "Maximum Episode Time Horizon Reached"

    # Action formatting for HUD
    action_str = str(action)
    if isinstance(action, np.ndarray):
        action_str = f"[{', '.join(f'{x:.2f}' for x in action.flatten())}]"
    elif isinstance(action, (float, np.floating)):
        action_str = f"{action:.2f}"

    hud_data = {
        "title": f"{algo} on {env_id}",
        "step": f"{session['step_count']}",
        "action": action_str,
        "reward": f"{reward:+.2f}",
        "return": f"{session['cumulative_reward']:+.1f}",
    }
    frame_base64 = _render_frame_to_base64(env, hud_data)

    # Append to trajectory history (compact)
    session["history"].append({
        "step": session["step_count"],
        "action": action_str,
        "reward": float(reward),
        "return": float(session["cumulative_reward"]),
        "state": next_state.tolist()
    })

    return {
        "session_id": session_id,
        "env_id": env_id,
        "algo": algo,
        "step_count": session["step_count"],
        "reward": float(reward),
        "cumulative_reward": session["cumulative_reward"],
        "done": done,
        "state": next_state.tolist(),
        "action": action_str,
        "termination_reason": session["termination_reason"],
        "frame_base64": frame_base64,
        "telemetry": telemetry
    }


def reset_session(session_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Reset an existing simulation session."""
    if session_id not in _ACTIVE_SESSIONS:
        raise ValueError(f"Session {session_id} not found")

    session = _ACTIVE_SESSIONS[session_id]
    env = session["env"]
    env_id = session["env_id"]
    algo = session["algo"]

    state, info = env.reset(seed=seed)
    session["state"] = state
    session["step_count"] = 0
    session["cumulative_reward"] = 0.0
    session["done"] = False
    session["termination_reason"] = ""
    session["history"] = []

    hud_data = {
        "title": f"{algo} on {env_id}",
        "step": "0",
        "action": "RESET",
        "reward": "0.0",
        "return": "0.0",
    }
    frame_base64 = _render_frame_to_base64(env, hud_data)

    return {
        "session_id": session_id,
        "env_id": env_id,
        "algo": algo,
        "state": state.tolist(),
        "step_count": 0,
        "cumulative_reward": 0.0,
        "done": False,
        "frame_base64": frame_base64,
        "telemetry": {}
    }


def close_session(session_id: str):
    """Close and clean up environment session."""
    if session_id in _ACTIVE_SESSIONS:
        session = _ACTIVE_SESSIONS.pop(session_id)
        try:
            session["env"].close()
        except Exception:
            pass


def run_batch_duel(env_id: str, algo_a: str, algo_b: str, n_episodes: int = 15) -> Dict[str, Any]:
    """
    Run synchronized headless batch duels between two algorithms across identical initial seeds.
    Returns return distributions, episode lengths, and win/loss statistics.
    """
    n_episodes = min(max(n_episodes, 5), 50)
    agent_a = _load_cached_agent(algo_a, env_id)
    agent_b = _load_cached_agent(algo_b, env_id)

    env_spec = ENVIRONMENTS_METADATA.get(env_id, {})
    max_steps = env_spec.get("max_steps", 500)

    returns_a, returns_b = [], []
    lengths_a, lengths_b = [], []

    test_env = gym.make(env_id)

    def evaluate_episode(agent, algo_name, seed):
        s, _ = test_env.reset(seed=seed)
        total_r = 0.0
        steps = 0
        for _ in range(max_steps):
            if algo_name == "Random Baseline":
                a = test_env.action_space.sample()
            elif algo_name == "Heuristic":
                a = _heuristic_action(env_id, s)
            elif agent is not None:
                if hasattr(agent, "predict"):
                    a = agent.predict(s)
                elif hasattr(agent, "select_action"):
                    a = agent.select_action(s, evaluate=True)
                else:
                    a = test_env.action_space.sample()
            else:
                a = test_env.action_space.sample()

            s, r, term, trunc, _ = test_env.step(a)
            total_r += float(r)
            steps += 1
            if term or trunc:
                break
        return total_r, steps

    for ep in range(n_episodes):
        seed = 4200 + ep
        r_a, l_a = evaluate_episode(agent_a, algo_a, seed)
        r_b, l_b = evaluate_episode(agent_b, algo_b, seed)
        returns_a.append(r_a)
        returns_b.append(r_b)
        lengths_a.append(l_a)
        lengths_b.append(l_b)

    test_env.close()

    arr_a = np.array(returns_a)
    arr_b = np.array(returns_b)
    wins_a = int(np.sum(arr_a > arr_b))
    wins_b = int(np.sum(arr_b > arr_a))
    ties = int(np.sum(arr_a == arr_b))

    return {
        "env_id": env_id,
        "algo_a": algo_a,
        "algo_b": algo_b,
        "n_episodes": n_episodes,
        "results_a": {
            "returns": returns_a,
            "lengths": lengths_a,
            "mean_return": float(np.mean(arr_a)),
            "std_return": float(np.std(arr_a)),
            "median_return": float(np.median(arr_a)),
            "min_return": float(np.min(arr_a)),
            "max_return": float(np.max(arr_a)),
            "wins": wins_a,
        },
        "results_b": {
            "returns": returns_b,
            "lengths": lengths_b,
            "mean_return": float(np.mean(arr_b)),
            "std_return": float(np.std(arr_b)),
            "median_return": float(np.median(arr_b)),
            "min_return": float(np.min(arr_b)),
            "max_return": float(np.max(arr_b)),
            "wins": wins_b,
        },
        "ties": ties,
    }
