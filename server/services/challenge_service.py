"""
Challenge Service for Quantum RL Arena.
Tracks human challenge attempts, compares against agent benchmarks, and stores leaderboard entries.
"""

import os
import json
import time
from typing import Dict, Any, List

CHALLENGES_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "quantum_rl_benchmark", "results", "challenge_leaderboard.json")
)

CHALLENGE_PRESETS: Dict[str, Dict[str, Any]] = {
    "CartPole-v1": {
        "title": "CartPole Endurance Challenge",
        "description": "Can you keep the pole balanced for at least 300 steps using only manual left/right key taps?",
        "target_score": 300.0,
        "max_score": 500.0,
        "agent_to_beat": "DQN (Demonstration Solved) (Score: 500.0)",
        "quantum_agent_score": "Q-DQN (Score: 9.4)",
    },
    "MountainCar-v0": {
        "title": "Mountain Car Momentum Challenge",
        "description": "Build back-and-forth momentum to reach the flag atop the right mountain peak in under 160 steps.",
        "target_score": -160.0,
        "max_score": -80.0,
        "agent_to_beat": "Classical PPO (Score: -200.0)",
        "quantum_agent_score": "Q-PPO (Score: -200.0)",
    },
    "LunarLander-v3": {
        "title": "Lunar Lander Soft Touchdown Challenge",
        "description": "Pilot the lunar lander between the landing flags with zero angular velocity. Target score: > 100 points.",
        "target_score": 100.0,
        "max_score": 250.0,
        "agent_to_beat": "Classical DQN (Score: -87.8)",
        "quantum_agent_score": "Q-DQN (Score: -347.1)",
    },
    "Pendulum-v1": {
        "title": "Pendulum Swing-Up Inversion Challenge",
        "description": "Swing up and maintain the pendulum vertically upright. Target score: > -300 points.",
        "target_score": -300.0,
        "max_score": -100.0,
        "agent_to_beat": "Classical SAC (Score: -1214.3)",
        "quantum_agent_score": "Q-SAC (Score: -1266.1)",
    }
}


def _load_leaderboard() -> List[Dict[str, Any]]:
    if os.path.exists(CHALLENGES_FILE):
        try:
            with open(CHALLENGES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_leaderboard(records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(CHALLENGES_FILE), exist_ok=True)
    with open(CHALLENGES_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def get_challenges_and_leaderboard() -> Dict[str, Any]:
    """Retrieve all challenge presets and persistent user leaderboard."""
    return {
        "presets": CHALLENGE_PRESETS,
        "leaderboard": _load_leaderboard()
    }


def record_challenge_attempt(
    player_name: str,
    env_id: str,
    score: float,
    steps: int,
    completed: bool
) -> Dict[str, Any]:
    """Record a user attempt in the leaderboard."""
    records = _load_leaderboard()
    
    preset = CHALLENGE_PRESETS.get(env_id, {})
    target = preset.get("target_score", 0.0)
    beaten = (score >= target) if env_id != "MountainCar-v0" else (score >= target)

    entry = {
        "player_name": player_name.strip() or "Anonymous Researcher",
        "env_id": env_id,
        "score": round(score, 2),
        "steps": steps,
        "success": beaten,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    records.append(entry)
    _save_leaderboard(records)

    return {
        "success": True,
        "entry": entry,
        "target_score": target,
        "achieved_target": beaten
    }
