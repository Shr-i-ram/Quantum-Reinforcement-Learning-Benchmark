"""
Training Service for Quantum RL Arena.
Wraps the LiveTrainingJob background worker to manage interactive training sessions.
"""

from typing import Dict, Any, Optional
from quantum_rl_benchmark.demo.training_worker import TRAINING_JOB


def start_training(
    algo: str,
    env_id: str,
    total_steps: int = 5000,
    learning_rate: float = 1e-3,
    seed: int = 42
) -> Dict[str, Any]:
    """Start real background training session."""
    if TRAINING_JOB.is_running:
        return {"success": False, "message": "A training session is already running."}

    # Cap steps for demo safety: 500 to 20,000 steps
    total_steps = max(500, min(total_steps, 20000))

    try:
        TRAINING_JOB.start(
            algorithm=algo,
            env_name=env_id,
            total_steps=total_steps,
            learning_rate=learning_rate,
            seed=seed
        )
        return {"success": True, "message": f"Training {algo} on {env_id} started."}
    except Exception as e:
        return {"success": False, "message": str(e)}


def stop_training() -> Dict[str, Any]:
    """Stop active background training session."""
    if not TRAINING_JOB.is_running:
        return {"success": False, "message": "No active training session to stop."}
    TRAINING_JOB.stop()
    return {"success": True, "message": "Stopping training job gracefully..."}


def get_training_status() -> Dict[str, Any]:
    """Get current training progress, step metrics, and loss/return histories."""
    status = dict(TRAINING_JOB.status)
    status["is_running"] = TRAINING_JOB.is_running
    status["error"] = TRAINING_JOB.error
    
    # Calculate step per second FPS
    elapsed = status.get("elapsed_time_s", 0.0)
    current_step = status.get("current_step", 0)
    fps = (current_step / elapsed) if elapsed > 0.1 else 0.0
    status["fps"] = round(fps, 1)

    return status
