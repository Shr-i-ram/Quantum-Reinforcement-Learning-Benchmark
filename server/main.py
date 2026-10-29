"""
Quantum RL Arena - FastAPI Backend Server.
Exposes full REST API for environment simulation, live quantum circuit execution,
background training studio, benchmark analytics, and static web app distribution.
"""

import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.services.environment_service import (
    get_all_environments,
    get_environment_info,
)
from server.services.simulation_service import (
    create_session,
    step_session,
    reset_session,
    close_session,
    run_batch_duel,
)
from server.services.quantum_service import (
    simulate_quantum_circuit,
    compute_quantum_scaling_analysis,
)
from server.services.training_service import (
    start_training,
    stop_training,
    get_training_status,
)
from server.services.benchmark_service import (
    get_experiment_registry,
    get_baselines_results,
    get_ablation_results,
    get_robustness_results,
    get_pareto_frontier,
    generate_research_report_markdown,
)
from server.services.challenge_service import (
    get_challenges_and_leaderboard,
    record_challenge_attempt,
)

app = FastAPI(
    title="Quantum RL Arena API",
    description="Interactive Laboratory for Classical and Quantum Reinforcement Learning",
    version="2.0.0",
)

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Request Models -----------------

class CreateSessionRequest(BaseModel):
    env_id: str = "CartPole-v1"
    algo: str = "DQN (Demonstration Solved)"
    seed: Optional[int] = 42

class StepSessionRequest(BaseModel):
    session_id: str
    action: Optional[Any] = None

class ResetSessionRequest(BaseModel):
    session_id: str
    seed: Optional[int] = None

class CloseSessionRequest(BaseModel):
    session_id: str

class BatchDuelRequest(BaseModel):
    env_id: str = "CartPole-v1"
    algo_a: str = "DQN (Demonstration Solved)"
    algo_b: str = "Q-DQN"
    n_episodes: int = 15

class QuantumSimRequest(BaseModel):
    features: List[float] = [0.2, -0.4, 0.6, -0.1]
    weights: Optional[List[List[List[float]]]] = None
    depth: int = 2

class StartTrainingRequest(BaseModel):
    algo: str = "DQN"
    env_id: str = "CartPole-v1"
    total_steps: int = 3000
    learning_rate: float = 0.001
    seed: int = 42

class RecordChallengeRequest(BaseModel):
    player_name: str = "Researcher"
    env_id: str = "CartPole-v1"
    score: float = 0.0
    steps: int = 0
    completed: bool = False

# ----------------- Environment Endpoints -----------------

@app.get("/api/environments")
def list_environments():
    return get_all_environments()

@app.get("/api/environments/{env_id}")
def get_env_details(env_id: str):
    try:
        return get_environment_info(env_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ----------------- Simulation Endpoints -----------------

@app.post("/api/sim/session")
def api_create_session(req: CreateSessionRequest):
    try:
        return create_session(env_id=req.env_id, algo=req.algo, seed=req.seed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/sim/step")
def api_step_session(req: StepSessionRequest):
    try:
        return step_session(session_id=req.session_id, human_action=req.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/sim/reset")
def api_reset_session(req: ResetSessionRequest):
    try:
        return reset_session(session_id=req.session_id, seed=req.seed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/sim/close")
def api_close_session(req: CloseSessionRequest):
    close_session(req.session_id)
    return {"success": True}

@app.post("/api/sim/batch_duel")
def api_batch_duel(req: BatchDuelRequest):
    try:
        return run_batch_duel(
            env_id=req.env_id,
            algo_a=req.algo_a,
            algo_b=req.algo_b,
            n_episodes=req.n_episodes,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Quantum Endpoints -----------------

@app.post("/api/quantum/simulate")
def api_simulate_quantum(req: QuantumSimRequest):
    try:
        return simulate_quantum_circuit(
            features=req.features,
            weights=req.weights,
            depth=req.depth,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/quantum/scaling")
def api_quantum_scaling(
    shots: int = Query(1000, ge=100, le=100000),
    depth: int = Query(2, ge=1, le=10)
):
    return compute_quantum_scaling_analysis(shots=shots, depth=depth)

# ----------------- Training Studio Endpoints -----------------

@app.post("/api/training/start")
def api_start_training(req: StartTrainingRequest):
    return start_training(
        algo=req.algo,
        env_id=req.env_id,
        total_steps=req.total_steps,
        learning_rate=req.learning_rate,
        seed=req.seed,
    )

@app.post("/api/training/stop")
def api_stop_training():
    return stop_training()

@app.get("/api/training/status")
def api_get_training_status():
    return get_training_status()

# ----------------- Benchmark Endpoints -----------------

@app.get("/api/benchmark/registry")
def api_get_registry():
    return get_experiment_registry()

@app.get("/api/benchmark/baselines")
def api_get_baselines():
    return get_baselines_results()

@app.get("/api/benchmark/ablations")
def api_get_ablations():
    return get_ablation_results()

@app.get("/api/benchmark/robustness")
def api_get_robustness():
    return get_robustness_results()

@app.get("/api/benchmark/pareto")
def api_get_pareto():
    return get_pareto_frontier()

@app.get("/api/benchmark/report", response_class=PlainTextResponse)
def api_get_report():
    return generate_research_report_markdown()

# ----------------- Challenge Endpoints -----------------

@app.get("/api/challenges")
def api_get_challenges():
    return get_challenges_and_leaderboard()

@app.post("/api/challenges/record")
def api_record_challenge(req: RecordChallengeRequest):
    return record_challenge_attempt(
        player_name=req.player_name,
        env_id=req.env_id,
        score=req.score,
        steps=req.steps,
        completed=req.completed,
    )

# ----------------- Static Frontend Delivery -----------------

FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")

if os.path.exists(FRONTEND_DIST):
    # Mount assets folder
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend_spa(full_path: str):
        # Don't intercept API calls
        if full_path.startswith("api"):
            raise HTTPException(status_code=404, detail="Not Found")
        target_file = os.path.join(FRONTEND_DIST, full_path)
        if os.path.isfile(target_file):
            return FileResponse(target_file)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    @app.get("/")
    def index_placeholder():
        return {
            "status": "online",
            "message": "Quantum RL Arena FastAPI backend is operational. Build frontend with 'npm run build' in frontend/ to serve the UI."
        }
