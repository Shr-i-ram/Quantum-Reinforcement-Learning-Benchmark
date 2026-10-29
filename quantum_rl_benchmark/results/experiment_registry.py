"""
Central Experiment Registry for the Quantum RL Benchmark.
Maintains a structured, reproducible manifest of all benchmark runs,
demonstration models, baselines, and ablation studies.
"""

import os
import json
import glob
from typing import Dict, Any, List, Optional


REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "experiment_registry.json")


def build_experiment_registry() -> Dict[str, Any]:
    """Scan the repository and build the comprehensive experiment manifest."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "quantum_rl_benchmark", "results", "data")
    ckpt_dir = os.path.join(base_dir, "quantum_rl_benchmark", "results", "checkpoints")

    registry = {
        "version": "1.0.0",
        "description": "Comprehensive Manifest of Quantum & Classical RL Benchmark Experiments",
        "benchmark_experiments": {},
        "demonstration_experiments": {},
        "baselines": {},
        "ablations": {},
        "robustness": {},
    }

    # 1. Primary Benchmark Experiments (all 20 runs)
    for res_file in sorted(glob.glob(os.path.join(data_dir, "*_results.json"))):
        fname = os.path.basename(res_file)
        if "demonstration" in fname or "baselines" in fname or "ablation" in fname or "robustness" in fname:
            continue

        with open(res_file, "r") as f:
            d = json.load(f)

        algo = d.get("algorithm")
        env = d.get("environment")
        key = f"{algo}_{env}"
        regime = d.get("configuration_regime", "original")
        eval_res = d.get("eval_results", {})
        agent_info = d.get("agent_info", {})
        is_q = agent_info.get("is_quantum", algo.startswith("Q-"))

        # Find best and latest checkpoints
        best_ckpt = os.path.join(ckpt_dir, f"{key}_best.pt")
        latest_ckpt = os.path.join(ckpt_dir, f"{key}_latest.pt")
        primary_ckpt = best_ckpt if os.path.exists(best_ckpt) else (latest_ckpt if os.path.exists(latest_ckpt) else None)

        record = {
            "experiment_id": key,
            "algorithm": algo,
            "environment": env,
            "category": "benchmark_primary",
            "is_quantum": is_q,
            "configuration_regime": regime,
            "training_seed": d.get("config", {}).get("seed", 42),
            "training_budget_steps": d.get("total_steps", 0),
            "episodes_completed": d.get("episodes", 0),
            "wall_clock_time_s": d.get("training_time", 0.0),
            "completion_status": d.get("status", "completed"),
            "checkpoint_path": primary_ckpt,
            "checkpoint_exists": primary_ckpt is not None and os.path.exists(primary_ckpt),
            "results_file": res_file,
            "n_parameters": agent_info.get("n_parameters", 0),
            "quantum_resources": {
                "n_qubits": 4 if is_q else 0,
                "circuit_depth": 2 if is_q else 0,
                "trainable_quantum_params": 16 if is_q else 0,
                "entangling_gates": 12 if is_q else 0,
                "total_quantum_gates": 32 if is_q else 0,
                "device": "lightning.qubit" if is_q else "classical",
                "analytic_mode": True if is_q else False,
            },
            "metrics": {
                "mean_return": eval_res.get("mean_return"),
                "std_return": eval_res.get("std_return"),
                "median_return": eval_res.get("median_return"),
                "min_return": eval_res.get("min_return"),
                "max_return": eval_res.get("max_return"),
                "n_eval_episodes": eval_res.get("n_episodes", len(eval_res.get("episode_returns", []))),
                "episode_returns": eval_res.get("episode_returns", []),
            }
        }
        registry["benchmark_experiments"][key] = record

    # 2. Demonstration Experiments (Verified solvers, e.g. CartPole 480+)
    demo_file = os.path.join(data_dir, "DQN_CartPole-v1_demonstration_results.json")
    if os.path.exists(demo_file):
        with open(demo_file, "r") as f:
            d = json.load(f)
        eval_res = d.get("eval_results", {})
        key = "DQN_CartPole-v1_demonstration"
        ckpt_path = os.path.join(ckpt_dir, "DQN_CartPole-v1_demonstration.pt")
        registry["demonstration_experiments"][key] = {
            "experiment_id": key,
            "algorithm": "DQN",
            "environment": "CartPole-v1",
            "model_variant": "demonstration_reference",
            "category": "demonstration",
            "is_quantum": False,
            "target_objective": "Solved criterion (>= 475.0 return)",
            "training_budget_steps": d.get("total_steps", 40000),
            "wall_clock_time_s": d.get("training_time", 0.0),
            "checkpoint_path": ckpt_path,
            "checkpoint_exists": os.path.exists(ckpt_path),
            "results_file": demo_file,
            "metrics": {
                "mean_return": eval_res.get("mean_return"),
                "std_return": eval_res.get("std_return"),
                "median_return": eval_res.get("median_return"),
                "min_return": eval_res.get("min_return"),
                "max_return": eval_res.get("max_return"),
                "n_eval_episodes": eval_res.get("n_episodes", 20),
                "episode_returns": eval_res.get("episode_returns", []),
            }
        }

    # 3. Baselines
    base_file = os.path.join(data_dir, "baselines_results.json")
    if os.path.exists(base_file):
        with open(base_file, "r") as f:
            registry["baselines"] = json.load(f)

    # 4. Ablations
    ablation_file = os.path.join(data_dir, "ablation_results.json")
    if os.path.exists(ablation_file):
        with open(ablation_file, "r") as f:
            registry["ablations"] = json.load(f)

    # 5. Robustness
    robustness_file = os.path.join(data_dir, "robustness_results.json")
    if os.path.exists(robustness_file):
        with open(robustness_file, "r") as f:
            registry["robustness"] = json.load(f)

    # Save to JSON
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Experiment Registry written to: {REGISTRY_PATH}")
    return registry


def load_experiment_registry() -> Dict[str, Any]:
    """Load or build the experiment registry."""
    if not os.path.exists(REGISTRY_PATH):
        return build_experiment_registry()
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    reg = build_experiment_registry()
    n_bench = len(reg["benchmark_experiments"])
    n_demo = len(reg["demonstration_experiments"])
    n_base = len(reg["baselines"])
    print(f"Registry summary: {n_bench} benchmark runs, {n_demo} demonstration runs, {n_base} baselines recorded.")

