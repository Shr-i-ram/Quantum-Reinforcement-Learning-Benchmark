"""
Audit script for Quantum RL Benchmark.
Validates loading, inference, parameters, quantum execution, and results data.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import gymnasium as gym

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_rl_benchmark.config.settings import (
    get_default_config, ENVIRONMENTS, EXPERIMENT_MATRIX, get_experiment_key
)
from visualize_agent import load_trained_agent, find_default_checkpoint, is_algo_viable
from quantum_rl_benchmark.models.quantum.circuit import QuantumCircuitLayer


def audit_checkpoints_and_inference():
    print("=" * 80)
    print("AUDIT: Checkpoint Loading & Inference Verification")
    print("=" * 80)
    
    config = get_default_config()
    report = []
    
    for env_name, algos in EXPERIMENT_MATRIX.items():
        spec = ENVIRONMENTS[env_name]
        gym_id = spec["gymnasium_id"]
        
        # Test environment creation
        try:
            env = gym.make(gym_id)
            init_obs, _ = env.reset(seed=42)
            obs_shape = init_obs.shape
            env_status = "OK"
            env.close()
        except Exception as e:
            env_status = f"FAIL: {e}"
            obs_shape = None
            
        print(f"\n[Environment] {env_name} ({gym_id}) -> Status: {env_status}, Obs Shape: {obs_shape}")
        
        for algo in algos:
            key = get_experiment_key(algo, env_name)
            item = {
                "algorithm": algo,
                "environment": env_name,
                "key": key,
                "checkpoint_found": False,
                "checkpoint_path": None,
                "load_success": False,
                "inference_success": False,
                "total_steps": None,
                "param_count": None,
                "deterministic_action": None,
                "stochastic_action": None,
                "quantum_circuit_present": False,
                "error": None,
            }
            
            try:
                ckpt_path = find_default_checkpoint(algo, env_name)
                item["checkpoint_found"] = True
                item["checkpoint_path"] = ckpt_path
                
                # Check file size & metadata in checkpoint
                raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                item["total_steps"] = raw_ckpt.get("total_steps", raw_ckpt.get("training_step", "Unknown"))
                
                # Load agent
                agent = load_trained_agent(algo, ckpt_path, config)
                item["load_success"] = True
                
                # Inspect parameters
                if hasattr(agent, "get_info"):
                    info = agent.get_info()
                    item["param_count"] = info.get("n_parameters", None)
                    item["quantum_circuit_present"] = info.get("is_quantum", False)
                elif hasattr(agent, "policy"):
                    item["param_count"] = sum(p.numel() for p in agent.policy.parameters())
                elif hasattr(agent, "q_network"):
                    item["param_count"] = sum(p.numel() for p in agent.q_network.parameters())
                    
                # Test inference on single observation
                env = gym.make(gym_id)
                obs, _ = env.reset(seed=42)
                
                # Deterministic action
                det_action = agent.select_action(obs, deterministic=True)
                if isinstance(det_action, tuple):
                    det_action = det_action[0]
                item["deterministic_action"] = str(det_action)
                
                # Stochastic action
                stoch_action = agent.select_action(obs, deterministic=False)
                if isinstance(stoch_action, tuple):
                    stoch_action = stoch_action[0]
                item["stochastic_action"] = str(stoch_action)
                
                # Step environment
                if spec["action_type"] == "continuous":
                    act_to_step = det_action if isinstance(det_action, np.ndarray) else np.array([det_action])
                else:
                    act_to_step = int(det_action) if not isinstance(det_action, (int, np.integer)) else det_action
                    
                next_obs, r, term, trunc, _ = env.step(act_to_step)
                env.close()
                
                item["inference_success"] = True
                print(f"  [OK] {algo:8s} on {env_name:18s} | Steps: {str(item['total_steps']):8s} | Params: {item['param_count']} | Action: {str(det_action)[:15]}")
                
            except Exception as e:
                item["error"] = str(e)
                print(f"  [FAIL] {algo:8s} on {env_name:18s} | Error: {e}")
                
            report.append(item)
            
    return report


def audit_quantum_gradients():
    print("\n" + "=" * 80)
    print("AUDIT: Quantum Gradient Propagation Verification")
    print("=" * 80)
    
    q_layer = QuantumCircuitLayer(n_qubits=4, depth=2, differentiation_method="adjoint")
    
    # Test batch inputs
    dummy_input = torch.randn(8, 4, requires_grad=True)
    out = q_layer(dummy_input)
    loss = out.sum()
    loss.backward()
    
    print(f"QuantumCircuitLayer output shape: {out.shape}")
    print(f"Input gradient exists: {dummy_input.grad is not None}, norm: {dummy_input.grad.norm().item():.4f}")
    print(f"Param gradient exists: {q_layer.params.grad is not None}, norm: {q_layer.params.grad.norm().item():.4f}")
    
    assert dummy_input.grad is not None and dummy_input.grad.norm() > 0, "Input gradients vanished!"
    assert q_layer.params.grad is not None and q_layer.params.grad.norm() > 0, "Param gradients vanished!"
    print("Quantum gradient check PASSED.")


def audit_results_data():
    print("\n" + "=" * 80)
    print("AUDIT: Stored Benchmark Results Data vs all_metrics.json")
    print("=" * 80)
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "quantum_rl_benchmark", "results", "data")
    all_metrics_path = os.path.join(data_dir, "all_metrics.json")
    
    if os.path.exists(all_metrics_path):
        with open(all_metrics_path) as f:
            all_metrics = json.load(f)
        print(f"all_metrics.json loaded. Total keys: {len(all_metrics)}")
    else:
        all_metrics = {}
        print("all_metrics.json not found!")
        
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_results.json"):
            path = os.path.join(data_dir, f)
            with open(path) as fh:
                d = json.load(fh)
            algo = d.get("algorithm")
            env = d.get("environment")
            regime = d.get("configuration_regime", "original")
            eval_res = d.get("eval_results", {})
            mean_ret = eval_res.get("mean_return", "N/A")
            std_ret = eval_res.get("std_return", "N/A")
            ep_rets = eval_res.get("episode_returns", [])
            steps = d.get("total_steps", "N/A")
            t_time = d.get("training_time", "N/A")
            
            print(f"File {f:35s} | Regime: {regime:10s} | Steps: {str(steps):8s} | Mean Return: {mean_ret:9.2f} +/- {std_ret:6.2f} (n={len(ep_rets)}) | Time: {t_time:.1f}s")


if __name__ == "__main__":
    rep = audit_checkpoints_and_inference()
    audit_quantum_gradients()
    audit_results_data()
