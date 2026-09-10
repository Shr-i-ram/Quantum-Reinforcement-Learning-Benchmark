"""
Post-training evaluation and metric computation.

Provides comprehensive evaluation beyond the training loop,
including detailed per-experiment analysis.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional


class Evaluator:
    """
    Comprehensive evaluator for all trained models.

    Loads results from training and computes additional metrics.
    """

    def __init__(self, data_dir: str, fig_dir: str):
        self.data_dir = data_dir
        self.fig_dir = fig_dir
        self.all_results = {}

    def load_all_results(self) -> Dict[str, Dict]:
        """Load all completed experiment results."""
        for f in os.listdir(self.data_dir):
            if f.endswith("_results.json"):
                path = os.path.join(self.data_dir, f)
                with open(path, "r") as fh:
                    data = json.load(fh)
                key = data.get("algorithm", "") + "_" + data.get("environment", "")
                self.all_results[key] = data
        return self.all_results

    def compute_sample_efficiency(self, history: Dict, threshold: float) -> Dict[str, Any]:
        """
        Compute sample efficiency metrics.

        Returns steps to threshold, whether threshold was reached, and AULC.
        """
        episode_rewards = history.get("episode_rewards", [])
        total_steps = history.get("total_steps", [])

        if not episode_rewards or not total_steps:
            return {
                "steps_to_threshold": None,
                "reached_threshold": False,
                "aulc": 0.0,
                "episodes_to_threshold": None,
            }

        rewards = np.array([float(r) for r in episode_rewards], dtype=np.float64)
        steps = np.array([float(s) for s in total_steps], dtype=np.float64)

        # Moving average (window 50)
        window = min(50, len(rewards))
        if window > 0:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        else:
            moving_avg = rewards

        # Steps to threshold
        reached = False
        steps_to_thresh = None
        eps_to_thresh = None
        for i, avg in enumerate(moving_avg):
            if avg >= threshold:
                reached = True
                idx = i + window - 1
                if idx < len(steps):
                    steps_to_thresh = int(steps[idx])
                eps_to_thresh = i + 1
                break

        # AULC (Area Under Learning Curve) - using moving average
        if len(moving_avg) > 1:
            # Normalize rewards to [0, 1] for AULC
            r_min = rewards.min()
            r_max = rewards.max()
            r_range = r_max - r_min
            if r_range > 0:
                normalized = (moving_avg - r_min) / r_range
            else:
                normalized = np.ones_like(moving_avg) * 0.5
            trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            aulc = float(trapz_func(normalized) / len(normalized))
        else:
            aulc = 0.0

        return {
            "steps_to_threshold": steps_to_thresh,
            "reached_threshold": reached,
            "aulc": aulc,
            "episodes_to_threshold": eps_to_thresh,
        }

    def compute_stability_metrics(self, history: Dict) -> Dict[str, float]:
        """Compute within-run training stability metrics."""
        rewards = history.get("episode_rewards", [])
        if len(rewards) < 2:
            return {
                "reward_variance": 0.0,
                "reward_std": 0.0,
                "max_drawdown": 0.0,
                "reward_fluctuation": 0.0,
            }

        rewards = np.array([float(r) for r in rewards], dtype=np.float64)

        # Moving average for stability
        window = min(20, len(rewards) // 2) or 1
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        else:
            moving_avg = rewards

        # Compute max drawdown from moving average
        peak = moving_avg[0]
        max_dd = 0
        for val in moving_avg:
            if val > peak:
                peak = val
            dd = (peak - val) / (abs(peak) + 1e-8)
            if dd > max_dd:
                max_dd = dd

        # Reward fluctuation (coefficient of variation)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        cv = std_r / (abs(mean_r) + 1e-8)

        return {
            "reward_variance": float(np.var(rewards)),
            "reward_std": float(std_r),
            "max_drawdown": float(max_dd),
            "reward_fluctuation": float(cv),
        }

    def compute_quantum_metrics(self, result: Dict) -> Dict[str, Any]:
        """Extract and compute quantum-specific metrics."""
        agent_info = result.get("agent_info", {})
        if not agent_info.get("is_quantum", False):
            return {}

        quantum_info = agent_info.get("quantum_info", {})
        history = result.get("history", {})

        q_grad_norms = history.get("quantum_gradient_norm", [])
        q_grad_means = history.get("quantum_gradient_mean", [])
        q_grad_vars = history.get("quantum_gradient_var", [])

        metrics = {
            "n_qubits": quantum_info.get("n_qubits", 4),
            "circuit_depth": quantum_info.get("circuit_depth", 0),
            "n_params": quantum_info.get("n_params", 0),
            "n_entangling_gates": quantum_info.get("n_entangling_gates", 0) * quantum_info.get("circuit_depth", 0),
            "total_gates": quantum_info.get("total_gates", 0),
            "shots": quantum_info.get("shots", None),
            "differentiation_method": quantum_info.get("differentiation_method", "backprop"),
            "device": quantum_info.get("device", "lightning.qubit"),
            "quantum_evaluations": agent_info.get("quantum_evaluations", 0),
            "quantum_time": agent_info.get("quantum_time", 0),
        }

        if q_grad_norms:
            metrics["mean_gradient_norm"] = float(np.mean(q_grad_norms))
            metrics["max_gradient_norm"] = float(np.max(q_grad_norms))
            metrics["min_gradient_norm"] = float(np.min(q_grad_norms))
            metrics["final_gradient_norm"] = float(q_grad_norms[-1])
        if q_grad_means:
            metrics["mean_gradient_mean"] = float(np.mean(q_grad_means))
        if q_grad_vars:
            metrics["mean_gradient_var"] = float(np.mean(q_grad_vars))
            metrics["final_gradient_var"] = float(q_grad_vars[-1])

        return metrics

    def evaluate_all(self, env_specs: Dict, performance_thresholds: Dict) -> Dict[str, Any]:
        """
        Comprehensive evaluation of all experiments.
        Returns aggregated evaluation data.
        """
        self.load_all_results()

        evaluation = {
            "experiments": {},
            "by_environment": {},
            "by_algorithm": {},
        }

        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue

            algo = result["algorithm"]
            env = result["environment"]
            history = result.get("history", {})
            eval_results = result.get("eval_results", {})

            # Sample efficiency
            threshold = performance_thresholds.get(env, 0)
            sample_eff = self.compute_sample_efficiency(history, threshold)

            # Stability
            stability = self.compute_stability_metrics(history)

            # Quantum metrics
            quantum_metrics = self.compute_quantum_metrics(result)

            # Computational metrics
            agent_info = result.get("agent_info", {})
            compute_metrics = {
                "total_training_time": agent_info.get("total_training_time", 0),
                "n_parameters": agent_info.get("n_parameters", 0),
                "quantum_time": agent_info.get("quantum_time", 0),
            }

            experiment_eval = {
                "algorithm": algo,
                "environment": env,
                "performance": eval_results,
                "sample_efficiency": sample_eff,
                "stability": stability,
                "quantum_metrics": quantum_metrics,
                "compute_metrics": compute_metrics,
                "total_steps": result.get("total_steps", 0),
                "episodes": result.get("episodes", 0),
            }

            evaluation["experiments"][key] = experiment_eval

            # Group by environment
            if env not in evaluation["by_environment"]:
                evaluation["by_environment"][env] = {}
            evaluation["by_environment"][env][algo] = experiment_eval

            # Group by algorithm
            if algo not in evaluation["by_algorithm"]:
                evaluation["by_algorithm"][algo] = {}
            evaluation["by_algorithm"][algo][env] = experiment_eval

        return evaluation
