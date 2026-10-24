"""
Comprehensive metrics computation for the benchmark.

Aggregates all experiment results, computes cross-experiment metrics,
and generates summary tables.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List


class MetricsComputer:
    """
    Computes and aggregates all benchmark metrics.

    Loads all experiment results and evaluation data, then produces
    comprehensive metrics tables and summaries.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.all_results = {}
        self.all_evaluations = {}

    def load_results(self) -> Dict[str, Dict]:
        """Load all experiment results."""
        for f in os.listdir(self.data_dir):
            if f.endswith("_results.json"):
                path = os.path.join(self.data_dir, f)
                with open(path, "r") as fh:
                    data = json.load(fh)
                key = f.replace("_results.json", "")
                self.all_results[key] = data
        return self.all_results

    def compute_all_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive metrics across all experiments."""
        self.load_results()

        metrics = {
            "summary_table": self._compute_summary_table(),
            "performance_comparison": self._compute_performance_comparison(),
            "sample_efficiency_table": self._compute_sample_efficiency_table(),
            "stability_table": self._compute_stability_table(),
            "computational_cost_table": self._compute_computational_cost_table(),
            "quantum_metrics_table": self._compute_quantum_metrics_table(),
            "ppo_vs_qppo": self._compute_ppo_vs_qppo(),
            "dqn_vs_qdqn": self._compute_dqn_vs_qdqn(),
            "sac_vs_qsac": self._compute_sac_vs_qsac(),
            "complexity_analysis": self._compute_complexity_analysis(),
            "final_summary": self._compute_final_summary(),
        }

        # Save all metrics
        metrics_path = os.path.join(self.data_dir, "all_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        return metrics

    def _compute_summary_table(self) -> List[Dict]:
        """Main summary table of all experiments."""
        rows = []
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            eval_r = result.get("eval_results", {})
            agent_info = result.get("agent_info", {})
            rows.append({
                "algorithm": result["algorithm"],
                "environment": result["environment"],
                "mean_return": eval_r.get("mean_return", 0),
                "std_return": eval_r.get("std_return", 0),
                "best_return": eval_r.get("max_return", 0),
                "median_return": eval_r.get("median_return", 0),
                "total_steps": result.get("total_steps", 0),
                "episodes": result.get("episodes", 0),
                "training_time_s": result.get("training_time", 0),
                "n_parameters": agent_info.get("n_parameters", 0),
                "is_quantum": agent_info.get("is_quantum", False),
                "configuration_regime": result.get("configuration_regime", result.get("config", {}).get("configuration_regime", "original")),
            })
        return sorted(rows, key=lambda x: (x["environment"], x["algorithm"]))

    def _compute_performance_comparison(self) -> Dict[str, Any]:
        """Compare classical vs quantum performance per environment."""
        comparisons = {}
        envs = set()
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            envs.add(result["environment"])

        for env in envs:
            env_results = {}
            for key, result in self.all_results.items():
                if result.get("status") != "completed":
                    continue
                if result["environment"] != env:
                    continue
                algo = result["algorithm"]
                eval_r = result.get("eval_results", {})
                env_results[algo] = {
                    "mean_return": eval_r.get("mean_return", 0),
                    "is_quantum": result.get("agent_info", {}).get("is_quantum", False),
                }
            comparisons[env] = env_results

        return comparisons

    def _compute_sample_efficiency_table(self) -> List[Dict]:
        """Sample efficiency metrics for all experiments."""
        rows = []
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            history = result.get("history", {})
            ep_rewards = history.get("episode_rewards", [])
            total_steps = history.get("total_steps", [])

            # Simple AULC
            if len(ep_rewards) > 1:
                rewards = np.array([float(r) for r in ep_rewards], dtype=np.float64)
                window = min(20, len(rewards) // 2) or 1
                if window > 1:
                    ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
                else:
                    ma = rewards
                r_min, r_max = rewards.min(), rewards.max()
                if r_max - r_min > 0:
                    norm = (ma - r_min) / (r_max - r_min)
                else:
                    norm = np.ones_like(ma) * 0.5
                trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
                aulc = float(trapz_func(norm) / len(norm))
            else:
                aulc = 0.0

            rows.append({
                "algorithm": result["algorithm"],
                "environment": result["environment"],
                "aulc": aulc,
                "episodes": result.get("episodes", 0),
                "total_steps": result.get("total_steps", 0),
            })
        return rows

    def _compute_stability_table(self) -> List[Dict]:
        """Training stability metrics."""
        rows = []
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            history = result.get("history", {})
            rewards = history.get("episode_rewards", [])

            if len(rewards) < 2:
                rows.append({
                    "algorithm": result["algorithm"],
                    "environment": result["environment"],
                    "reward_std": 0,
                    "reward_fluctuation": 0,
                    "max_drawdown": 0,
                })
                continue

            rewards = np.array([float(r) for r in rewards], dtype=np.float64)
            std = float(np.std(rewards))
            cv = std / (abs(np.mean(rewards)) + 1e-8)

            # Max drawdown
            window = min(20, len(rewards) // 2) or 1
            if window > 1:
                ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
            else:
                ma = rewards
            peak = ma[0]
            max_dd = 0
            for val in ma:
                if val > peak:
                    peak = val
                dd = (peak - val) / (abs(peak) + 1e-8)
                if dd > max_dd:
                    max_dd = dd

            rows.append({
                "algorithm": result["algorithm"],
                "environment": result["environment"],
                "reward_std": float(std),
                "reward_fluctuation": float(cv),
                "max_drawdown": float(max_dd),
            })
        return rows

    def _compute_computational_cost_table(self) -> List[Dict]:
        """Computational cost metrics."""
        rows = []
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            agent_info = result.get("agent_info", {})
            rows.append({
                "algorithm": result["algorithm"],
                "environment": result["environment"],
                "training_time_s": agent_info.get("total_training_time", 0),
                "n_parameters": agent_info.get("n_parameters", 0),
                "is_quantum": agent_info.get("is_quantum", False),
                "quantum_time_s": agent_info.get("quantum_time", 0),
            })
        return rows

    def _compute_quantum_metrics_table(self) -> List[Dict]:
        """Quantum-specific metrics for quantum experiments."""
        rows = []
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            if not result.get("agent_info", {}).get("is_quantum", False):
                continue
            agent_info = result.get("agent_info", {})
            q_info = agent_info.get("quantum_info", {})
            history = result.get("history", {})
            q_grads = history.get("quantum_gradient_norm", [])

            row = {
                "algorithm": result["algorithm"],
                "environment": result["environment"],
                "n_qubits": q_info.get("n_qubits", 4),
                "circuit_depth": q_info.get("circuit_depth", 0),
                "n_params": q_info.get("n_params", 0),
                "total_gates": q_info.get("total_gates", 0),
                "quantum_evaluations": agent_info.get("quantum_evaluations", 0),
            }
            if q_grads:
                row["mean_grad_norm"] = float(np.mean(q_grads))
                row["final_grad_norm"] = float(q_grads[-1])
            rows.append(row)
        return rows

    def _compute_ppo_vs_qppo(self) -> Dict[str, Any]:
        """Direct PPO vs Q-PPO comparison."""
        comparison = {}
        envs = set()
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            envs.add(result["environment"])

        for env in envs:
            ppo_result = None
            qppo_result = None
            for key, result in self.all_results.items():
                if result.get("environment") != env or result.get("status") != "completed":
                    continue
                if result["algorithm"] == "PPO":
                    ppo_result = result
                elif result["algorithm"] == "Q-PPO":
                    qppo_result = result

            if ppo_result and qppo_result:
                ppo_ret = ppo_result["eval_results"]["mean_return"]
                qppo_ret = qppo_result["eval_results"]["mean_return"]
                comparison[env] = {
                    "ppo_return": ppo_ret,
                    "qppo_return": qppo_ret,
                    "difference": qppo_ret - ppo_ret,
                    "relative_difference": (qppo_ret - ppo_ret) / (abs(ppo_ret) + 1e-8),
                    "ppo_wins": ppo_ret > qppo_ret,
                    "qppo_wins": qppo_ret > ppo_ret,
                }
        return comparison

    def _compute_dqn_vs_qdqn(self) -> Dict[str, Any]:
        """Direct DQN vs Q-DQN comparison."""
        comparison = {}
        envs = set()
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            envs.add(result["environment"])

        for env in envs:
            dqn_result = None
            qdqn_result = None
            for key, result in self.all_results.items():
                if result.get("environment") != env or result.get("status") != "completed":
                    continue
                if result["algorithm"] == "DQN":
                    dqn_result = result
                elif result["algorithm"] == "Q-DQN":
                    qdqn_result = result

            if dqn_result and qdqn_result:
                dqn_ret = dqn_result["eval_results"]["mean_return"]
                qdqn_ret = qdqn_result["eval_results"]["mean_return"]
                comparison[env] = {
                    "dqn_return": dqn_ret,
                    "qdqn_return": qdqn_ret,
                    "difference": qdqn_ret - dqn_ret,
                    "relative_difference": (qdqn_ret - dqn_ret) / (abs(dqn_ret) + 1e-8),
                    "dqn_wins": dqn_ret > qdqn_ret,
                    "qdqn_wins": qdqn_ret > dqn_ret,
                }
        return comparison

    def _compute_sac_vs_qsac(self) -> Dict[str, Any]:
        """Direct SAC vs Q-SAC comparison."""
        comparison = {}
        envs = set()
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            envs.add(result["environment"])

        for env in envs:
            sac_result = None
            qsac_result = None
            for key, result in self.all_results.items():
                if result.get("environment") != env or result.get("status") != "completed":
                    continue
                if result["algorithm"] == "SAC":
                    sac_result = result
                elif result["algorithm"] == "Q-SAC":
                    qsac_result = result

            if sac_result and qsac_result:
                sac_ret = sac_result["eval_results"]["mean_return"]
                qsac_ret = qsac_result["eval_results"]["mean_return"]
                comparison[env] = {
                    "sac_return": sac_ret,
                    "qsac_return": qsac_ret,
                    "difference": qsac_ret - sac_ret,
                    "relative_difference": (qsac_ret - sac_ret) / (abs(sac_ret) + 1e-8),
                    "sac_wins": sac_ret > qsac_ret,
                    "qsac_wins": qsac_ret > sac_ret,
                }
        return comparison

    def _compute_complexity_analysis(self) -> Dict[str, Any]:
        """Analyze performance changes across environment complexity."""
        complexity_levels = {
            "CartPole-v1": 1, "MountainCar-v0": 2, "LunarLander-v3": 3,
            "Pendulum-v1": 4, "BipedalWalker-v3": 5,
        }

        algo_perfs = {}
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            algo = result["algorithm"]
            env = result["environment"]
            if algo not in algo_perfs:
                algo_perfs[algo] = []
            algo_perfs[algo].append({
                "complexity": complexity_levels.get(env, 0),
                "environment": env,
                "mean_return": result["eval_results"]["mean_return"],
                "training_time": result.get("training_time", 0),
            })

        # Sort each algorithm by complexity
        for algo in algo_perfs:
            algo_perfs[algo] = sorted(algo_perfs[algo], key=lambda x: x["complexity"])

        return algo_perfs

    def _compute_final_summary(self) -> Dict[str, Any]:
        """Compute the final research summary."""
        summary = {
            "best_per_environment": {},
            "classical_vs_quantum": {},
            "ppo_vs_qppo_summary": {},
            "dqn_vs_qdqn_summary": {},
            "sac_vs_qsac_summary": {},
            "most_sample_efficient": {},
            "limitations": [],
        }

        # Best per environment
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            env = result["environment"]
            algo = result["algorithm"]
            mean_ret = result["eval_results"]["mean_return"]
            if env not in summary["best_per_environment"] or \
               mean_ret > summary["best_per_environment"][env]["mean_return"]:
                summary["best_per_environment"][env] = {
                    "algorithm": algo,
                    "mean_return": mean_ret,
                }

        # Classical vs quantum overall
        classical_returns = []
        quantum_returns = []
        for key, result in self.all_results.items():
            if result.get("status") != "completed":
                continue
            mean_ret = result["eval_results"]["mean_return"]
            if result.get("agent_info", {}).get("is_quantum", False):
                quantum_returns.append(mean_ret)
            else:
                classical_returns.append(mean_ret)

        if classical_returns:
            summary["classical_vs_quantum"]["classical_mean"] = float(np.mean(classical_returns))
        if quantum_returns:
            summary["classical_vs_quantum"]["quantum_mean"] = float(np.mean(quantum_returns))

        # Limitations
        summary["limitations"] = [
            "Single random seed (no multi-run statistical variance estimates)",
            "Classical simulation of quantum circuits (no real quantum hardware)",
            "Fixed 4-qubit configuration",
            "Lightning-Qubit simulator overhead",
            "Five benchmark environments",
            "Fixed quantum circuit architecture",
            "Limited hyperparameter search",
            "No real quantum hardware execution",
        ]

        return summary
