"""
Complete visualization suite for the Quantum RL Benchmark.

Generates all required visualizations from actual saved experiment results.
All figures are saved in high resolution (PNG + PDF).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from typing import Dict, Any, List, Optional


# Color scheme
COLORS = {
    "PPO": "#2196F3",
    "Q-PPO": "#E91E63",
    "DQN": "#4CAF50",
    "Q-DQN": "#FF9800",
    "SAC": "#9C27B0",
    "Q-SAC": "#F44336",
}

MARKERS = {
    "PPO": "o",
    "Q-PPO": "s",
    "DQN": "^",
    "Q-DQN": "D",
    "SAC": "v",
    "Q-SAC": "P",
}


class VisualizationSuite:
    """
    Generates all visualizations for the benchmark.
    """

    def __init__(self, fig_dir: str, data_dir: str):
        self.fig_dir = fig_dir
        self.data_dir = data_dir
        self.all_results = {}
        self.metrics = {}

    def load_data(self):
        """Load all experiment results and metrics."""
        for f in os.listdir(self.data_dir):
            if f.endswith("_results.json"):
                path = os.path.join(self.data_dir, f)
                with open(path, "r") as fh:
                    data = json.load(fh)
                key = f.replace("_results.json", "")
                self.all_results[key] = data

        metrics_path = os.path.join(self.data_dir, "all_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as fh:
                self.metrics = json.load(fh)

    def generate_all(self):
        """Generate all visualizations."""
        self.load_data()
        print("  Generating visualizations...")

        self.plot_learning_curves()
        self.plot_final_performance()
        self.plot_sample_efficiency()
        self.plot_aulc()
        self.plot_complexity_scaling()
        self.plot_performance_heatmap()
        self.plot_runtime_vs_performance()
        self.plot_quantum_resources()
        self.plot_quantum_gradients()
        self.plot_algorithm_ranking()
        self.plot_training_stability()

        print("  All visualizations generated.")

    def _smooth(self, data, window=20):
        """Apply moving average smoothing."""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def _get_completed(self) -> List[Dict]:
        """Get all completed experiment results."""
        return [r for r in self.all_results.values() if r.get("status") == "completed"]

    def plot_learning_curves(self):
        """V1: Learning curves for every environment."""
        envs = sorted(set(r["environment"] for r in self._get_completed()))

        for env in envs:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            env_results = [r for r in self._get_completed() if r["environment"] == env]

            for result in env_results:
                algo = result["algorithm"]
                history = result.get("history", {})
                ep_rewards = history.get("episode_rewards", [])
                total_steps = history.get("total_steps", [])

                if not ep_rewards or not total_steps:
                    continue

                steps = np.array([float(s) for s in total_steps[:len(ep_rewards)]], dtype=np.float64)
                rewards = np.array([float(r) for r in ep_rewards], dtype=np.float64)

                # Smooth
                window = min(20, max(1, len(rewards) // 5))
                if window > 1 and len(rewards) > window:
                    smoothed = self._smooth(rewards, window)
                    steps_plot = steps[window - 1:]
                else:
                    smoothed = rewards
                    steps_plot = steps

                ax.plot(steps_plot, smoothed, label=algo,
                        color=COLORS.get(algo, "gray"),
                        marker=MARKERS.get(algo, "o"),
                        markevery=max(1, len(steps_plot) // 10),
                        linewidth=2, markersize=5, alpha=0.9)

                # Light raw data
                ax.plot(steps, rewards, color=COLORS.get(algo, "gray"),
                        alpha=0.1, linewidth=0.5)

            ax.set_xlabel("Environment Steps", fontsize=12)
            ax.set_ylabel("Episode Return", fontsize=12)
            ax.set_title(f"Learning Curves — {env}", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dir, f"learning_curve_{env}.png"),
                       dpi=200, bbox_inches="tight")
            plt.savefig(os.path.join(self.fig_dir, f"learning_curve_{env}.pdf"),
                       bbox_inches="tight")
            plt.close()

        print("    [OK] Learning curves")

    def plot_final_performance(self):
        """V2: Final performance comparison for each environment."""
        envs = sorted(set(r["environment"] for r in self._get_completed()))

        for env in envs:
            env_results = [r for r in self._get_completed() if r["environment"] == env]
            algos = [r["algorithm"] for r in env_results]
            returns = [r["eval_results"]["mean_return"] for r in env_results]

            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            bars = ax.bar(algos, returns,
                         color=[COLORS.get(a, "gray") for a in algos],
                         edgecolor="black", linewidth=0.5)

            # Add value labels
            for bar, val in zip(bars, returns):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(abs(min(returns)), abs(max(returns))),
                       f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

            ax.set_ylabel("Mean Evaluation Return", fontsize=12)
            ax.set_title(f"Final Performance — {env}", fontsize=14, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dir, f"final_performance_{env}.png"),
                       dpi=200, bbox_inches="tight")
            plt.savefig(os.path.join(self.fig_dir, f"final_performance_{env}.pdf"),
                       bbox_inches="tight")
            plt.close()

        print("    [OK] Final performance charts")

    def plot_sample_efficiency(self):
        """V3: Sample efficiency — performance vs environment steps."""
        envs = sorted(set(r["environment"] for r in self._get_completed()))

        for env in envs:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            env_results = [r for r in self._get_completed() if r["environment"] == env]

            for result in env_results:
                algo = result["algorithm"]
                history = result.get("history", {})
                ep_rewards = history.get("episode_rewards", [])
                total_steps = history.get("total_steps", [])

                if not ep_rewards:
                    continue

                steps = np.array([float(s) for s in total_steps[:len(ep_rewards)]], dtype=np.float64)
                rewards = np.array([float(r) for r in ep_rewards], dtype=np.float64)

                # Cumulative best reward
                cumulative_best = np.maximum.accumulate(rewards)

                window = min(10, max(1, len(cumulative_best) // 5))
                if window > 1 and len(cumulative_best) > window:
                    smoothed = self._smooth(cumulative_best, window)
                    steps_plot = steps[window - 1:]
                else:
                    smoothed = cumulative_best
                    steps_plot = steps

                ax.plot(steps_plot, smoothed, label=algo,
                        color=COLORS.get(algo, "gray"),
                        linewidth=2, alpha=0.9)

            ax.set_xlabel("Environment Steps", fontsize=12)
            ax.set_ylabel("Best Episode Return (Cumulative)", fontsize=12)
            ax.set_title(f"Sample Efficiency — {env}", fontsize=14, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dir, f"sample_efficiency_{env}.png"),
                       dpi=200, bbox_inches="tight")
            plt.close()

        print("    [OK] Sample efficiency charts")

    def plot_aulc(self):
        """V4: Area Under Learning Curve comparison."""
        data = self.metrics.get("sample_efficiency_table", [])
        if not data:
            # Compute from results
            data = []
            for key, result in self._get_completed().__iter__():
                pass
            for result in self._get_completed():
                history = result.get("history", {})
                ep_rewards = history.get("episode_rewards", [])
                if len(ep_rewards) > 1:
                    rewards = np.array([float(r) for r in ep_rewards], dtype=np.float64)
                    window = min(20, max(1, len(rewards) // 5))
                    if window > 1 and len(rewards) > window:
                        ma = self._smooth(rewards, window)
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
                    aulc = 0.5
                data.append({
                    "algorithm": result["algorithm"],
                    "environment": result["environment"],
                    "aulc": aulc,
                })

        envs = sorted(set(d["environment"] for d in data))
        algos = sorted(set(d["algorithm"] for d in data))

        fig, axes = plt.subplots(1, len(envs), figsize=(4 * len(envs), 5), sharey=True)
        if len(envs) == 1:
            axes = [axes]

        for i, env in enumerate(envs):
            env_data = {d["algorithm"]: d["aulc"] for d in data if d["environment"] == env}
            vals = [env_data.get(a, 0) for a in algos]
            colors = [COLORS.get(a, "gray") for a in algos]

            axes[i].bar(algos, vals, color=colors, edgecolor="black", linewidth=0.5)
            axes[i].set_title(env, fontsize=10, fontweight="bold")
            axes[i].tick_params(axis="x", rotation=45, labelsize=8)
            axes[i].grid(True, alpha=0.3, axis="y")

        axes[0].set_ylabel("AULC (Normalized)", fontsize=11)
        fig.suptitle("Area Under Learning Curve", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, "aulc_comparison.png"),
                   dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(self.fig_dir, "aulc_comparison.pdf"),
                   bbox_inches="tight")
        plt.close()

        print("    [OK] AULC comparison")

    def plot_complexity_scaling(self):
        """V5: Complexity scaling — normalized performance across environments."""
        complexity_levels = {
            "CartPole-v1": 1, "MountainCar-v0": 2, "LunarLander-v3": 3,
            "Pendulum-v1": 4, "BipedalWalker-v3": 5,
        }

        # Group by algorithm, sorted by complexity
        algo_data = {}
        for result in self._get_completed():
            algo = result["algorithm"]
            env = result["environment"]
            if algo not in algo_data:
                algo_data[algo] = []
            algo_data[algo].append({
                "complexity": complexity_levels.get(env, 0),
                "environment": env,
                "mean_return": result["eval_results"]["mean_return"],
            })

        # Normalize returns per environment for cross-env comparison
        env_returns = {}
        for result in self._get_completed():
            env = result["environment"]
            if env not in env_returns:
                env_returns[env] = []
            env_returns[env].append(result["eval_results"]["mean_return"])

        env_mins = {e: min(r) for e, r in env_returns.items()}
        env_maxs = {e: max(r) for e, r in env_returns.items()}

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))

        discrete_range = (0.5, 3.5)
        continuous_range = (3.5, 5.5)
        ax.axvspan(discrete_range[0], discrete_range[1], alpha=0.05, color="blue",
                   label="Discrete Control")
        ax.axvspan(continuous_range[0], continuous_range[1], alpha=0.05, color="red",
                   label="Continuous Control")

        for algo, data_points in algo_data.items():
            data_points.sort(key=lambda x: x["complexity"])
            complexities = [d["complexity"] for d in data_points]
            normalized = []
            for d in data_points:
                env = d["environment"]
                r_min, r_max = env_mins[env], env_maxs[env]
                if r_max - r_min > 0:
                    norm = (d["mean_return"] - r_min) / (r_max - r_min)
                else:
                    norm = 0.5
                normalized.append(norm)

            ax.plot(complexities, normalized, label=algo,
                    color=COLORS.get(algo, "gray"),
                    marker=MARKERS.get(algo, "o"),
                    linewidth=2, markersize=8, alpha=0.9)

        ax.set_xlabel("Environment Complexity Level", fontsize=12)
        ax.set_ylabel("Normalized Performance", fontsize=12)
        ax.set_title("Performance vs Environment Complexity", fontsize=14, fontweight="bold")
        ax.set_xticks(range(1, 6))
        ax.set_xticklabels([
            "CartPole\n(v1)", "MountainCar\n(v0)", "LunarLander\n(v3)",
            "Pendulum\n(v1)", "BipedalWalker\n(v3)"
        ], fontsize=9)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

        # Add transition line
        ax.axvline(x=3.5, color="gray", linestyle="--", alpha=0.5)
        ax.text(3.5, ax.get_ylim()[1] * 0.95, "Discrete → Continuous",
               ha="center", fontsize=9, color="gray", style="italic")

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, "complexity_scaling.png"),
                   dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(self.fig_dir, "complexity_scaling.pdf"),
                   bbox_inches="tight")
        plt.close()

        print("    [OK] Complexity scaling plot")

    def plot_performance_heatmap(self):
        """V6: Algorithm × environment performance heatmap."""
        completed = self._get_completed()
        if not completed:
            return

        envs = sorted(set(r["environment"] for r in completed))
        algos = sorted(set(r["algorithm"] for r in completed))

        # Build matrix
        returns_matrix = np.zeros((len(algos), len(envs)))
        for r in completed:
            a_idx = algos.index(r["algorithm"])
            e_idx = envs.index(r["environment"])
            returns_matrix[a_idx, e_idx] = r["eval_results"]["mean_return"]

        # Normalize per environment (min-max)
        env_mins = returns_matrix.min(axis=0)
        env_maxs = returns_matrix.max(axis=0)
        env_ranges = env_maxs - env_mins
        env_ranges[env_ranges == 0] = 1
        normalized = (returns_matrix - env_mins) / env_ranges

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Raw returns
        im1 = ax1.imshow(returns_matrix, aspect="auto", cmap="RdYlGn")
        ax1.set_xticks(range(len(envs)))
        ax1.set_xticklabels([e.replace("-", "\n") for e in envs], fontsize=8)
        ax1.set_yticks(range(len(algos)))
        ax1.set_yticklabels(algos, fontsize=10)
        ax1.set_title("Raw Mean Returns", fontsize=12, fontweight="bold")
        plt.colorbar(im1, ax=ax1, shrink=0.8)

        for i in range(len(algos)):
            for j in range(len(envs)):
                ax1.text(j, i, f"{returns_matrix[i, j]:.0f}",
                        ha="center", va="center", fontsize=8, fontweight="bold")

        # Normalized
        im2 = ax2.imshow(normalized, aspect="auto", cmap="RdYlGn")
        ax2.set_xticks(range(len(envs)))
        ax2.set_xticklabels([e.replace("-", "\n") for e in envs], fontsize=8)
        ax2.set_yticks(range(len(algos)))
        ax2.set_yticklabels(algos, fontsize=10)
        ax2.set_title("Normalized Performance", fontsize=12, fontweight="bold")
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        for i in range(len(algos)):
            for j in range(len(envs)):
                ax2.text(j, i, f"{normalized[i, j]:.2f}",
                        ha="center", va="center", fontsize=8, fontweight="bold")

        fig.suptitle("Performance Heatmap: Algorithm × Environment",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, "performance_heatmap.png"),
                   dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(self.fig_dir, "performance_heatmap.pdf"),
                   bbox_inches="tight")
        plt.close()

        # Training time heatmap
        time_matrix = np.zeros((len(algos), len(envs)))
        for r in completed:
            a_idx = algos.index(r["algorithm"])
            e_idx = envs.index(r["environment"])
            time_matrix[a_idx, e_idx] = r.get("training_time", 0)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(time_matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(envs)))
        ax.set_xticklabels([e.replace("-", "\n") for e in envs], fontsize=8)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels(algos, fontsize=10)
        ax.set_title("Training Time (seconds)", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(algos)):
            for j in range(len(envs)):
                ax.text(j, i, f"{time_matrix[i, j]:.0f}s",
                       ha="center", va="center", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, "training_time_heatmap.png"),
                   dpi=200, bbox_inches="tight")
        plt.close()

        print("    [OK] Performance heatmaps")

    def plot_runtime_vs_performance(self):
        """V7: Runtime vs performance scatter plot."""
        completed = self._get_completed()

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        for result in completed:
            algo = result["algorithm"]
            env = result["environment"]
            runtime = result.get("training_time", 0)
            perf = result["eval_results"]["mean_return"]

            ax.scatter(runtime, perf, color=COLORS.get(algo, "gray"),
                      marker=MARKERS.get(algo, "o"), s=120, alpha=0.8,
                      edgecolors="black", linewidth=0.5, zorder=5)
            ax.annotate(f"{algo}\n{env}", (runtime, perf),
                       fontsize=7, ha="center", va="bottom",
                       xytext=(0, 8), textcoords="offset points")

        ax.set_xlabel("Training Time (seconds)", fontsize=12)
        ax.set_ylabel("Mean Evaluation Return", fontsize=12)
        ax.set_title("Training Time vs Performance", fontsize=14, fontweight="bold")

        # Add legend
        for algo in sorted(set(r["algorithm"] for r in completed)):
            ax.scatter([], [], color=COLORS.get(algo, "gray"),
                      marker=MARKERS.get(algo, "o"), s=80, label=algo, edgecolors="black")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, "runtime_vs_performance.png"),
                   dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(self.fig_dir, "runtime_vs_performance.pdf"),
                   bbox_inches="tight")
        plt.close()

        print("    [OK] Runtime vs performance")

    def plot_quantum_resources(self):
        """V8: Quantum-specific resource visualizations."""
        quantum_results = [r for r in self._get_completed()
                          if r.get("agent_info", {}).get("is_quantum", False)]

        if not quantum_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Circuit depth vs performance
        ax = axes[0, 0]
        for r in quantum_results:
            q_info = r.get("agent_info", {}).get("quantum_info", {})
            depth = q_info.get("circuit_depth", 0)
            perf = r["eval_results"]["mean_return"]
            ax.scatter(depth, perf, color=COLORS.get(r["algorithm"], "gray"),
                      marker=MARKERS.get(r["algorithm"], "o"), s=100,
                      edgecolors="black", linewidth=0.5)
            ax.annotate(f"{r['algorithm']}\n{r['environment']}", (depth, perf),
                       fontsize=7, ha="center", va="bottom",
                       xytext=(0, 8), textcoords="offset points")
        ax.set_xlabel("Circuit Depth")
        ax.set_ylabel("Mean Return")
        ax.set_title("Circuit Depth vs Performance")
        ax.grid(True, alpha=0.3)

        # Circuit evaluations vs runtime
        ax = axes[0, 1]
        for r in quantum_results:
            agent_info = r.get("agent_info", {})
            q_evals = agent_info.get("quantum_evaluations", 0)
            runtime = r.get("training_time", 0)
            ax.scatter(q_evals, runtime, color=COLORS.get(r["algorithm"], "gray"),
                      marker=MARKERS.get(r["algorithm"], "o"), s=100,
                      edgecolors="black", linewidth=0.5)
            ax.annotate(r["algorithm"], (q_evals, runtime), fontsize=7)
        ax.set_xlabel("Quantum Circuit Evaluations")
        ax.set_ylabel("Total Training Time (s)")
        ax.set_title("Circuit Evaluations vs Runtime")
        ax.grid(True, alpha=0.3)

        # Quantum runtime vs total runtime
        ax = axes[1, 0]
        for r in quantum_results:
            agent_info = r.get("agent_info", {})
            q_time = agent_info.get("quantum_time", 0)
            total_time = r.get("training_time", 0)
            ax.scatter(q_time, total_time, color=COLORS.get(r["algorithm"], "gray"),
                      marker=MARKERS.get(r["algorithm"], "o"), s=100,
                      edgecolors="black", linewidth=0.5)
            ax.annotate(f"{r['algorithm']}\n{r['environment']}", (q_time, total_time),
                       fontsize=7, ha="center", va="bottom",
                       xytext=(0, 8), textcoords="offset points")
        ax.set_xlabel("Quantum Layer Time (s)")
        ax.set_ylabel("Total Training Time (s)")
        ax.set_title("Quantum Runtime vs Total Runtime")
        ax.grid(True, alpha=0.3)

        # 4-qubit performance by algorithm
        ax = axes[1, 1]
        algo_perf = {}
        for r in quantum_results:
            algo = r["algorithm"]
            if algo not in algo_perf:
                algo_perf[algo] = []
            algo_perf[algo].append(r["eval_results"]["mean_return"])

        algos = sorted(algo_perf.keys())
        means = [np.mean(algo_perf[a]) for a in algos]
        colors = [COLORS.get(a, "gray") for a in algos]

        ax.bar(algos, means, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Mean Return (across envs)")
        ax.set_title("4-Qubit Quantum Model Performance")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Quantum Resource Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, "quantum_resources.png"),
                   dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(self.fig_dir, "quantum_resources.pdf"),
                   bbox_inches="tight")
        plt.close()

        print("    [OK] Quantum resource analysis")

    def plot_quantum_gradients(self):
        """V9: Quantum gradient behavior during training."""
        quantum_results = [r for r in self._get_completed()
                          if r.get("agent_info", {}).get("is_quantum", False)]

        if not quantum_results:
            return

        n = len(quantum_results)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

        for i, result in enumerate(quantum_results):
            row, col = i // cols, i % cols
            ax = axes[row][col]
            history = result.get("history", {})
            grad_norms = history.get("quantum_gradient_norm", [])

            if grad_norms:
                steps = range(len(grad_norms))
                ax.plot(steps, grad_norms, color=COLORS.get(result["algorithm"], "gray"),
                       linewidth=1.5, alpha=0.8)
                ax.set_xlabel("Update Step")
                ax.set_ylabel("Gradient Norm")
                ax.set_title(f"{result['algorithm']} — {result['environment']}",
                           fontsize=10, fontweight="bold")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No gradient data", ha="center", va="center",
                       transform=ax.transAxes)
                ax.set_title(f"{result['algorithm']} — {result['environment']}")

        # Hide empty subplots
        for i in range(n, rows * cols):
            row, col = i // cols, i % cols
            axes[row][col].set_visible(False)

        fig.suptitle("Quantum Gradient Norm During Training",
                    fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, "quantum_gradients.png"),
                   dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(self.fig_dir, "quantum_gradients.pdf"),
                   bbox_inches="tight")
        plt.close()

        print("    [OK] Quantum gradient analysis")

    def plot_algorithm_ranking(self):
        """V10: Algorithm rankings by various criteria."""
        completed = self._get_completed()
        if not completed:
            return

        algos = sorted(set(r["algorithm"] for r in completed))

        # Aggregate metrics per algorithm
        algo_metrics = {}
        for algo in algos:
            algo_results = [r for r in completed if r["algorithm"] == algo]
            returns = [r["eval_results"]["mean_return"] for r in algo_results]
            times = [r.get("training_time", 0) for r in algo_results]
            steps = [r.get("total_steps", 0) for r in algo_results]

            algo_metrics[algo] = {
                "mean_return": np.mean(returns),
                "total_time": sum(times),
                "mean_steps": np.mean(steps) if steps else 0,
            }

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Performance ranking
        ax = axes[0]
        sorted_algos = sorted(algos, key=lambda a: algo_metrics[a]["mean_return"], reverse=True)
        vals = [algo_metrics[a]["mean_return"] for a in sorted_algos]
        ax.barh(sorted_algos, vals, color=[COLORS.get(a, "gray") for a in sorted_algos],
                edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Mean Return (across environments)")
        ax.set_title("Performance Ranking", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Training time ranking
        ax = axes[1]
        sorted_algos = sorted(algos, key=lambda a: algo_metrics[a]["total_time"])
        vals = [algo_metrics[a]["total_time"] for a in sorted_algos]
        ax.barh(sorted_algos, vals, color=[COLORS.get(a, "gray") for a in sorted_algos],
                edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Total Training Time (s)")
        ax.set_title("Training Time Ranking (lower is better)", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Composite: return per second
        ax = axes[2]
        rps = {}
        for algo in algos:
            t = algo_metrics[algo]["total_time"]
            r = algo_metrics[algo]["mean_return"]
            rps[algo] = r / (t + 1e-8)
        sorted_algos = sorted(algos, key=lambda a: rps[a], reverse=True)
        vals = [rps[a] for a in sorted_algos]
        ax.barh(sorted_algos, vals, color=[COLORS.get(a, "gray") for a in sorted_algos],
                edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Mean Return per Second")
        ax.set_title("Efficiency Ranking (return/time)", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        fig.suptitle("Algorithm Rankings", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir, "algorithm_ranking.png"),
                   dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(self.fig_dir, "algorithm_ranking.pdf"),
                   bbox_inches="tight")
        plt.close()

        print("    [OK] Algorithm rankings")

    def plot_training_stability(self):
        """V11: Training stability visualization."""
        completed = self._get_completed()
        envs = sorted(set(r["environment"] for r in completed))

        for env in envs:
            env_results = [r for r in completed if r["environment"] == env]
            if len(env_results) < 2:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Reward trajectory
            ax = axes[0]
            for result in env_results:
                history = result.get("history", {})
                rewards = [float(r) for r in history.get("episode_rewards", [])]
                if rewards:
                    rewards_arr = np.array(rewards, dtype=np.float64)
                    window = min(20, max(1, len(rewards_arr) // 5))
                    if window > 1 and len(rewards_arr) > window:
                        smoothed = self._smooth(rewards_arr, window)
                    else:
                        smoothed = rewards_arr
                    ax.plot(smoothed, label=result["algorithm"],
                           color=COLORS.get(result["algorithm"], "gray"),
                           linewidth=1.5, alpha=0.8)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Episode Return (smoothed)")
            ax.set_title("Reward Trajectory")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Loss trajectory
            ax = axes[1]
            for result in env_results:
                history = result.get("history", {})
                policy_loss = history.get("policy_loss", [])
                q_loss = history.get("q_loss", [])
                actor_loss = history.get("actor_loss", [])
                loss = [float(l) for l in (policy_loss or q_loss or actor_loss)]
                if loss:
                    loss_arr = np.array(loss, dtype=np.float64)
                    window = min(20, max(1, len(loss_arr) // 5))
                    if window > 1 and len(loss_arr) > window:
                        smoothed = self._smooth(loss_arr, window)
                    else:
                        smoothed = loss_arr
                    ax.plot(smoothed, label=result["algorithm"],
                           color=COLORS.get(result["algorithm"], "gray"),
                           linewidth=1.5, alpha=0.8)
            ax.set_xlabel("Update Step")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Trajectory")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Moving average reward with error band
            ax = axes[2]
            for result in env_results:
                history = result.get("history", {})
                rewards = [float(r) for r in history.get("episode_rewards", [])]
                if len(rewards) > 10:
                    rewards = np.array(rewards, dtype=np.float64)
                    window = min(20, len(rewards) // 3) or 5
                    ma = self._smooth(rewards, window)
                    std = np.array([np.std(rewards[max(0, i-window):i+window])
                                   for i in range(len(ma))])
                    steps = range(len(ma))
                    ax.plot(steps, ma, label=result["algorithm"],
                           color=COLORS.get(result["algorithm"], "gray"),
                           linewidth=1.5)
                    ax.fill_between(steps, ma - std, ma + std,
                                   color=COLORS.get(result["algorithm"], "gray"),
                                   alpha=0.1)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Episode Return")
            ax.set_title("Moving Average ± Std")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            fig.suptitle(f"Training Stability — {env}", fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dir, f"stability_{env}.png"),
                       dpi=200, bbox_inches="tight")
            plt.close()

        print("    [OK] Training stability plots")
