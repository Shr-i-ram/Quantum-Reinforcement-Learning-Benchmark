"""
Experiment orchestrator.

Manages the entire benchmark workflow from validation through
training, evaluation, metrics, and visualization.
"""

import os
import sys
import json
import time
import platform
import numpy as np
import torch
from typing import Dict, Any, List

from quantum_rl_benchmark.config.settings import (
    ExperimentConfig, ENVIRONMENTS, EXPERIMENT_MATRIX,
    get_experiment_key, get_default_config,
)
from quantum_rl_benchmark.environments.manager import EnvironmentManager
from quantum_rl_benchmark.environments.complexity import EnvironmentComplexity
from quantum_rl_benchmark.training.trainer import Trainer
from quantum_rl_benchmark.evaluation.evaluator import Evaluator
from quantum_rl_benchmark.metrics.metrics import MetricsComputer
from quantum_rl_benchmark.visualization.plots import VisualizationSuite


class Orchestrator:
    """
    Orchestrates the complete benchmark workflow.

    Workflow:
    1. Validate software environment
    2. Validate Gymnasium environments
    3. Train all required models
    4. Evaluate completed models
    5. Compute metrics
    6. Generate visualizations
    7. Produce final summary
    """

    def __init__(self, config: ExperimentConfig = None):
        if config is None:
            config = get_default_config()
        self.config = config
        self.start_time = None
        self.results_summary = {}
        self.available_envs = {}  # set after validation

    def run(self):
        """Execute the complete benchmark workflow."""
        self.start_time = time.time()

        print("=" * 70)
        print("  QUANTUM REINFORCEMENT LEARNING BENCHMARK")
        print("  Comparison of QRL Algorithms in Increasing Complex Environments")
        print("=" * 70)
        print()

        # Step 1: Validate software environment
        self._print_section("STEP 1: Software Environment Validation")
        self._validate_software()

        # Step 2: Validate environments
        self._print_section("STEP 2: Environment Validation")
        self.available_envs = self._validate_environments()
        available_env_names = [k for k, v in self.available_envs.items() if v]
        if len(available_env_names) < len(ENVIRONMENTS):
            print(f"\n  Running benchmark with {len(available_env_names)}/{len(ENVIRONMENTS)} environments.")

        # Step 3: Compute complexity metrics
        self._print_section("STEP 3: Environment Complexity Analysis")
        self._analyze_complexity()

        # Step 4: Training
        self._print_section("STEP 4: Training")
        training_results = self._run_all_experiments()

        # Step 5: Evaluation (included in training, but we re-load)
        self._print_section("STEP 5: Aggregating Results")
        evaluator = Evaluator(self.config.data_dir, self.config.figure_dir)
        evaluation = evaluator.evaluate_all(ENVIRONMENTS, self.config.performance_thresholds)

        # Step 6: Metrics
        self._print_section("STEP 6: Computing Metrics")
        metrics_computer = MetricsComputer(self.config.data_dir)
        all_metrics = metrics_computer.compute_all_metrics()

        # Step 7: Visualizations
        self._print_section("STEP 7: Generating Visualizations")
        viz = VisualizationSuite(self.config.figure_dir, self.config.data_dir)
        viz.generate_all()

        # Step 8: Final Summary
        self._print_section("STEP 8: Final Summary")
        self._print_final_summary(all_metrics)

        # Save config
        self.config.save(os.path.join(self.config.data_dir, "experiment_config.json"))

        total_time = time.time() - self.start_time
        print(f"\n{'=' * 70}")
        print(f"  BENCHMARK COMPLETE")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Results: {self.config.data_dir}")
        print(f"  Figures: {self.config.figure_dir}")
        print(f"  Config:  {self.config.data_dir}/experiment_config.json")
        print(f"  Metrics: {self.config.data_dir}/all_metrics.json")
        print(f"{'=' * 70}")

    def _print_section(self, title: str):
        print(f"\n{'-' * 70}")
        print(f"  {title}")
        print(f"{'-' * 70}")

    def _validate_software(self):
        """Validate all required software packages."""
        print("  Checking software dependencies...")

        required = {
            "gymnasium": "gymnasium",
            "torch": "torch",
            "pennylane": "pennylane",
            "numpy": "numpy",
            "pandas": "pandas",
            "matplotlib": "matplotlib",
            "scipy": "scipy",
            "sklearn": "sklearn",
        }

        versions = {}
        all_ok = True
        for name, module_name in required.items():
            try:
                mod = __import__(module_name)
                ver = getattr(mod, "__version__", "unknown")
                versions[name] = ver
                print(f"    [OK] {name} {ver}")
            except ImportError:
                print(f"    [FAIL] {name} NOT FOUND")
                all_ok = False

        # Check PennyLane lightning
        try:
            import pennylane as qml
            qml.device("lightning.qubit", wires=2)
            versions["pennylane-lightning"] = "available"
            print(f"    [OK] pennylane-lightning.qubit available")
        except Exception as e:
            print(f"    [FAIL] pennylane-lightning.qubit: {e}")
            all_ok = False

        # Check PyTorch CUDA
        cuda_available = torch.cuda.is_available()
        print(f"    {'[OK]' if cuda_available else '[--]'} CUDA available: {cuda_available}")

        # Platform info
        print(f"    [--] Platform: {platform.platform()}")
        print(f"    [--] Python: {sys.version}")
        print(f"    [--] PyTorch: {torch.__version__}")
        if cuda_available:
            print(f"    [--] GPU: {torch.cuda.get_device_name(0)}")

        # Save system info
        sys_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "versions": versions,
        }
        info_path = os.path.join(self.config.data_dir, "system_info.json")
        with open(info_path, "w") as f:
            json.dump(sys_info, f, indent=2)

        if not all_ok:
            print("\n  WARNING: Some dependencies are missing!")
            print("  Install missing packages before running the benchmark.")
            return False

        print("  All dependencies validated.")
        return True

    def _validate_environments(self) -> Dict[str, bool]:
        """Validate all required Gymnasium environments."""
        print("  Checking Gymnasium environments...")

        env_manager = EnvironmentManager(seed=self.config.seed)
        results = env_manager.validate_all(ENVIRONMENTS)

        all_ok = True
        for env_name, available in results.items():
            status = "[OK]" if available else "[FAIL]"
            print(f"    {status} {env_name}")
            if not available:
                all_ok = False

        if not all_ok:
            print("\n  WARNING: Some environments are not available!")
            print("  Install gymnasium[box2d] for LunarLander and BipedalWalker.")

        return results

    def _analyze_complexity(self):
        """Analyze and display environment complexity."""
        complexity = EnvironmentComplexity(ENVIRONMENTS)

        ranking = complexity.get_complexity_ranking()
        print("  Environment complexity ranking:")
        for i, (env, score) in enumerate(ranking, 1):
            desc = complexity.complexity_description(env)
            print(f"    {i}. {env} (score: {score:.3f})")
            print(f"       {desc}")

        boundary = complexity.get_discrete_continuous_boundary()
        print(f"\n  Discrete -> Continuous transition at complexity level {boundary}")

    def _run_all_experiments(self) -> Dict[str, Any]:
        """Run all algorithm-environment combinations."""
        trainer = Trainer(self.config)
        all_results = {}

        # Build experiment queue, skipping unavailable environments
        experiments = []
        for env_name, algos in EXPERIMENT_MATRIX.items():
            if self.available_envs.get(env_name, False):
                for algo in algos:
                    experiments.append((algo, env_name))
            else:
                print(f"  [SKIP] {env_name}: environment not available")

        total = len(experiments)
        print(f"  Total experiments: {total}")
        print()

        for i, (algo, env_name) in enumerate(experiments, 1):
            key = get_experiment_key(algo, env_name)
            print(f"\n  [{i}/{total}] {algo} on {env_name}")

            # Check if already completed
            results_path = os.path.join(self.config.data_dir, f"{key}_results.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    existing = json.load(f)
                if existing.get("status") == "completed":
                    print(f"    [SKIP] Already completed. Loading existing results.")
                    all_results[key] = existing
                    continue

            try:
                result = trainer.train(algo, env_name, ENVIRONMENTS)
                all_results[key] = result
            except Exception as e:
                print(f"    [ERROR] {algo} on {env_name} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results[key] = {
                    "algorithm": algo,
                    "environment": env_name,
                    "status": "failed",
                    "error": str(e),
                }

        return all_results

    def _print_final_summary(self, metrics: Dict):
        """Print the comprehensive final summary."""
        summary = metrics.get("final_summary", {})

        print("\n" + "=" * 70)
        print("  FINAL RESEARCH SUMMARY")
        print("=" * 70)

        # Best per environment
        print("\n  BEST ALGORITHM PER ENVIRONMENT:")
        print("  " + "-" * 40)
        for env, info in summary.get("best_per_environment", {}).items():
            print(f"    {env}: {info['algorithm']} (return: {info['mean_return']:.2f})")

        # Classical vs quantum
        cvq = summary.get("classical_vs_quantum", {})
        if cvq:
            print(f"\n  CLASSICAL vs QUANTUM OVERALL:")
            print(f"    Classical mean return: {cvq.get('classical_mean', 'N/A')}")
            print(f"    Quantum mean return:   {cvq.get('quantum_mean', 'N/A')}")

        # PPO vs Q-PPO
        ppo_comp = metrics.get("ppo_vs_qppo", {})
        if ppo_comp:
            print(f"\n  PPO vs Q-PPO:")
            for env, comp in ppo_comp.items():
                winner = "Q-PPO" if comp["qppo_wins"] else "PPO"
                diff = comp["difference"]
                print(f"    {env}: {winner} wins (diff: {diff:+.2f})")

        # DQN vs Q-DQN
        dqn_comp = metrics.get("dqn_vs_qdqn", {})
        if dqn_comp:
            print(f"\n  DQN vs Q-DQN:")
            for env, comp in dqn_comp.items():
                winner = "Q-DQN" if comp["qdqn_wins"] else "DQN"
                diff = comp["difference"]
                print(f"    {env}: {winner} wins (diff: {diff:+.2f})")

        # SAC vs Q-SAC
        sac_comp = metrics.get("sac_vs_qsac", {})
        if sac_comp:
            print(f"\n  SAC vs Q-SAC:")
            for env, comp in sac_comp.items():
                winner = "Q-SAC" if comp["qsac_wins"] else "SAC"
                diff = comp["difference"]
                print(f"    {env}: {winner} wins (diff: {diff:+.2f})")

        # Limitations
        print(f"\n  LIMITATIONS:")
        for lim in summary.get("limitations", []):
            print(f"    - {lim}")

        # Research questions
        print(f"\n  RESEARCH QUESTIONS ADDRESSED:")
        print(f"    1. Which classical model performs best per environment?")
        print(f"    2. Which quantum model performs best per environment?")
        print(f"    3. Does Q-PPO outperform PPO? See PPO vs Q-PPO above.")
        print(f"    4. Does Q-DQN outperform DQN? See DQN vs Q-DQN above.")
        print(f"    5. Does Q-SAC outperform SAC? See SAC vs Q-SAC above.")
        print(f"    6. How does relative performance change with difficulty? See complexity_scaling.png")
        print(f"    7. Are quantum models more sample-efficient? See AULC comparison.")
        print(f"    8. Are quantum models more stable? See stability plots.")
        print(f"    9. What computational overhead? See runtime_vs_performance.png")
        print(f"    10. Is overhead justified? See algorithm_ranking.png")
