#!/usr/bin/env python3
"""
Quantum Reinforcement Learning Benchmark
========================================

"Comparison of Quantum Reinforcement Learning Algorithms
 in Increasingly Complex Gymnasium Environments"

Run the complete benchmark with a single command:

    python run_experiment.py

This will:
1. Validate the software environment
2. Validate all required Gymnasium environments
3. Train every algorithm on every applicable environment
4. Save checkpoints regularly
5. Resume interrupted experiments
6. Evaluate all completed models
7. Compute all metrics
8. Generate all visualizations
9. Produce summary tables and final results

Configuration can be modified in quantum_rl_benchmark/config/settings.py
or by passing a custom ExperimentConfig.
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Run the complete benchmark."""
    from quantum_rl_benchmark.orchestrator import Orchestrator
    from quantum_rl_benchmark.config.settings import get_default_config

    # Create default configuration
    config = get_default_config()

    # Print configuration summary
    print("=" * 70)
    print("  Configuration")
    print("=" * 70)
    print(f"  Seed: {config.seed}")
    print(f"  Quantum device: {config.quantum.device_name}")
    print(f"  Qubits: {config.quantum.n_qubits}")
    print(f"  Circuit depth: {config.quantum.circuit_depth}")
    print(f"  Shots: {config.quantum.shots} (analytic)")
    print(f"  Differentiation: {config.quantum.differentiation_method}")
    print(f"  Results dir: {config.base_dir}")
    print()

    # Run orchestrator
    orchestrator = Orchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
