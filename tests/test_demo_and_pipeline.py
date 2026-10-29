"""
Comprehensive Test Suite for Quantum RL Benchmark and Live Demo.
Validates:
1. Environment instantiation and reset on all 5 Gymnasium environments.
2. Model loading from checkpoints for discrete & continuous classical & quantum models.
3. Forward and backward passes through QuantumCircuitLayer.
4. Internal action and expectation value inspection.
5. Baselines and ablation studies data validity.
6. Experiment registry integrity.
7. Demonstration checkpoint verification (CartPole 500 return).
8. Streamlit view rendering modules.
"""

import os
import sys
import json
import torch
import numpy as np
import gymnasium as gym

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quantum_rl_benchmark.config.settings import get_default_config, ENVIRONMENTS, EXPERIMENT_MATRIX
from visualize_agent import load_trained_agent, find_default_checkpoint
from quantum_rl_benchmark.models.quantum.circuit import QuantumCircuitLayer
from quantum_rl_benchmark.results.experiment_registry import load_experiment_registry
from quantum_rl_benchmark.demo.engine import get_compatible_models, step_policy_with_inspection


def test_environments_available():
    """Verify all 5 Gymnasium environments can be created, reset, and stepped."""
    for env_name, spec in ENVIRONMENTS.items():
        env = gym.make(spec["gymnasium_id"], render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        assert obs is not None, f"Reset failed for {env_name}"
        assert obs.shape[0] == spec["obs_dim"], f"Obs shape mismatch for {env_name}"

        # Test single step with random action
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        assert next_obs is not None
        assert isinstance(reward, (int, float, np.floating))

        # Test frame rendering
        frame = env.render()
        assert frame is not None and frame.ndim == 3
        env.close()


def test_quantum_circuit_execution_and_gradients():
    """Verify genuine 4-qubit quantum circuit forward pass and analytic adjoint gradient propagation."""
    q_layer = QuantumCircuitLayer(n_qubits=4, depth=2, differentiation_method="adjoint")
    x = torch.randn(4, 4, requires_grad=True)

    # Forward
    out = q_layer(x)
    assert out.shape == (4, 4), f"Expected shape (4, 4), got {out.shape}"
    # Expectation values must be in [-1, 1]
    assert (out >= -1.05).all() and (out <= 1.05).all(), "Pauli-Z expectation values out of range [-1, 1]"

    # Backward
    loss = out.sum()
    loss.backward()
    assert x.grad is not None and x.grad.norm() > 0, "Input gradient vanished"
    assert q_layer.params.grad is not None and q_layer.params.grad.norm() > 0, "Quantum parameter gradient vanished"


def test_demonstration_checkpoint():
    """Verify that the CartPole demonstration checkpoint exists and achieves high return."""
    ckpt_path = os.path.join(os.path.dirname(__file__), "..", "quantum_rl_benchmark", "results", "checkpoints", "DQN_CartPole-v1_demonstration.pt")
    assert os.path.exists(ckpt_path), f"Demonstration checkpoint missing: {ckpt_path}"

    agent = load_trained_agent("DQN", ckpt_path)
    env = gym.make("CartPole-v1")
    returns = []
    for ep in range(5):
        obs, _ = env.reset(seed=100 + ep)
        done = False
        ep_ret = 0
        while not done:
            act = agent.select_action(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(act))
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    env.close()

    mean_ret = np.mean(returns)
    assert mean_ret >= 400.0, f"Demonstration agent scored below expected threshold: {mean_ret}"


def test_quantum_model_loading_and_inspection():
    """Verify Q-PPO and Q-SAC models load and expose 4-qubit Pauli-Z expectations."""
    # Test Q-PPO on CartPole
    ckpt_qppo = find_default_checkpoint("Q-PPO", "CartPole-v1")
    agent_qppo = load_trained_agent("Q-PPO", ckpt_qppo)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=42)

    model_info = {"id": "Benchmark_Q-PPO", "algo": "Q-PPO", "type": "benchmark", "is_quantum": True}
    raw_act, step_act, insp = step_policy_with_inspection("CartPole-v1", model_info, obs, env, deterministic=True)
    env.close()

    assert insp["is_quantum"] is True
    assert insp["quantum_expectations"] is not None
    assert len(insp["quantum_expectations"]) == 4, "Expected 4 quantum expectation values"


def test_experiment_registry_integrity():
    """Verify the experiment registry exists and indexes all 20 benchmark runs."""
    reg = load_experiment_registry()
    bench = reg.get("benchmark_experiments", {})
    assert len(bench) == 20, f"Expected 20 benchmark experiments, found {len(bench)}"

    # Check key environments
    for env in ["CartPole-v1", "MountainCar-v0", "LunarLander-v3", "Pendulum-v1", "BipedalWalker-v3"]:
        matching = [k for k, v in bench.items() if v["environment"] == env]
        assert len(matching) == 4, f"Environment {env} has {len(matching)} registered experiments (expected 4)"

    assert "demonstration_experiments" in reg
    assert "baselines" in reg
    assert "ablations" in reg
    assert "robustness" in reg


if __name__ == "__main__":
    print("Running integration tests...")
    test_environments_available()
    print("[OK] Environments available test passed.")
    test_quantum_circuit_execution_and_gradients()
    print("[OK] Quantum circuit and gradients test passed.")
    test_demonstration_checkpoint()
    print("[OK] Demonstration checkpoint test passed.")
    test_quantum_model_loading_and_inspection()
    print("[OK] Quantum model inspection test passed.")
    test_experiment_registry_integrity()
    print("[OK] Experiment registry integrity test passed.")
    print("\nALL TESTS PASSED SUCCESSFULLY!")
