"""
Controlled Ablation Studies for Quantum RL Benchmark.
Evaluates:
1. Quantum Circuit Depth: Depth 1 vs Depth 2
2. Quantum Width: 2 Qubits vs 4 Qubits
3. Classical Control Architecture vs Hybrid Quantum VQC
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from quantum_rl_benchmark.models.quantum.circuit import QuantumCircuitLayer, QuantumFeatureProjection


class ClassicalControlLayer(nn.Module):
    """
    Classical Control Head: Matched architecture replacing the VQC with classical MLP.
    Explicitly labeled as a classical control architecture (NOT quantum).
    """
    def __init__(self, in_features: int = 4, out_features: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_features),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def run_circuit_depth_ablation(batch_size: int = 32, n_trials: int = 50) -> Dict[str, Any]:
    """Compare Depth 1 vs Depth 2 quantum circuit layers."""
    results = {}
    x = torch.randn(batch_size, 4, requires_grad=True)

    for depth in [1, 2]:
        layer = QuantumCircuitLayer(n_qubits=4, depth=depth, differentiation_method="adjoint")
        
        # Warmup
        _ = layer(x)

        # Benchmark forward time
        t0 = time.perf_counter()
        for _ in range(n_trials):
            out = layer(x)
        fwd_time = (time.perf_counter() - t0) / n_trials

        # Benchmark backward time
        t0 = time.perf_counter()
        for _ in range(n_trials):
            x_var = torch.randn(batch_size, 4, requires_grad=True)
            out = layer(x_var)
            loss = out.sum()
            loss.backward()
        bwd_time = (time.perf_counter() - t0) / n_trials

        # Gate count and parameter metrics
        n_params = depth * 4 * 2
        n_entangling = depth * (2 * 4 - 2)
        total_gates = depth * (4 * 2 + (2 * 4 - 2))

        results[f"depth_{depth}"] = {
            "depth": depth,
            "n_qubits": 4,
            "trainable_quantum_parameters": n_params,
            "entangling_cnot_gates": n_entangling,
            "total_quantum_gates": total_gates,
            "forward_latency_ms": fwd_time * 1000.0,
            "backward_latency_ms": bwd_time * 1000.0,
            "total_step_latency_ms": (fwd_time + bwd_time) * 1000.0,
            "sample_output_mean": float(out.detach().mean().item()),
            "sample_output_std": float(out.detach().std().item()),
        }

    return results


def run_qubit_width_ablation(batch_size: int = 32, n_trials: int = 50) -> Dict[str, Any]:
    """Compare 2-Qubit vs 4-Qubit quantum circuit architectures."""
    results = {}

    for n_qubits in [2, 4]:
        dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def vqc_circuit(inputs, params):
            for i in range(n_qubits):
                qml.RY(inputs[:, i], wires=i)
            # Depth 2
            for d in range(2):
                for i in range(n_qubits):
                    qml.RY(params[d, i, 0], wires=i)
                    qml.RZ(params[d, i, 1], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(n_qubits - 2, -1, -1):
                    qml.CNOT(wires=[i + 1, i])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        params = nn.Parameter(torch.randn(2, n_qubits, 2) * 0.1)
        x = torch.randn(batch_size, n_qubits, requires_grad=True)

        # Warmup
        _ = vqc_circuit(x, params)

        # Timing
        t0 = time.perf_counter()
        for _ in range(n_trials):
            out = vqc_circuit(x, params)
        fwd_time = (time.perf_counter() - t0) / n_trials

        n_params = 2 * n_qubits * 2
        n_cnot = 2 * (2 * n_qubits - 2) if n_qubits > 1 else 0
        state_dim = 2 ** n_qubits

        results[f"{n_qubits}_qubits"] = {
            "n_qubits": n_qubits,
            "hilbert_space_dim": state_dim,
            "circuit_depth": 2,
            "trainable_parameters": n_params,
            "entangling_cnot_gates": n_cnot,
            "forward_latency_ms": fwd_time * 1000.0,
        }

    return results


def run_classical_control_vs_quantum_ablation(batch_size: int = 32, n_trials: int = 50) -> Dict[str, Any]:
    """Compare 4-Qubit VQC against a Matched Classical Control Architecture."""
    x = torch.randn(batch_size, 4, requires_grad=True)

    # 1. 4-Qubit Quantum Circuit
    q_layer = QuantumCircuitLayer(n_qubits=4, depth=2, differentiation_method="adjoint")
    _ = q_layer(x)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        out_q = q_layer(x)
    q_fwd_time = (time.perf_counter() - t0) / n_trials

    t0 = time.perf_counter()
    for _ in range(n_trials):
        x_var = torch.randn(batch_size, 4, requires_grad=True)
        out = q_layer(x_var)
        loss = out.sum()
        loss.backward()
    q_bwd_time = (time.perf_counter() - t0) / n_trials

    # 2. Classical Control Architecture
    c_layer = ClassicalControlLayer(in_features=4, out_features=4, hidden_dim=16)
    _ = c_layer(x)

    t0 = time.perf_counter()
    for _ in range(n_trials):
        out_c = c_layer(x)
    c_fwd_time = (time.perf_counter() - t0) / n_trials

    t0 = time.perf_counter()
    for _ in range(n_trials):
        x_var = torch.randn(batch_size, 4, requires_grad=True)
        out = c_layer(x_var)
        loss = out.sum()
        loss.backward()
    c_bwd_time = (time.perf_counter() - t0) / n_trials

    return {
        "hybrid_quantum_4q": {
            "architecture_type": "hybrid_quantum_vqc",
            "is_quantum": True,
            "n_qubits": 4,
            "circuit_depth": 2,
            "trainable_parameters": 16,
            "forward_latency_ms": q_fwd_time * 1000.0,
            "backward_latency_ms": q_bwd_time * 1000.0,
            "total_step_latency_ms": (q_fwd_time + q_bwd_time) * 1000.0,
            "output_bounds": [-1.0, 1.0],
            "description": "4-qubit variational circuit with angle encoding, CNOT ladder, and analytic Pauli-Z measurement.",
        },
        "classical_control_control": {
            "architecture_type": "classical_mlp_control",
            "is_quantum": False,
            "trainable_parameters": c_layer.n_parameters,
            "forward_latency_ms": c_fwd_time * 1000.0,
            "backward_latency_ms": c_bwd_time * 1000.0,
            "total_step_latency_ms": (c_fwd_time + c_bwd_time) * 1000.0,
            "speedup_factor_vs_quantum": (q_fwd_time + q_bwd_time) / (c_fwd_time + c_bwd_time + 1e-9),
            "output_bounds": [-1.0, 1.0],
            "description": "Classical MLP control head (4 -> 16 -> 4 with Tanh) matched to the 4D feature representation.",
        }
    }


def run_all_ablations(save_path: str = None) -> Dict[str, Any]:
    """Execute all three controlled ablation studies."""
    print("Running Ablation 1: Quantum Circuit Depth (1 vs 2)...")
    depth_res = run_circuit_depth_ablation()

    print("Running Ablation 2: Quantum Width (2 vs 4 Qubits)...")
    width_res = run_qubit_width_ablation()

    print("Running Ablation 3: Classical Control Architecture vs Hybrid Quantum VQC...")
    control_res = run_classical_control_vs_quantum_ablation()

    results = {
        "depth_ablation": depth_res,
        "width_ablation": width_res,
        "classical_control_ablation": control_res,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Ablation results saved to: {save_path}")

    return results


if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "..", "..", "quantum_rl_benchmark", "results", "data", "ablation_results.json")
    res = run_all_ablations(out_file)
    print("Ablation study complete.")

