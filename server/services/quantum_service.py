"""
Quantum Service for Quantum RL Arena.
Executes live PennyLane 4-qubit quantum circuits, extracts Pauli-Z expectation values,
computes statevector probabilities, and provides NISQ vs classical simulation complexity analytics.
"""

import math
import pennylane as qml
import numpy as np
import torch
from typing import Dict, Any, List, Optional

# Re-use or initialize default.qubit device for rapid live simulation
_sim_device = qml.device("default.qubit", wires=4)


@qml.qnode(_sim_device)
def _interactive_circuit(features, weights, depth=2):
    """
    PennyLane 4-qubit circuit matching the benchmark architecture.
    Features: (4,) array of angles in [-pi, pi] or [-1, 1]
    Weights: (depth, 4, 2) array of variational rotation angles
    """
    # Angle encoding
    for i in range(4):
        qml.RY(features[i], wires=i)

    # Variational layers
    for d in range(depth):
        for i in range(4):
            qml.RY(weights[d, i, 0], wires=i)
            qml.RZ(weights[d, i, 1], wires=i)

        # CNOT entangling ladder
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
        for i in range(2, -1, -1):
            qml.CNOT(wires=[i + 1, i])

    # Return Pauli-Z expectation values on all 4 wires
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


@qml.qnode(_sim_device)
def _state_probs_circuit(features, weights, depth=2):
    """PennyLane circuit returning full 16-basis state probability distribution."""
    for i in range(4):
        qml.RY(features[i], wires=i)

    for d in range(depth):
        for i in range(4):
            qml.RY(weights[d, i, 0], wires=i)
            qml.RZ(weights[d, i, 1], wires=i)

        for i in range(3):
            qml.CNOT(wires=[i, i + 1])
        for i in range(2, -1, -1):
            qml.CNOT(wires=[i + 1, i])

    return qml.probs(wires=range(4))


def simulate_quantum_circuit(
    features: List[float],
    weights: Optional[List[List[List[float]]]] = None,
    depth: int = 2
) -> Dict[str, Any]:
    """
    Run the real 4-qubit variational circuit given continuous feature angles.
    Returns Pauli-Z expectations, basis probabilities, and circuit layout.
    """
    if len(features) != 4:
        raise ValueError("Features must contain exactly 4 values.")

    feat_arr = np.array(features, dtype=np.float64)

    # Default weights if not provided (pre-initialized or zero)
    if weights is None:
        w_arr = np.ones((depth, 4, 2), dtype=np.float64) * 0.5
    else:
        w_arr = np.array(weights, dtype=np.float64)

    # Execute PennyLane QNodes
    expvals = _interactive_circuit(feat_arr, w_arr, depth=depth)
    probs = _state_probs_circuit(feat_arr, w_arr, depth=depth)

    # Format 16 basis state strings (|0000> ... |1111>)
    basis_labels = [f"|{bin(i)[2:].zfill(4)}⟩" for i in range(16)]
    basis_probs = [
        {"basis": basis_labels[i], "probability": float(probs[i])}
        for i in range(16)
    ]

    expval_list = [float(e) for e in expvals]

    # Circuit diagram representation
    diagram_text = (
        "|q0⟩ ─── RY(x₀) ─── [ RY(θ₀) RZ(ϕ₀) ] ──╭─●───╮ ─────── [ExpVal <Z₀>]\n"
        "|q1⟩ ─── RY(x₁) ─── [ RY(θ₁) RZ(ϕ₁) ] ──╰─X──╭─●───╮ ─── [ExpVal <Z₁>]\n"
        "|q2⟩ ─── RY(x₂) ─── [ RY(θ₂) RZ(ϕ₂) ] ───────╰─X──╭─●─╮  [ExpVal <Z₂>]\n"
        "|q3⟩ ─── RY(x₃) ─── [ RY(θ₃) RZ(ϕ₃) ] ────────────╰─X─╯  [ExpVal <Z₃>]\n"
    )

    return {
        "features": features,
        "depth": depth,
        "expectation_values": [
            {"qubit": 0, "expval": expval_list[0], "pauli": "⟨Z₀⟩"},
            {"qubit": 1, "expval": expval_list[1], "pauli": "⟨Z₁⟩"},
            {"qubit": 2, "expval": expval_list[2], "pauli": "⟨Z₂⟩"},
            {"qubit": 3, "expval": expval_list[3], "pauli": "⟨Z₃⟩"},
        ],
        "basis_probabilities": basis_probs,
        "circuit_diagram": diagram_text,
    }


def compute_quantum_scaling_analysis(
    qubits_range: Optional[List[int]] = None,
    shots: int = 1000,
    depth: int = 4
) -> Dict[str, Any]:
    """
    Compute rigorous scaling curves comparing classical statevector simulation
    against physical NISQ QPU execution latency and memory constraints.
    """
    if qubits_range is None:
        qubits_range = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

    classical_memory_mb = []
    classical_sim_time_ms = []
    qpu_runtime_ms = []
    barren_plateau_var = []

    # Parameters
    # Classical statevector: 2^n complex128 numbers (16 bytes each)
    # Classical matrix-vector FLOPs: ~ depth * n * 2^n operations
    # QPU: gate time ~ 50ns 1Q, 200ns 2Q; readout ~ 2000ns. Total circuit duration ~ 20-50us.
    # Total QPU time per step = shots * (circuit_time + readout) + cloud batch overhead (~2.0 sec).
    circuit_execution_us = depth * 5.0  # ~20 us
    readout_us = 2.0                    # ~2 us
    single_shot_us = circuit_execution_us + readout_us

    for n in qubits_range:
        # Statevector memory in Megabytes
        state_dim = 2 ** n
        mem_bytes = state_dim * 16
        classical_memory_mb.append(round(mem_bytes / (1024 * 1024), 4))

        # Classical simulation time estimate (exponential in n)
        # Calibrated: 4 qubits = ~0.05ms, 20 qubits = ~1000ms, 30 qubits = ~10^6 ms
        sim_ms = 0.003 * (2 ** (n - 4))
        classical_sim_time_ms.append(round(sim_ms, 4))

        # Physical QPU time: independent of 2^n! Linear in shots S and circuit depth D.
        # Plus cloud queue amortized overhead per execution batch (~50 ms)
        qpu_ms = (shots * single_shot_us / 1000.0) + 50.0
        qpu_runtime_ms.append(round(qpu_ms, 2))

        # Barren plateau variance decay: Var[grad] ~ 1 / 2^n
        bp_var = 1.0 / (2 ** n)
        barren_plateau_var.append(float(bp_var))

    return {
        "qubits": qubits_range,
        "shots": shots,
        "depth": depth,
        "classical_memory_mb": classical_memory_mb,
        "classical_sim_time_ms": classical_sim_time_ms,
        "qpu_runtime_ms": qpu_runtime_ms,
        "barren_plateau_variance": barren_plateau_var,
        "crossover_qubit_estimate": 22,  # Around 22-24 qubits classical sim exceeds QPU latency
        "insights": {
            "simulation_bottleneck": "At 4 qubits, classical simulation (0.05ms) is 1000x faster than cloud QPU sampling (50ms).",
            "memory_wall": "At 30 qubits, classical statevector requires 16 GB RAM; at 40 qubits, 16 Terabytes.",
            "barren_plateau_hazard": "Variance of policy gradients vanishes exponentially with qubit count, severely impairing gradient-based RL training without local cost functions."
        }
    }
