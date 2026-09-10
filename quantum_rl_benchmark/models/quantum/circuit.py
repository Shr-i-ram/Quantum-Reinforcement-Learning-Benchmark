"""
4-qubit baseline quantum circuit design.

This module defines the fixed quantum circuit architecture used across all
quantum RL models (Q-PPO, Q-DQN, Q-SAC). The circuit is a variational
quantum eigensolver-style ansatz with angle encoding and parameterized
rotations.

Architecture:
    - 4 qubits
    - Angle encoding (data re-uploading)
    - Parameterized RY + RZ rotations
    - Fixed entangling CNOT ladder
    - Circuit depth configurable (default: 2)
    - All measurements are expectation values of PauliZ

The circuit is differentiable through PennyLane's PyTorch integration.
"""

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple

import time

# Global device cache
_devices = {}
_circuit_eval_counter = {"count": 0, "time": 0.0}


def _get_device(n_qubits: int = 4, shots: Optional[int] = None):
    """Get or create a PennyLane device."""
    key = (n_qubits, shots)
    if key not in _devices:
        _devices[key] = qml.device(
            "lightning.qubit",
            wires=n_qubits,
            shots=shots,
        )
    return _devices[key]


def _build_circuit(inputs, params, n_qubits, depth):
    """
    Build the quantum circuit.

    Args:
        inputs: 4-dimensional feature vector for angle encoding
        params: variational parameters shape (depth, n_qubits, 2) for RY, RZ
        n_qubits: number of qubits (fixed at 4)
        depth: number of variational layers
    """
    # Angle encoding: encode each feature as RY rotation on corresponding qubit
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    # Variational layers
    for d in range(depth):
        # Parameterized rotations
        for i in range(n_qubits):
            qml.RY(params[d, i, 0], wires=i)
            qml.RZ(params[d, i, 1], wires=i)

        # Entangling CNOT ladder
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Reverse ladder for richer entanglement
        for i in range(n_qubits - 2, -1, -1):
            qml.CNOT(wires=[i + 1, i])


class QuantumCircuitLayer(nn.Module):
    """
    Differentiable quantum circuit layer for PyTorch.

    Wraps a PennyLane quantum circuit as a PyTorch nn.Module.
    Accepts 4-dimensional input features and produces n_qubits
    expectation values as output.

    Args:
        n_qubits: Number of qubits (must be 4)
        depth: Number of variational layers
        init_scale: Scale for parameter initialization
        shots: Number of shots (None for analytic)
        differentiation_method: 'backprop' or 'parameter-shift'
    """

    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 2,
        init_scale: float = 0.1,
        shots: Optional[int] = None,
        differentiation_method: str = "backprop",
    ):
        super().__init__()
        assert n_qubits == 4, f"Exactly 4 qubits required, got {n_qubits}"

        self.n_qubits = n_qubits
        self.depth = depth
        self.init_scale = init_scale
        self.shots = shots
        self.diff_method = differentiation_method

        # Variational parameters: (depth, n_qubits, 2) for RY + RZ
        self.params = nn.Parameter(
            torch.randn(depth, n_qubits, 2) * init_scale
        )

        # Build the quantum function
        dev = _get_device(n_qubits, shots)

        @qml.qnode(dev, interface="torch", diff_method=differentiation_method)
        def quantum_circuit(inputs, params):
            _build_circuit(inputs, params, n_qubits, depth)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_circuit = quantum_circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum circuit.

        Args:
            x: Input tensor of shape (batch_size, n_qubits=4)

        Returns:
            Tensor of shape (batch_size, n_qubits) with expectation values
        """
        batch_size = x.shape[0]
        outputs = []

        t0 = time.time()
        for i in range(batch_size):
            circuit_input = x[i]
            out = self.quantum_circuit(circuit_input, self.params)
            outputs.append(torch.stack(out))

        _circuit_eval_counter["count"] += batch_size
        _circuit_eval_counter["time"] += (time.time() - t0)

        return torch.stack(outputs)

    def get_circuit_info(self) -> Dict[str, Any]:
        """Get information about the quantum circuit."""
        return {
            "n_qubits": self.n_qubits,
            "circuit_depth": self.depth,
            "n_params": self.depth * self.n_qubits * 2,
            "n_entangling_gates": 2 * self.n_qubits - 2,  # per layer
            "total_entangling_gates": self.depth * (2 * self.n_qubits - 2),
            "n_gates_per_layer": self.n_qubits * 2 + 2 * self.n_qubits - 2,
            "total_gates": self.depth * (self.n_qubits * 2 + 2 * self.n_qubits - 2),
            "shots": self.shots,
            "differentiation_method": self.diff_method,
            "device": "lightning.qubit",
            "encoding": "angle",
            "total_circuit_evaluations": 0,  # tracked externally
        }


# Global counter for circuit evaluations
_circuit_eval_counter = {"count": 0, "time": 0.0}


def reset_circuit_eval_counter():
    """Reset the global circuit evaluation counter."""
    _circuit_eval_counter["count"] = 0
    _circuit_eval_counter["time"] = 0.0


def get_circuit_eval_count() -> int:
    """Get the current circuit evaluation count."""
    return _circuit_eval_counter["count"]


def get_circuit_eval_time() -> float:
    """Get the total quantum circuit evaluation time."""
    return _circuit_eval_counter["time"]


def get_circuit_info(circuit_layer: QuantumCircuitLayer) -> Dict[str, Any]:
    """Get comprehensive circuit information."""
    info = circuit_layer.get_circuit_info()
    info["circuit_evaluations"] = get_circuit_eval_count()
    info["quantum_time"] = get_circuit_eval_time()
    return info


class QuantumFeatureProjection(nn.Module):
    """
    Classical feature projection to 4-dimensional quantum features.

    Projects arbitrary-dimensional observations down to 4 features
    for input to the quantum circuit. Uses a small learned linear layer
    with optional nonlinearity.

    Args:
        input_dim: Dimensionality of environment observations
        output_dim: Output dimension (must be 4, matching qubit count)
        hidden_dim: Hidden layer dimension
    """

    def __init__(self, input_dim: int, output_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # bound to [-1, 1] for angle encoding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
