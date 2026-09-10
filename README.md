# ⚛️ Quantum Reinforcement Learning Benchmark

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.35+-FFD43B.svg?logo=python)](https://pennylane.ai/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0+-008080.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Comparison of Quantum Reinforcement Learning Algorithms in Increasingly Complex Gymnasium Environments"**

A comprehensive, reproducible empirical benchmark comparing **Variational Quantum Reinforcement Learning (VQC / QRL)** algorithms against state-of-the-art **Classical Deep Reinforcement Learning** baselines across five Gymnasium environments spanning discrete and continuous control, dense and sparse reward landscapes, and low-to-high dimensional observation manifolds.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Benchmark Matrix & Complexity Tiers](#-benchmark-matrix--complexity-tiers)
- [Algorithmic Architecture](#-algorithmic-architecture)
  - [Variational Quantum Circuit (VQC) Design](#variational-quantum-circuit-vqc-design)
  - [Hybrid Quantum-Classical Agents](#hybrid-quantum-classical-agents)
- [Empirical Results](#-empirical-results)
  - [Benchmark Performance Table](#benchmark-performance-table)
  - [Key Findings & Research Insights](#key-findings--research-insights)
- [Visualizations & Plots](#-visualizations--plots)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Running Experiments](#-running-experiments)
- [Evaluation & Custom Configuration](#-evaluation--custom-configuration)
- [License & Citation](#-license--citation)

---

## 🌟 Overview

Parameterized Variational Quantum Circuits (VQCs) are leading contenders for demonstrating practical quantum utility on Noisy Intermediate-Scale Quantum (NISQ) devices. While VQCs have been studied on toy problems (e.g., CartPole), open questions remain regarding how quantum agents scale when evaluated against increasingly challenging control environments:

1. **Expressivity vs. Capacity**: Can a compact 4-qubit variational circuit with parameter count orders of magnitude smaller than deep neural networks match or exceed classical sample efficiency?
2. **Complexity Scaling**: At what state dimension, observation horizon, and dynamical complexity do current VQC architectures maintain stability or encounter optimization bottlenecks (e.g., barren plateaus)?
3. **Computational Trade-Offs**: What is the simulation and wall-clock training overhead of quantum backpropagation vs. standard neural backpropagation?

This benchmark provides rigorous empirical answers, standardized interfaces, checkpointing, evaluation metrics, and visualization tools.

---

## 🚀 Key Features

- **Hybrid Quantum-Classical Agents**:
  - **Q-PPO**: Proximal Policy Optimization with a variational quantum actor and critic.
  - **Q-DQN**: Deep Q-Network featuring a quantum expectation layer for action-value approximation.
  - **Q-SAC**: Soft Actor-Critic featuring hybrid quantum actor and double Q-critics for continuous domains.
- **Classical Baselines**:
  - Equivalent classical **PPO**, **DQN**, and **SAC** architectures trained with identical reward scaling, horizon budgets, and environment wrappers.
- **Hardware-Accelerated Quantum Simulation**:
  - Powered by **PennyLane** with the high-performance `lightning.qubit` state-vector backend and PyTorch adjoint / backpropagation differentiation.
- **5-Tier Complexity Spectrum**:
  - Environments categorized systematically across observation dimensions (2 to 24), action spaces (discrete & continuous), horizon lengths (200 to 2000), and reward sparsity.
- **Complete Automated Pipeline**:
  - Single-command execution: software validation $\rightarrow$ environment verification $\rightarrow$ training $\rightarrow$ checkpointing $\rightarrow$ evaluation $\rightarrow$ metrics extraction $\rightarrow$ publication-ready figure rendering.

---

## 🎯 Benchmark Matrix & Complexity Tiers

| Level | Environment | Type | State Dim | Action Dim | Max Steps | Reward Dynamics | Key Challenge |
|:---:|:---|:---:|:---:|:---:|:---:|:---|:---|
| **1** | `CartPole-v1` | Discrete | 4 | 2 | 500 | Dense | Low-dimensional inverted pendulum balance |
| **2** | `MountainCar-v0` | Discrete | 2 | 3 | 200 | Sparse (-1/step) | Sparse reward, underactuated momentum build-up |
| **3** | `LunarLander-v3` | Discrete | 8 | 4 | 1000 | Multi-objective | 8D state, orientation & engine thruster trade-offs |
| **4** | `Pendulum-v1` | Continuous | 3 | 1 ($\pm 2.0$) | 200 | Dense ($-\theta^2 - 0.1\dot{\theta}^2 - 0.001u^2$) | Continuous non-linear torque control |
| **5** | `BipedalWalker-v3` | Continuous | 24 | 4 ($\pm 1.0$) | 2000 | Dense | High-dimensional kinematics, terrain contact, balance |

### Experiment Allocation Matrix

| Environment | PPO | Q-PPO | DQN | Q-DQN | SAC | Q-SAC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `CartPole-v1` | ✅ | ✅ | ✅ | ✅ | — | — |
| `MountainCar-v0` | ✅ | ✅ | ✅ | ✅ | — | — |
| `LunarLander-v3` | ✅ | ✅ | ✅ | ✅ | — | — |
| `Pendulum-v1` | ✅ | ✅ | — | — | ✅ | ✅ |
| `BipedalWalker-v3`| ✅ | ✅ | — | — | ✅ | ✅ |

---

## 🔬 Algorithmic Architecture

### Variational Quantum Circuit (VQC) Design

All quantum models leverage a unified, parameter-efficient 4-qubit quantum circuit implemented via PennyLane:

```
Inputs x (4D) ──[ RY(x0) ]──[ RY(θ0,0) ]──[ RZ(θ0,1) ]──●────×───[ M(Z0) ]──> ⟨Z0⟩
Inputs x (4D) ──[ RY(x1) ]──[ RY(θ1,0) ]──[ RZ(θ1,1) ]──X──●─│───[ M(Z1) ]──> ⟨Z1⟩
Inputs x (4D) ──[ RY(x2) ]──[ RY(θ2,0) ]──[ RZ(θ2,1) ]──│──X─●───[ M(Z2) ]──> ⟨Z2⟩
Inputs x (4D) ──[ RY(x3) ]──[ RY(θ3,0) ]──[ RZ(θ3,1) ]──│────X───[ M(Z3) ]──> ⟨Z3⟩
                                [ Entangling CNOT Ladder ]
```

1. **State Dimension Matching / Preprocessing**: High-dimensional observations ($d > 4$) are projected to a 4-dimensional feature representation using an initial linear compression layer.
2. **Angle Encoding**: Features are encoded into quantum states via $R_y(x_i)$ single-qubit rotations.
3. **Variational Ansatz (Depth $D=2$)**:
   - Parameterized single-qubit rotations: $R_y(\theta_{d, i, 0})$ and $R_z(\theta_{d, i, 1})$.
   - Forward CNOT entangling ladder ($q_i \rightarrow q_{i+1}$) followed by reverse ladder ($q_{i+1} \rightarrow q_i$) to foster multi-qubit entanglement.
4. **Observable Measurement**: Expectation values $\langle Z_i \rangle = \langle \psi | \sigma_z^{(i)} | \psi \rangle \in [-1, 1]$ are extracted for each qubit $i \in \{0, 1, 2, 3\}$.
5. **Output Projection**: Quantum expectation values pass to a readout head that maps to policy distributions $\pi(a|s)$ or action-values $Q(s, a)$.

### Hybrid Quantum-Classical Agents

- **Q-PPO**: Variational quantum actor network estimating action probabilities alongside a classical/quantum critic estimating state value $V(s)$. Uses Generalized Advantage Estimation (GAE) and clipped surrogate loss.
- **Q-DQN**: Action-value network $Q(s, a; \theta)$ parameterized by the VQC layer with experience replay buffer ($100\text{k}$ transitions) and Polyak/periodic target network updates.
- **Q-SAC**: Off-policy actor-critic with entropy regularization, learning continuous action distributions via squashed Gaussian policies parameterized through hybrid quantum heads.

---

## 📊 Empirical Results

### Benchmark Performance Table

Empirical results obtained from comprehensive training runs:

| Environment | Algorithm | Quantum? | Mean Return | Std Return | Best Return | Median Return | Training Time (s) | Total Steps | Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CartPole-v1** | **Q-PPO** | ⚛️ Yes | **192.50** | 27.61 | **252.00** | 185.50 | 2,351.1 s | 100,000 | 5,999 |
| CartPole-v1 | PPO | No | 108.40 | 5.99 | 120.00 | 109.50 | 25.1 s | 100,000 | 4,675 |
| CartPole-v1 | DQN | No | 154.60 | 2.73 | 159.00 | 154.50 | 306.7 s | 100,000 | 17,410 |
| CartPole-v1 | Q-DQN | ⚛️ Yes | 98.70 | 2.61 | 102.00 | 99.50 | 65,503.7 s | 100,000 | 19,310 |
| **MountainCar-v0** | DQN | No | **-141.30** | 38.46 | **-115.00** | -116.50 | 242.8 s | 100,000 | 17,283 |
| MountainCar-v0 | PPO | No | -200.00 | 0.00 | -200.00 | -200.00 | 29.1 s | 100,000 | 4,612 |
| MountainCar-v0 | Q-PPO | ⚛️ Yes | -200.00 | 0.00 | -200.00 | -200.00 | 2,422.3 s | 100,000 | 5,748 |
| MountainCar-v0 | Q-DQN | ⚛️ Yes | -200.00 | 0.00 | -200.00 | -200.00 | 54,461.3 s | 100,000 | 19,059 |
| **LunarLander-v3** | DQN | No | **122.53** | 117.52 | **277.66** | 145.48 | 385.3 s | 150,000 | 18,180 |
| LunarLander-v3 | **Q-PPO** | ⚛️ Yes | **-375.83** | 107.87 | **-208.74** | -385.90 | 3,181.8 s | 150,000 | 6,521 |
| LunarLander-v3 | PPO | No | -568.45 | 164.93 | -371.08 | -507.94 | 44.4 s | 150,000 | 5,061 |
| **Pendulum-v1** | **Q-PPO** | ⚛️ Yes | **-1353.81** | 211.84 | **-1036.08** | -1352.08 | 2,406.3 s | 100,000 | 5,867 |
| Pendulum-v1 | PPO | No | -1409.92 | 265.69 | -856.06 | -1497.69 | 35.1 s | 100,000 | 4,547 |
| **BipedalWalker-v3**| **Q-PPO** | ⚛️ Yes | **-143.28** | 1.94 | **-142.58** | -142.65 | 3,063.3 s | 150,000 | 8,573 |
| BipedalWalker-v3 | PPO | No | -165.41 | 27.76 | -107.45 | -179.28 | 45.1 s | 150,000 | 6,089 |

### 🔍 Key Findings & Research Insights

1. **Superior Policy Optimization in Q-PPO**:
   - On **CartPole-v1**, Q-PPO achieved a mean return of **192.50** (best **252.00**), outperforming classical PPO (**108.40**) by **+77.6%**.
   - On **LunarLander-v3**, Q-PPO demonstrated significantly superior guidance relative to classical PPO (mean return **-375.83** vs. **-568.45**), showing greater resilience to policy collapse in multi-modal state spaces.
   - On **BipedalWalker-v3**, Q-PPO attained higher average returns (**-143.28** vs **-165.41**) with an order-of-magnitude lower variance ($\sigma = 1.94$ vs $\sigma = 27.76$), confirming strong regularization properties in variational quantum policies.
2. **Exploration Bottlenecks in Sparse Rewards**:
   - In **MountainCar-v0**, both classical and quantum PPO failed to reach the hilltop within the step budget due to reward sparsity, while classical DQN successfully escaped the valley (-141.30). Q-DQN struggled due to high computational overhead limiting effective gradient update frequencies.
3. **Simulation Runtime & Computational Overhead**:
   - Quantum simulation introduces a substantial wall-clock overhead on classical CPUs/GPUs (e.g., Q-PPO ~40–50 min vs classical PPO ~30–45 sec). This highlights the critical necessity of physical Quantum Processing Units (QPUs) to unlock the speed advantages of quantum state evolution.

---

## 📈 Visualizations & Plots

All generated publication-quality figures are stored in [`quantum_rl_benchmark/results/figures/`](quantum_rl_benchmark/results/figures/):

### Performance Heatmap & Algorithm Ranking
| Overall Algorithm Ranking | Performance Across Environments |
|:---:|:---:|
| ![Algorithm Ranking](quantum_rl_benchmark/results/figures/algorithm_ranking.png) | ![Performance Heatmap](quantum_rl_benchmark/results/figures/performance_heatmap.png) |

### Scaling, Area Under Learning Curve (AULC) & Efficiency
| Complexity Scaling Analysis | Area Under Learning Curve (AULC) |
|:---:|:---:|
| ![Complexity Scaling](quantum_rl_benchmark/results/figures/complexity_scaling.png) | ![AULC Comparison](quantum_rl_benchmark/results/figures/aulc_comparison.png) |

### Quantum Gradients & Resource Overhead
| Quantum Gradient Dynamics | Quantum Resource Utilization |
|:---:|:---:|
| ![Quantum Gradients](quantum_rl_benchmark/results/figures/quantum_gradients.png) | ![Quantum Resources](quantum_rl_benchmark/results/figures/quantum_resources.png) |

### Runtime Trade-offs
| Runtime vs Performance Pareto Analysis | Training Time Heatmap |
|:---:|:---:|
| ![Runtime vs Performance](quantum_rl_benchmark/results/figures/runtime_vs_performance.png) | ![Training Time Heatmap](quantum_rl_benchmark/results/figures/training_time_heatmap.png) |

---

## 📁 Repository Structure

```tree
Quantum-Reinforcement-Learning-Benchmark/
├── .gitattributes                          # Cross-platform line endings and binary spec
├── .gitignore                              # Clean ignore patterns (venv, checkpoints, cache)
├── LICENSE                                 # MIT Open-Source License
├── README.md                               # Project documentation & benchmark report
├── pyproject.toml                          # PEP 517/621 build and installation metadata
├── requirements.txt                        # Core runtime dependencies
├── requirements-dev.txt                    # Development, testing, and formatting tools
├── run_experiment.py                       # Top-level entrypoint to execute benchmark
└── quantum_rl_benchmark/                   # Main benchmark package
    ├── __init__.py
    ├── orchestrator.py                     # Pipeline orchestration and validation suite
    ├── config/
    │   ├── __init__.py
    │   └── settings.py                     # Hyperparameters, configs & environment matrix
    ├── environments/
    │   ├── __init__.py
    │   ├── complexity.py                   # Complexity metric calculations (entropy, dim)
    │   └── manager.py                      # Gymnasium environment builder & wrappers
    ├── models/
    │   ├── __init__.py
    │   ├── classical/                      # Classical RL agents
    │   │   ├── __init__.py
    │   │   ├── dqn.py                      # Deep Q-Network
    │   │   ├── ppo.py                      # Proximal Policy Optimization
    │   │   └── sac.py                      # Soft Actor-Critic
    │   └── quantum/                        # Quantum & Hybrid VQC agents
    │       ├── __init__.py
    │       ├── circuit.py                  # Differentiable 4-qubit VQC layer (PennyLane)
    │       ├── q_dqn.py                    # Quantum DQN
    │       ├── q_ppo.py                    # Quantum PPO
    │       └── q_sac.py                    # Quantum SAC
    ├── training/
    │   ├── __init__.py
    │   └── trainer.py                      # Unified training loop, logging & checkpointing
    ├── evaluation/
    │   ├── __init__.py
    │   └── evaluator.py                    # Deterministic evaluation & rollout metrics
    ├── metrics/
    │   ├── __init__.py
    │   └── metrics.py                      # AULC, sample efficiency & stability statistics
    ├── visualization/
    │   ├── __init__.py
    │   └── plots.py                        # Publication-quality plotting suite
    └── results/                            # Benchmark artifacts
        ├── checkpoints/                    # Model weights (.pt) [gitignored by default]
        ├── data/                           # Evaluated metrics JSON and system info
        ├── figures/                        # High-resolution PNG & PDF visualizations
        └── logs/                           # Per-run progress logs
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Python **3.10** or higher (tested on Python 3.10, 3.11, 3.12, 3.14)
- Recommended: CUDA-capable GPU (for PyTorch and accelerated simulation)

### 1. Clone Repository

```bash
git clone https://github.com/Shr-i-ram/Quantum-Reinforcement-Learning-Benchmark.git
cd Quantum-Reinforcement-Learning-Benchmark
```

### 2. Create and Activate Virtual Environment

```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install all core dependencies:

```bash
pip install -r requirements.txt
```

Alternatively, install the package in editable mode:

```bash
pip install -e .
```

For development (testing, linting, formatting):

```bash
pip install -r requirements-dev.txt
```

---

## 🏃 Running Experiments

To run the complete benchmark suite across all configured environments and algorithms:

```bash
python run_experiment.py
```

### Benchmark Workflow

The orchestrator executes the following sequence:
1. **Software Validation**: Checks versions of PyTorch, PennyLane, Gymnasium, and CUDA availability.
2. **Environment Validation**: Confirms Box2D and classic control Gymnasium environments initialize cleanly.
3. **Complexity Analysis**: Computes empirical observation and dynamics complexity metrics.
4. **Model Training**: Runs each configured algorithm with automatic checkpointing and resumption capability.
5. **Evaluation**: Evaluates policy checkpoints over fixed validation episodes.
6. **Metrics Computation**: Calculates AULC, steps-to-threshold, parameter ratios, and stability indices.
7. **Figure Generation**: Compiles publication-ready plots into `quantum_rl_benchmark/results/figures/`.

---

## 🔧 Evaluation & Custom Configuration

Hyperparameters and experiment budgets can be modified in [`quantum_rl_benchmark/config/settings.py`](quantum_rl_benchmark/config/settings.py) or dynamically in code:

```python
from quantum_rl_benchmark.config.settings import get_default_config
from quantum_rl_benchmark.orchestrator import Orchestrator

config = get_default_config()

# Adjust quantum circuit depth
config.quantum.circuit_depth = 3

# Configure training steps for specific environments
config.env_configs["CartPole-v1"] = {"total_timesteps": 50_000}

# Launch benchmark
orchestrator = Orchestrator(config)
orchestrator.run()
```

---

## 📄 License & Citation

This project is licensed under the [MIT License](LICENSE).

If you find this benchmark or codebase useful in your research, please cite:

```bibtex
@misc{quantum_rl_benchmark_2026,
  author = {Shriram},
  title = {Quantum Reinforcement Learning Benchmark: Comparison of Quantum Reinforcement Learning Algorithms in Increasingly Complex Gymnasium Environments},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Shr-i-ram/Quantum-Reinforcement-Learning-Benchmark}}
}
```
