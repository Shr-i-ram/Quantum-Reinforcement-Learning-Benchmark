# Project Audit: Quantum Reinforcement Learning Benchmark

**Project Title**: *Comparison of Quantum Reinforcement Learning Algorithms in Increasingly Complex Gymnasium Environments*  
**Date of Audit**: October 2026  
**Auditor**: Antigravity Autonomous Research & Engineering Agent  
**Repository Working Directory**: `C:\Users\Shriram\Desktop\python\RL_Project`  
**Python Runtime**: Python 3.14.3, PyTorch 2.13.0+cu130, PennyLane 0.45.1, Gymnasium 1.3.0  

---

## 1. Executive Summary & Research Context

This audit evaluates the empirical integrity, model checkpoints, algorithmic implementations, and evaluation pipelines of the **Quantum Reinforcement Learning (QRL) Benchmark**. The benchmark studies whether hybrid variational quantum circuits (VQCs) under a strict hardware-realistic **4-qubit cap** can compete with classical Deep Reinforcement Learning (DQN, PPO, SAC) across five complexity tiers of standard Gymnasium environments (`CartPole-v1`, `MountainCar-v0`, `LunarLander-v3`, `Pendulum-v1`, `BipedalWalker-v3`).

### Core Audit Verdict
- **All 20 intended experiments have valid, loadable checkpoints** in `quantum_rl_benchmark/results/checkpoints/` that successfully execute deterministic and stochastic inference.
- **The Quantum Circuit Architecture is genuine**: All quantum agents execute a 4-qubit parameterized ansatz with angle encoding, parameterized $R_y$ and $R_z$ rotations, entangling CNOT ladders, and analytic Pauli-Z expectation value measurements ($\langle Z_i \rangle \in [-1, 1]$) simulated via PennyLane's `lightning.qubit` with exact adjoint differentiation (`diff_method="adjoint"`).
- **Quantum Gradients Propagate Correctly**: An automated gradient test confirmed that $\nabla_\theta \mathcal{L}$ and $\nabla_x \mathcal{L}$ propagate without vanishing or exploding through the hybrid architecture.
- **Primary Learning Bottlenecks Identified**:
  1. *CartPole-v1 Classical DQN/PPO Plateau*: In the original training runs, classical DQN achieved periodic 500-step episodes during training, but evaluation returns plateaued at ~155 because evaluation and checkpoint saving only occurred once at the final step ($T=100,000$), without mid-training evaluation checkpointing or proper truncation bootstrap handling.
  2. *Continuous PPO Action Drift*: In classical continuous PPO, the unbounded linear policy head drifted to extreme action magnitudes ($|a| > 20$), whereas Q-PPO was intrinsically regularized by the bounded Pauli expectation values ($\langle Z_i \rangle \in [-1, 1]$), explaining Q-PPO's empirical superiority on `Pendulum-v1` and `BipedalWalker-v3`.
  3. *Dimensional Bottleneck*: For environments with observation dimensions $d > 4$ (`LunarLander-v3` with $d=8$, `BipedalWalker-v3` with $d=24$), the classical compression head (`QuantumFeatureProjection` $d \rightarrow 4$) creates an information bottleneck prior to quantum encoding.
  4. *Off-Policy Quantum Simulation Latency*: Q-DQN required ~15 to 18 hours per 100,000 steps due to repeated sequential circuit evaluations during off-policy replay updates, restricting update frequency compared to classical networks.

---

## 2. Checkpoint & Artifact Inventory

The audit verified all 20 primary benchmark experiment checkpoints by loading their weights, inspecting parameter counts, and stepping the actual Gymnasium environments:

| Environment | Action Domain | Algorithm | Regime | Total Steps | Checkpoint Status | Param Count | In-Memory Inference |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `CartPole-v1` | Discrete ($|\mathcal{A}|=2$) | **PPO** | Original | 100,000 | `best.pt` / `latest.pt` | 4,675 | Verified OK |
| `CartPole-v1` | Discrete ($|\mathcal{A}|=2$) | **Q-PPO** | Original | 100,000 | `best.pt` / `latest.pt` | 5,999 | Verified OK |
| `CartPole-v1` | Discrete ($|\mathcal{A}|=2$) | **DQN** | Original | 100,000 | `best.pt` / `latest.pt` | 17,410 | Verified OK |
| `CartPole-v1` | Discrete ($|\mathcal{A}|=2$) | **Q-DQN** | Original | 100,000 | `latest.pt` | 19,310 | Verified OK |
| `MountainCar-v0` | Discrete ($|\mathcal{A}|=3$) | **PPO** | Original | 100,000 | `best.pt` / `latest.pt` | 4,612 | Verified OK |
| `MountainCar-v0` | Discrete ($|\mathcal{A}|=3$) | **Q-PPO** | Original | 100,000 | `best.pt` / `latest.pt` | 5,748 | Verified OK |
| `MountainCar-v0` | Discrete ($|\mathcal{A}|=3$) | **DQN** | Original | 100,000 | `best.pt` / `latest.pt` | 17,283 | Verified OK |
| `MountainCar-v0` | Discrete ($|\mathcal{A}|=3$) | **Q-DQN** | Original | 100,000 | `latest.pt` | 19,059 | Verified OK |
| `LunarLander-v3` | Discrete ($|\mathcal{A}|=4$) | **PPO** | Original | 150,000 | `best.pt` / `latest.pt` | 5,061 | Verified OK |
| `LunarLander-v3` | Discrete ($|\mathcal{A}|=4$) | **Q-PPO** | Original | 150,000 | `best.pt` / `latest.pt` | 6,521 | Verified OK |
| `LunarLander-v3` | Discrete ($|\mathcal{A}|=4$) | **DQN** | Original | 150,000 | `best.pt` / `latest.pt` | 18,180 | Verified OK |
| `LunarLander-v3` | Discrete ($|\mathcal{A}|=4$) | **Q-DQN** | Optimized | 100,000 | `latest.pt` | 19,832 | Verified OK |
| `Pendulum-v1` | Continuous ($\mathcal{A}=[-2, 2]^1$) | **PPO** | Original | 100,000 | `best.pt` / `latest.pt` | 4,547 | Verified OK |
| `Pendulum-v1` | Continuous ($\mathcal{A}=[-2, 2]^1$) | **Q-PPO** | Original | 100,000 | `best.pt` / `latest.pt` | 5,867 | Verified OK |
| `Pendulum-v1` | Continuous ($\mathcal{A}=[-2, 2]^1$) | **SAC** | Optimized | 30,000 | `best.pt` / `latest.pt` | 51,844 | Verified OK |
| `Pendulum-v1` | Continuous ($\mathcal{A}=[-2, 2]^1$) | **Q-SAC** | Optimized | 20,000 | `best.pt` / `latest.pt` | 53,744 | Verified OK |
| `BipedalWalker-v3` | Continuous ($\mathcal{A}=[-1, 1]^4$) | **PPO** | Original | 150,000 | `best.pt` / `latest.pt` | 6,089 | Verified OK |
| `BipedalWalker-v3` | Continuous ($\mathcal{A}=[-1, 1]^4$) | **Q-PPO** | Original | 150,000 | `best.pt` / `latest.pt` | 8,573 | Verified OK |
| `BipedalWalker-v3` | Continuous ($\mathcal{A}=[-1, 1]^4$) | **SAC** | Optimized | 40,000 | `best.pt` / `latest.pt` | 61,450 | Verified OK |
| `BipedalWalker-v3` | Continuous ($\mathcal{A}=[-1, 1]^4$) | **Q-SAC** | Optimized | 25,000 | `best.pt` / `latest.pt` | 62,606 | Verified OK |

---

## 3. Algorithmic & Methodological Audit

### 3.1 Observation and Action Space Handling
- **Discrete Actions**: `CartPole-v1` (2 actions), `MountainCar-v0` (3 actions), `LunarLander-v3` (4 actions) are properly mapped via `argmax` (DQN/Q-DQN) or `torch.distributions.Categorical` (PPO/Q-PPO).
- **Continuous Actions**:
  - `Pendulum-v1`: Torque in $[-2.0, 2.0]$. SAC and Q-SAC use squashed Gaussian policy heads ($a = \text{bound} \cdot \tanh(u)$) with appropriate Jacobian log-prob correction $\log(1 - \tanh^2(u) + \epsilon) + \log(\text{bound})$.
  - `BipedalWalker-v3`: 4 joint torques in $[-1.0, 1.0]$.
  - *Bug identified in Classical Continuous PPO*: The Gaussian mean was computed via `actor_mean(shared_out) * action_bound` without a bounding nonlinearity (`tanh`). In long rollouts, `actor_mean` drifted to $>20.0$, generating saturated actions. In contrast, `Q-PPO` bounded inputs to the actor head via Pauli expectation values ($\langle Z_i \rangle \in [-1, 1]$), providing implicit regularization.

### 3.2 Termination vs. Truncation Handling
- In standard Gymnasium $\ge 0.29$, `env.step()` returns `(obs, reward, terminated, truncated, info)`.
- *Finding*: When an agent balances CartPole until the maximum episode limit ($T=500$), `truncated` is True, while `terminated` is False. Setting $done = terminated \lor truncated$ in the temporal difference target $r + \gamma \max Q(s', a')(1 - done)$ zeroes out the bootstrap value at step 500, penalizing long-horizon survival. Distinguishing termination from truncation allows the agent to bootstrap value across time-limit boundaries.

### 3.3 Evaluation Protocol
- The benchmark evaluates policies using **deterministic actions** (`deterministic=True`) across evaluation rollouts.
- All evaluation runs use fixed evaluation seeds (`seed + 1000 + episode_idx`), ensuring reproducibility.
- *Sample Counts*: Original regime experiments recorded $n=10$ evaluation episodes; optimized regime recorded $n=20$ episodes. Descriptive statistics reported are sample means, sample medians, and standard deviations.

### 3.4 Quantum Circuit Implementation & Resources
- **Qubit Count**: Strictly 4 qubits across all quantum agents (`Q-PPO`, `Q-DQN`, `Q-SAC`).
- **Circuit Depth**: 2 variational layers.
- **Gate Counts per Forward Pass**:
  - Angle encoding: 4 single-qubit $R_y$ gates.
  - Variational layers ($D=2$): $2 \times 4 \times 2 = 16$ rotation gates ($R_y, R_z$).
  - Entangling ladders ($D=2$): Forward CNOT ladder ($3$ gates) + Reverse CNOT ladder ($3$ gates) per layer $= 12$ CNOT gates.
  - Total gates per circuit evaluation: $32$ gates ($12$ entangling CNOTs, $20$ rotations).
- **Trainable Quantum Parameters**: $D \times N \times 2 = 2 \times 4 \times 2 = 16$ parameters.
- **Simulation Engine**: PennyLane with `lightning.qubit` C++ statevector simulator and adjoint differentiation.

---

## 4. Preservation & Extension Strategy

To maintain strict scientific integrity and avoid altering existing benchmark results:
1. **Preservation Guarantee**: All original 20 checkpoint files (`.pt`) and result files (`*_results.json`) remain unaltered.
2. **Demonstration Checkpoints**: Any newly trained reference policies (e.g. `DQN_CartPole-v1_demonstration.pt`) are stored under distinct filenames and logged in an external Demonstration registry rather than overwriting benchmark artifacts.
3. **Random and Heuristic Baselines**: Stored in a unified registry (`quantum_rl_benchmark/results/experiment_registry.json`) for systematic comparisons across all 5 environments.
4. **Controlled Ablation Suite**: Implemented as isolated, documented control experiments (e.g. 2-qubit vs 4-qubit, depth-1 vs depth-2, classical control head).

---

## 5. Audit Recommendations for the Research Platform

1. **Deploy Dedicated Demonstration Checkpoint**:
   - Save a verified classical CartPole solver reaching 500.0 (`DQN_CartPole-v1_demonstration.pt`) to provide an immediate, incontrovertible proof of learning for university reviewers.
2. **Implement Random and Heuristic Baselines**:
   - Provide concrete evidence that both classical and quantum models learn policies superior to untrained random agents and heuristic rules.
3. **Build the Interactive Research Live Demo**:
   - Develop a comprehensive Streamlit research laboratory with real-time environment rendering, HUD telemetry, live action inspection (including Pauli expectation values), human control mode, side-by-side arena, live training studio, and quantum circuit visualizer.
4. **Deploy Experiment Registry**:
   - Centralize all benchmark, baseline, and demonstration metadata in `experiment_registry.json`.
