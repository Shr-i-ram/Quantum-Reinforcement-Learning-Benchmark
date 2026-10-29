# Demonstration & University Review Guide

**Project Title**: *Comparison of Quantum Reinforcement Learning Algorithms in Increasingly Complex Gymnasium Environments*  
**Authors**: Shriram Narayanan, Evelyn Maria Paulraj, and Koppesh P  

---

## 🚀 Quick Launch (1-Minute Setup)

From the project root:

```bash
# 1. Activate the virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 2. Launch the Interactive Web Demonstration Platform
streamlit run app.py
```

The application will immediately open in your default browser at `http://localhost:8501`.

---

## 🎓 Recommended 15-Minute University Review Presentation Sequence

Follow this curated sequence during your final project evaluation:

### Part 1: Problem Statement & Motivation (2–3 Minutes)
1. Navigate to **`🚀 Mission Control`**.
2. Explain the core research question: *Can a compact 4-qubit variational circuit with 16 parameters compete with classical deep neural networks across scaling complexity tiers?*
3. Highlight the **20/20 completed experiment matrix** spanning 5 environments (`CartPole-v1`, `MountainCar-v0`, `LunarLander-v3`, `Pendulum-v1`, `BipedalWalker-v3`).
4. Point out the hard maximum cap of **4 qubits**, analytic adjoint differentiation via PennyLane `lightning.qubit`, and the IEEE conference paper ([`IEEE_Quantum_RL_Benchmark_Paper.pdf`](IEEE_Quantum_RL_Benchmark_Paper.pdf)).

### Part 2: Live Incontrovertible Proof of Learning (3 Minutes)
1. Navigate to **`🎮 Environment Playground`**.
2. Select **`CartPole-v1`** and choose **`DQN (Demonstration Solved Model - 500 Return)`**.
3. Click **`🏁 Run Episode`** to watch the cart effortlessly balance the pole for the full **500 steps** (cumulative return: `+500.0`).
4. Highlight that this establishes a verified, incontrovertible proof that the reinforcement learning pipeline and temporal-difference bootstrap mechanics are sound.

### Part 3: Classical vs. Quantum Head-to-Head Comparison (4 Minutes)
1. Navigate to **`⚔️ Head-to-Head Arena`**.
2. Select **`PPO vs Q-PPO on CartPole-v1`** or **`SAC vs Q-SAC on Pendulum-v1`**.
3. Under **Side-by-Side Visual Playback**, click **`🏁 Start Side-by-Side Duel`** to watch both agents simultaneously interact with the environment.
4. Switch to **Statistical Batch Duel**, select 15 episodes, and click **`🚀 Execute Batch Duel`**.
5. Explain the resulting Plotly box plots:
   - **Q-PPO Advantage**: Q-PPO outperforms Classical PPO across 4 of 5 environments (CartPole: 192.5 vs 108.4; LunarLander: -375.8 vs -568.4; Pendulum: -1353.8 vs -1409.9; BipedalWalker: -143.3 vs -165.4).
   - **Regularization Mechanism**: Pauli-Z expectation values ($\langle Z_i \rangle \in [-1, 1]$) prevent policy output saturation.

### Part 4: Quantum Circuit & Activation Inspection (3 Minutes)
1. Navigate to **`🔬 Quantum Architecture Lab`**.
2. Show the ASCII circuit schematic: 4 qubits, angle encoding $R_y(x_i)$, Depth 2 parameterized rotations ($R_y, R_z$), and entangling CNOT ladders.
3. Switch to the **Live Quantum Measurement Explorer** tab:
   - Move the input feature sliders ($x_0, x_1, x_2, x_3$).
   - Show how the real-time PennyLane circuit rotates the quantum statevector and produces analytic Pauli-Z expectation values $\langle Z_i \rangle$.
4. Open the **Simulator vs. Hardware Scaling** tab to explain that the wall-clock latency is due to classical statevector matrix simulation ($O(2^n)$ in software), not an inherent limitation of physical quantum hardware ($O(1)$ gate execution).

### Part 5: Controlled Ablations, Robustness & Conclusions (3 Minutes)
1. Navigate to **`🧩 Ablations & Robustness`**.
2. Point out the controlled variations:
   - **Depth 1 vs 2**: 16 gates vs 32 gates.
   - **Width (2 vs 4 Qubits)**: 4-dim vs 16-dim Hilbert state space.
   - **Observation Noise Perturbations**: Show that Q-PPO exhibits higher Lipschitz stability against Gaussian observation noise than classical DQN.
3. Conclude at **`💡 Research Insights`** and demonstrate the one-click **Download Research Report (.md)** button.

---

## 🛠️ Feature Walkthrough & Controls

| Tab / View | Key Features | Recommended Use |
|---|---|---|
| **🚀 Mission Control** | Status cards, 20-run matrix table, research hypotheses | High-level presentation opening |
| **🎮 Environment Playground** | 5 Gymnasium environments, Step/Play/Reset, HUD overlay, Deep Action Inspector, Human Challenge Mode | Live interactive demonstration |
| **⚔️ Head-to-Head Arena** | Side-by-side synchronized rollouts, batch return boxplots, latency comparison | Comparative algorithm analysis |
| **⚡ Live Training Studio** | Real background training loop, streaming loss/reward curves, safe step limits | Demonstrating live model convergence |
| **📊 Benchmark Explorer** | Interactive filters, return charts, Pareto frontier, heatmaps, master table | Data deep-dives and examiner Q&A |
| **🔬 Quantum Lab** | Qubit diagram, slider-controlled Pauli measurement explorer, simulation cost analysis | Explaining the quantum computing theory |
| **🧩 Ablations & Robustness** | Depth/width scaling tables, observation noise degradation curves | Scientific rigor and methodology proof |
| **💡 Research Insights** | Evidence-based answers to 10 research questions, Markdown report generator | Presentation closing and takeaways |
| **🎓 Guided Presentation Tour** | 8-step structured slideshow mode with Next/Previous buttons | Fail-safe rehearsed walkthrough |

---

## ⚠️ Troubleshooting & Recovery

- **Port in use (`address already in use: 8501`)**:
  ```bash
  streamlit run app.py --server.port=8502
  ```
- **Reset Demo State**:
  Click the **`🔄 Reset Demo State`** button in the sidebar to clear cached sessions.
- **Run without GUI/Internet**:
  All models, environments, and dependencies run 100% locally on CPU or GPU without external network access.

