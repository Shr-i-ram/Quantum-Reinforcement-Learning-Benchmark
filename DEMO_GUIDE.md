# Demonstration & University Review Guide: Quantum RL Arena

**Project Title**: *Comparison of Quantum Reinforcement Learning Algorithms in Increasingly Complex Gymnasium Environments*  
**Authors**: Shriram Narayanan, Evelyn Maria Paulraj, and Koppesh P  
**Platform**: Quantum RL Arena (FastAPI + React TypeScript + PennyLane + PyTorch)

---

## 🚀 Quick Launch (Single Command)

From the project root:

```bash
# Option 1: Python Launcher (auto-detects virtualenv, builds if needed, opens browser)
python run_arena.py

# Option 2: Windows Batch Script
.\start_arena.bat
```

The application automatically starts the FastAPI backend server on `http://localhost:8000` and launches your default browser to the web platform.

---

## 🎓 Recommended 15-Minute University Review Presentation Sequence

Follow this curated sequence during your final project evaluation:

### Part 1: Problem Statement & Mission Control (2 Minutes)
1. Open the application at `http://localhost:8000`.
2. Review **Mission Control**:
   - Research question: *Does replacing classical neural network layers with variational quantum circuits (VQCs) provide an empirical advantage across scaling Gymnasium complexity?*
   - Highlight the **20/20 completed benchmark experiment matrix** across 5 environments (`CartPole-v1`, `MountainCar-v0`, `LunarLander-v3`, `Pendulum-v1`, `BipedalWalker-v3`).
   - Mention the authors and point out the IEEE conference paper ([`IEEE_Quantum_RL_Benchmark_Paper.pdf`](file:///C:/Users/Shriram/Desktop/python/RL_Project/IEEE_Quantum_RL_Benchmark_Paper.pdf)).
3. Click the **Guided Tour** button in the top navigation bar to display the built-in 8-step review sequence.

### Part 2: Live Incontrovertible Proof of Learning (3 Minutes)
1. Switch to **`🎮 Playground`**.
2. Select **`CartPole-v1`** and choose **`⭐ DQN (Demonstration Solved)`**.
3. Click **`Auto Play`** to watch the cart effortlessly balance the pole for the full **500 steps** (cumulative return: `+500.0`).
4. Point out the live telemetry:
   - State observation vector (`x`, `v`, `θ`, `ω`)
   - Q-values for discrete actions
   - Near-perfect balance proving that the RL bootstrap and training harness are fully functional.

### Part 3: Human Challenge Arena — Beat the AI (3 Minutes)
1. Switch to **`🏆 Challenges`**.
2. Select **`CartPole`** or **`MountainCar`**.
3. Use the keyboard arrow keys (or on-screen buttons) to manually balance the cart or oscillate the mountain car.
4. Try to beat the target score:
   - When the target score is met, celebration confetti fires!
   - Enter your name or reviewer initials and click **Record** to submit your score to the persistent Leaderboard.

### Part 4: Classical vs. Quantum Head-to-Head Duel (3 Minutes)
1. Switch to **`⚔️ Head-to-Head`**.
2. Select **`CartPole-v1`**, Model A: **`DQN (Demonstration Solved)`**, Model B: **`Q-DQN`**.
3. Click **`Execute Batch Duel`** (15 synchronized episodes):
   - Watch the server simulate identical seeds for both models.
   - Point out the win-loss split (Classical wins 100% of CartPole duels with mean return ~480 vs ~9.4).
   - Review the episode-by-episode table showing trajectory divergence.

### Part 5: Quantum Architecture & Live PennyLane Simulation (2 Minutes)
1. Switch to **`⚛️ Quantum Lab`**.
2. Move the **Input Feature Sliders** ($x_0, x_1, x_2, x_3$):
   - Point out that PennyLane is executing a real 4-qubit quantum circuit in real time!
   - Show how the Pauli-Z expectation values ($\langle Z_0 \rangle \dots \langle Z_3 \rangle$) update dynamically.
   - Review the 16-basis state probability distribution ($|0000\rangle \dots |1111\rangle$).
3. Scroll to **Computational Scaling & Hardware Bottlenecks**:
   - Explain that 4-qubit classical simulation (~0.05ms) is 1,000x faster than cloud QPU dispatch (50ms+).
   - Show the classical memory wall ($2^{30} \approx 16$ GB RAM) vs physical QPUs.
   - Explain the **Barren Plateau** hazard: $\text{Var}[\partial L/\partial \theta] \propto 1/2^n$.

### Part 6: Live Training Studio (2 Minutes)
1. Switch to **`🏋️ Training Studio`**.
2. Select **`CartPole-v1`**, Algorithm: **`DQN`**, Steps: **`1,500`**.
3. Click **`Start Real Training`**:
   - Point out that this is an actual background PyTorch training job running locally on the laptop.
   - Observe the live FPS throughput, elapsed timer, and episode reward learning curve updating in real time.
4. When finished, notice the saved checkpoint and click **`Inspect in Playground`** to test the newly trained weights.

### Part 7: Benchmark Explorer & Research Insights (1 Minute)
1. Switch to **`📊 Benchmarks`**:
   - Search and sort across the 20 benchmark configurations.
   - View the **Pareto Frontier** (Return vs Training Time).
   - View the **Circuit Ablations** and **Noise Robustness** tabs.
2. Switch to **`📑 Research Insights`**:
   - Highlight the 8 rigorous research questions answered with empirical data.
   - Click **`Export Full Research Briefing (.md)`** to demonstrate exportability.
   - Conclude with the scientific value of a defensible, thoroughly measured comparison study.
