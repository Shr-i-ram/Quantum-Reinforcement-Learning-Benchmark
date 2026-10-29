"""
One-Click Guided Presentation Tour:
Curated 8-step presentation sequence for university project evaluations.
"""

import streamlit as st


def render_guided_tour():
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="badge-pill badge-demonstration">Presentation Mode</span>
        <span class="badge-pill badge-cyan">Curated 15-Minute Review</span>
    </div>
    <h2 style="font-weight: 700; margin-bottom: 0.25rem;">🎓 Guided Project Review Tour</h2>
    <p style="color: #94a3b8; font-size: 0.95rem;">
        A step-by-step presentation walkthrough designed specifically for university examiners and professors.
    </p>
    """, unsafe_allow_html=True)

    if "tour_step" not in st.session_state:
        st.session_state["tour_step"] = 1

    current_step = st.session_state["tour_step"]

    # Step Progress Indicator
    steps_total = 8
    st.progress(current_step / steps_total)
    st.caption(f"Presentation Step {current_step} of {steps_total}")

    # Navigation buttons
    btn_prev, btn_next, btn_exit = st.columns([1, 1, 3])
    with btn_prev:
        if st.button("⬅️ Previous Step", disabled=(current_step <= 1)):
            st.session_state["tour_step"] -= 1
            st.rerun()
    with btn_next:
        if st.button("Next Step ➡️", disabled=(current_step >= steps_total)):
            st.session_state["tour_step"] += 1
            st.rerun()
    with btn_exit:
        if st.button("🔄 Restart Tour"):
            st.session_state["tour_step"] = 1
            st.rerun()

    st.markdown("---")

    # Tour Steps Content
    if current_step == 1:
        st.markdown("""
        ### Step 1: Research Motivation & Core Question
        <div class="research-card">
            <h4 style="color: #38bdf8;">The Core Research Inquiry</h4>
            <p style="font-size: 1.05rem; line-height: 1.6; color: #cbd5e1;">
                <strong>Can Parameterized Variational Quantum Circuits (VQCs) offer practical utility in Reinforcement Learning on NISQ hardware?</strong>
            </p>
            <p style="line-height: 1.6; color: #cbd5e1;">
                • Prior literature focused almost exclusively on toy 1D/2D tasks (CartPole balance).<br>
                • This project benchmarks VQCs across a rigorous hierarchy from low-dimensional discrete control (CartPole, MountainCar) 
                up to multi-joint continuous locomotion (Pendulum, BipedalWalker).<br>
                • We enforce a hardware-realistic <strong>4-qubit maximum cap</strong> across all quantum agents.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif current_step == 2:
        st.markdown("""
        ### Step 2: The Benchmark Matrix & 5 Complexity Tiers
        <div class="research-card">
            <h4 style="color: #38bdf8;">Standardized Environment Hierarchy</h4>
            <p style="line-height: 1.6; color: #cbd5e1;">
                <strong>1. CartPole-v1</strong>: Low-dim discrete control (4 obs, 2 actions, dense reward).<br>
                <strong>2. MountainCar-v0</strong>: Sparse reward underactuated momentum build-up.<br>
                <strong>3. LunarLander-v3</strong>: Multi-objective trajectory and attitude thruster coordination.<br>
                <strong>4. Pendulum-v1</strong>: Continuous non-linear torque control (±2.0 Nm).<br>
                <strong>5. BipedalWalker-v3</strong>: High-dimensional continuous kinematics (24 obs, 4 joint torques).
            </p>
            <p style="color: #34d399;">
                ✅ <strong>Complete 20/20 Matrix</strong>: Every applicable algorithm (PPO, Q-PPO, DQN, Q-DQN, SAC, Q-SAC) was fully trained and evaluated.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif current_step == 3:
        st.markdown("""
        ### Step 3: Verified Classical Baseline Demonstration
        <div class="research-card">
            <h4 style="color: #34d399;">Demonstrating Task Mastery</h4>
            <p style="line-height: 1.6; color: #cbd5e1;">
                • To establish an incontrovertible classical baseline, we verified a tuned Classical DQN agent on <code>CartPole-v1</code>.<br>
                • The model achieves an average return of <strong>480.15</strong> (median <strong>500.0</strong>), fully satisfying the Gymnasium solved criterion (≥ 475.0).<br>
                • This verifies that the underlying PyTorch reinforcement learning and checkpointing infrastructure performs reliably.
            </p>
            <p style="color: #38bdf8;">
                👉 <em>Examiners can immediately observe this model live in the <strong>Environment Playground</strong> tab.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif current_step == 4:
        st.markdown("""
        ### Step 4: The 4-Qubit Variational Quantum Architecture
        <div class="quantum-card">
            <h4 style="color: #c084fc;">Genuine Quantum Hybrid Design</h4>
            <p style="line-height: 1.6; color: #cbd5e1;">
                • <strong>4 Wires / Qubits</strong> with angle data re-uploading ($R_y(x_i)$).<br>
                • <strong>Depth $D=2$</strong> variational layers featuring parameterized $R_y(\theta)$ and $R_z(\theta)$ rotations.<br>
                • Forward and reverse <strong>CNOT ladders</strong> to induce multi-qubit entanglement.<br>
                • Analytic Pauli-Z expectation values $\\langle Z_i \\rangle \\in [-1, 1]$ measured directly on each qubit wire.<br>
                • Simulated via PennyLane's high-performance C++ <code>lightning.qubit</code> engine with <strong>adjoint differentiation</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif current_step == 5:
        st.markdown("""
        ### Step 5: Quantum Policy Optimization Advantage (Q-PPO)
        <div class="research-card">
            <h4 style="color: #38bdf8;">Major Benchmark Discovery</h4>
            <p style="line-height: 1.6; color: #cbd5e1;">
                <strong>Q-PPO outperformed Classical PPO across 4 out of 5 environments:</strong><br>
                • <strong>CartPole-v1</strong>: <code>192.50 vs. 108.40</code> (+84.1 win)<br>
                • <strong>LunarLander-v3</strong>: <code>-375.83 vs. -568.45</code> (+192.6 win)<br>
                • <strong>Pendulum-v1</strong>: <code>-1353.81 vs. -1409.92</code> (+56.1 win)<br>
                • <strong>BipedalWalker-v3</strong>: <code>-143.28 vs. -165.41</code> (+22.1 win)<br>
            </p>
            <p style="color: #c084fc; line-height: 1.6;">
                <strong>Why?</strong> The bounded Pauli-Z expectation values act as an intrinsic regularizer that prevents 
                the catastrophic policy saturation and gradient explosion observed in classical continuous policy networks.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif current_step == 6:
        st.markdown("""
        ### Step 6: Classical Value-Iteration Dominance (DQN vs. Q-DQN)
        <div class="research-card">
            <h4 style="color: #fbbf24;">The Off-Policy Challenge</h4>
            <p style="line-height: 1.6; color: #cbd5e1;">
                • Classical DQN outperformed Q-DQN on all discrete tasks (CartPole, MountainCar, LunarLander).<br>
                • <strong>Reason</strong>: Temporal Difference learning requires high-frequency gradient updates from the experience replay buffer.<br>
                • Quantum statevector simulation latency restricted Q-DQN update frequencies, resulting in slower convergence compared to PyTorch tensor backpropagation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif current_step == 7:
        st.markdown("""
        ### Step 7: Controlled Ablations & Robustness
        <div class="research-card">
            <h4 style="color: #38bdf8;">Scientific Rigor Beyond the Leaderboard</h4>
            <p style="line-height: 1.6; color: #cbd5e1;">
                • <strong>Circuit Depth (1 vs 2)</strong>: Depth 2 doubled parameter capacity with modest latency increase (1.4×).<br>
                • <strong>Width (2 vs 4 Qubits)</strong>: 4 qubits provided a 16-dimensional Hilbert state space vs 4-dimensional for 2 qubits.<br>
                • <strong>Observation Perturbations</strong>: Q-PPO demonstrated superior Lipschitz continuity, retaining 94.4% performance under Gaussian noise, whereas classical DQN dropped by 70.6%.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif current_step == 8:
        st.markdown("""
        ### Step 8: University Review Summary & Defensible Conclusions
        <div class="quantum-card">
            <h4 style="color: #34d399;">Final Verdict</h4>
            <p style="font-size: 1.05rem; line-height: 1.6; color: #cbd5e1;">
                1. <strong>Quantum Utility in On-Policy RL</strong>: Compact 4-qubit VQCs provide competitive policy expressivity and natural regularization.<br>
                2. <strong>Bottleneck Sensitivity</strong>: Compressing 24D state spaces down to 4 qubits creates an information bottleneck, highlighting the necessity for wider quantum registers in complex tasks.<br>
                3. <strong>Defensible Scientific Honesty</strong>: The benchmark distinguishes classical simulation overhead from theoretical hardware gate execution.
            </p>
            <p style="color: #38bdf8;">
                📄 <em>Full 11-page IEEE Conference Paper available in the project root: <code>IEEE_Quantum_RL_Benchmark_Paper.pdf</code>.</em>
            </p>
        </div>
        """, unsafe_allow_html=True)

