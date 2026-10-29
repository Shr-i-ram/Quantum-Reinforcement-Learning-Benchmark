"""
Quantum Lab: Interactive circuit architecture inspector, PennyLane diagram,
measurement explorer, and simulation cost analysis.
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import pennylane as qml

from quantum_rl_benchmark.models.quantum.circuit import QuantumCircuitLayer


def render_quantum_lab():
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="badge-pill badge-quantum">PennyLane Lightning</span>
        <span class="badge-pill badge-purple">4-Qubit Architecture</span>
    </div>
    <h2 style="font-weight: 700; margin-bottom: 0.25rem;">🔬 Quantum Architecture Lab</h2>
    <p style="color: #94a3b8; font-size: 0.95rem;">
        Inspect the genuine 4-qubit variational quantum circuit (VQC), angle encoding dynamics, and statevector simulation mechanics.
    </p>
    """, unsafe_allow_html=True)

    q_tab1, q_tab2, q_tab3 = st.tabs(["📐 Circuit Architecture Inspector", "🎛️ Live Quantum Measurement Explorer", "⚙️ Simulator vs. Hardware Scaling"])

    # --- Tab 1: Circuit Inspector ---
    with q_tab1:
        st.markdown("### 4-Qubit Variational Quantum Circuit (VQC) Design")
        st.markdown("""
        All quantum agents in this benchmark adhere strictly to a **4-qubit hard cap**, ensuring compatibility with NISQ-era physical hardware.
        Simulation is executed via PennyLane's C++ statevector simulator (`lightning.qubit`) with exact **analytic adjoint differentiation**.
        """)

        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            st.metric("Total Qubits", "4 Qubits", help="Fixed across all hybrid agents")
        with col_q2:
            st.metric("Variational Depth", "2 Layers", help="Repeated rotation + entangling ladder")
        with col_q3:
            st.metric("Quantum Parameters", "16 Weights", help="2 rotations * 4 qubits * 2 layers")
        with col_q4:
            st.metric("Entangling CNOTs", "12 Gates", help="6 forward/reverse CNOTs per layer")

        st.markdown("##### ⚡ ASCII Circuit Architecture")
        st.code("""
Features x (4D) ──[ RY(x0) ]──[ RY(θ0,0) ]──[ RZ(θ0,1) ]──●────×───[ M(Z0) ]──> ⟨Z0⟩ ∈ [-1, 1]
Features x (4D) ──[ RY(x1) ]──[ RY(θ1,0) ]──[ RZ(θ1,1) ]──X──●─│───[ M(Z1) ]──> ⟨Z1⟩ ∈ [-1, 1]
Features x (4D) ──[ RY(x2) ]──[ RY(θ2,0) ]──[ RZ(θ2,1) ]──│──X─●───[ M(Z2) ]──> ⟨Z2⟩ ∈ [-1, 1]
Features x (4D) ──[ RY(x3) ]──[ RY(θ3,0) ]──[ RZ(θ3,1) ]──│────X───[ M(Z3) ]──> ⟨Z3⟩ ∈ [-1, 1]
                                [ Entangling CNOT Ladder ]
        """, language="text")

        st.markdown("##### 🧬 Hybrid Data Pipeline Flow")
        st.markdown("""
        1. **Classical Projection**: Raw observations $\mathbf{s} \in \mathbb{R}^d$ pass through `QuantumFeatureProjection` (with `Tanh` nonlinearity) bounding features to $\mathbf{x} \in [-1, 1]^4$.
        2. **Angle Encoding**: Data re-uploading via single-qubit Pauli-Y rotations: $R_y(x_i) = \exp(-i \frac{x_i}{2} Y)$.
        3. **Variational Entanglement**: Depth $D=2$ single-qubit rotations ($R_y, R_z$) coupled with forward-and-reverse CNOT ladders.
        4. **Pauli-Z Measurement**: Exact analytic expectation values $\langle Z_i \rangle = \langle \psi | \sigma_z^{(i)} | \psi \rangle \in [-1, 1]$ measured on all 4 qubits.
        5. **Classical Action Head**: Linear mapping to action logits (discrete) or Gaussian mean/std (continuous).
        """)

    # --- Tab 2: Live Measurement Explorer ---
    with q_tab2:
        st.markdown("### 🎛️ Interactive Quantum Measurement Explorer")
        st.caption("Adjust the 4 input features to watch real-time quantum state rotation and Pauli-Z expectation value changes:")

        sl_c1, sl_c2, sl_c3, sl_c4 = st.columns(4)
        with sl_c1:
            x0 = st.slider("Feature x0", -1.0, 1.0, 0.2, step=0.05)
        with sl_c2:
            x1 = st.slider("Feature x1", -1.0, 1.0, -0.4, step=0.05)
        with sl_c3:
            x2 = st.slider("Feature x2", -1.0, 1.0, 0.7, step=0.05)
        with sl_c4:
            x3 = st.slider("Feature x3", -1.0, 1.0, -0.1, step=0.05)

        # Run forward pass through QuantumCircuitLayer
        q_layer = QuantumCircuitLayer(n_qubits=4, depth=2, differentiation_method="adjoint")
        test_in = torch.tensor([x0, x1, x2, x3], dtype=torch.float32)
        with torch.no_grad():
            out_expvals = q_layer(test_in).numpy()

        st.markdown("##### 📊 Measured Expectation Values $\\langle Z_i \\rangle$:")
        fig = go.Figure(go.Bar(
            x=[f"Qubit {i} (Wire {i})" for i in range(4)],
            y=out_expvals,
            marker=dict(color=["#c084fc", "#a855f7", "#9333ea", "#7e22ce"]),
            text=[f"{v:+.4f}" for v in out_expvals],
            textposition="auto"
        ))
        fig.update_layout(
            yaxis=dict(range=[-1.1, 1.1], title="Expectation Value ⟨Z⟩"),
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Tab 3: Simulator vs Hardware ---
    with q_tab3:
        st.markdown("### ⚙️ Classical Simulation vs. Physical Quantum Hardware")
        st.markdown("""
        <div class="research-card">
            <h4 style="color: #38bdf8;">Scientific Disclosure: Simulation vs. Hardware Advantage</h4>
            <p style="color: #cbd5e1; line-height: 1.6;">
                This project investigates <strong>algorithmic properties, policy optimization dynamics, and expressivity</strong> 
                of hybrid quantum reinforcement learning architectures. All experiments were conducted using PennyLane's 
                <code>lightning.qubit</code> statevector simulator.
            </p>
            <p style="color: #cbd5e1; line-height: 1.6;">
                <strong>Key Implications:</strong>
                <br>• On classical hardware, simulating an $n$-qubit quantum circuit requires tracking $2^n$ complex amplitudes in memory. 
                For $n=4$, the statevector size is only $2^4 = 16$ amplitudes.
                <br>• The wall-clock overhead observed in this benchmark is an artifact of classical matrix simulation of quantum states, 
                not an inherent limitation of physical quantum execution.
                <br>• Physical execution on NISQ quantum processors would execute in microsecond gate times ($O(1)$ scaling in hardware), 
                but would introduce shot noise, gate infidelities, and decoherence.
            </p>
        </div>
        """, unsafe_allow_html=True)
