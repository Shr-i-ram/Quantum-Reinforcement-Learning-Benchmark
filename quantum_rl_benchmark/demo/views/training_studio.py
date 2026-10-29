"""
Live Training Studio: Launch and monitor lightweight training jobs in real time.
"""

import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from quantum_rl_benchmark.config.settings import ENVIRONMENTS, EXPERIMENT_MATRIX
from quantum_rl_benchmark.demo.training_worker import TRAINING_JOB


def render_training_studio():
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="badge-pill badge-demonstration">Background Execution</span>
        <span class="badge-pill badge-amber">Interactive Optimization</span>
    </div>
    <h2 style="font-weight: 700; margin-bottom: 0.25rem;">⚡ Live Training Studio</h2>
    <p style="color: #94a3b8; font-size: 0.95rem;">
        Launch real-time training jobs outside the UI loop with live streaming reward curves, asynchronous execution, and resource safeguards.
    </p>
    """, unsafe_allow_html=True)

    # Job Configuration
    st.markdown("### ⚙️ Training Job Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        env_names = list(ENVIRONMENTS.keys())
        env_choice = st.selectbox("Target Environment", env_names, index=0, key="train_env")

    algos = EXPERIMENT_MATRIX.get(env_choice, [])
    with col2:
        algo_choice = st.selectbox("Algorithm", algos, index=0, key="train_algo")

    with col3:
        # Resource safeguards: default budget is kept safe for fast execution
        budget_options = [2000, 5000, 10000, 20000]
        step_budget = st.select_slider("Step Budget (Steps)", options=budget_options, value=5000)

    col4, col5 = st.columns(2)
    with col4:
        lr_options = [1e-4, 3e-4, 1e-3, 3e-3]
        lr_choice = st.select_slider("Learning Rate", options=lr_options, value=1e-3)
    with col5:
        seed_choice = st.number_input("Random Seed", min_value=0, max_value=9999, value=42)

    # Launch & Stop Controls
    st.markdown("---")
    btn_c1, btn_c2, btn_c3 = st.columns([1.5, 1.5, 3])

    with btn_c1:
        if st.button("🚀 Launch Training Job", disabled=TRAINING_JOB.is_running, use_container_width=True):
            try:
                TRAINING_JOB.start(
                    algorithm=algo_choice,
                    env_name=env_choice,
                    total_steps=step_budget,
                    learning_rate=lr_choice,
                    seed=seed_choice
                )
                st.toast(f"Training started: {algo_choice} on {env_choice}", icon="🚀")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to launch training job: {e}")

    with btn_c2:
        if st.button("🛑 Stop Job", disabled=not TRAINING_JOB.is_running, use_container_width=True):
            TRAINING_JOB.stop()
            st.warning("Stop signal sent to background training worker.")

    with btn_c3:
        if TRAINING_JOB.is_running:
            st.info(f"⏳ Training in progress: **{TRAINING_JOB.status['algorithm']}** on **{TRAINING_JOB.status['environment']}**")
        else:
            st.caption("Training engine is idle and ready for job dispatch.")

    # Status & Telemetry
    st.markdown("### 📊 Real-Time Training Progress")
    stat = TRAINING_JOB.status

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Steps", f"{stat['current_step']} / {stat['total_steps']}")
    with m2:
        st.metric("Episodes", stat["episodes"])
    with m3:
        st.metric("Moving Avg Return (10)", f"{stat['avg_reward_10']:.2f}")
    with m4:
        st.metric("Elapsed Time", f"{stat['elapsed_time_s']:.1f}s")

    # Progress bar
    pct = 0.0
    if stat["total_steps"] > 0:
        pct = min(1.0, max(0.0, stat["current_step"] / stat["total_steps"]))
    st.progress(pct)

    # Live Learning Curve Plot
    hist = stat.get("history", {})
    ep_rews = hist.get("episode_rewards", [])

    if ep_rews:
        x_eps = list(range(1, len(ep_rews) + 1))
        fig = go.Figure()

        # Raw rewards
        fig.add_trace(go.Scatter(
            x=x_eps,
            y=ep_rews,
            mode="lines+markers",
            name="Episode Return",
            line=dict(color="#38bdf8", width=1.5),
            marker=dict(size=4)
        ))

        # 5-episode moving average
        if len(ep_rews) >= 5:
            ma5 = np.convolve(ep_rews, np.ones(5)/5, mode="valid")
            fig.add_trace(go.Scatter(
                x=list(range(5, len(ep_rews) + 1)),
                y=ma5,
                mode="lines",
                name="5-Ep Moving Avg",
                line=dict(color="#34d399", width=2.5)
            ))

        fig.update_layout(
            title=f"Live Training Learning Curve: {stat['algorithm']} on {stat['environment']}",
            xaxis_title="Completed Episode",
            yaxis_title="Cumulative Episode Return",
            height=340,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No episode rewards recorded yet. Progress curve will populate as training completes episodes.")

    # Saved Checkpoint Notification
    if stat.get("saved_checkpoint_path") and not TRAINING_JOB.is_running:
        st.success(f"✅ Training completed! Model saved to `{stat['saved_checkpoint_path']}`")
