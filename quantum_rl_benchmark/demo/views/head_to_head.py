"""
Classical vs. Quantum Head-to-Head Arena:
Side-by-side rollout comparisons and statistical batch evaluations.
"""

import time
import streamlit as st
import numpy as np
import gymnasium as gym
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantum_rl_benchmark.config.settings import ENVIRONMENTS, EXPERIMENT_MATRIX
from quantum_rl_benchmark.demo.engine import (
    get_cached_agent, render_annotated_frame, step_policy_with_inspection
)


def render_head_to_head():
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="badge-pill badge-classical">Classical Baseline</span>
        <span style="font-weight: bold; color: #94a3b8; margin: 0 0.5rem;">VS</span>
        <span class="badge-pill badge-quantum">Hybrid Quantum VQC</span>
    </div>
    <h2 style="font-weight: 700; margin-bottom: 0.25rem;">⚔️ Head-to-Head Comparison Arena</h2>
    <p style="color: #94a3b8; font-size: 0.95rem;">
        Directly benchmark paired classical and quantum algorithms under identical environment conditions.
    </p>
    """, unsafe_allow_html=True)

    # Comparison Pair Selector
    pairs = [
        ("PPO", "Q-PPO", "CartPole-v1"),
        ("DQN", "Q-DQN", "CartPole-v1"),
        ("PPO", "Q-PPO", "MountainCar-v0"),
        ("DQN", "Q-DQN", "MountainCar-v0"),
        ("DQN", "Q-DQN", "LunarLander-v3"),
        ("PPO", "Q-PPO", "LunarLander-v3"),
        ("SAC", "Q-SAC", "Pendulum-v1"),
        ("PPO", "Q-PPO", "Pendulum-v1"),
        ("SAC", "Q-SAC", "BipedalWalker-v3"),
        ("PPO", "Q-PPO", "BipedalWalker-v3"),
    ]

    pair_labels = [f"{c} vs {q} on {env}" for c, q, env in pairs]
    selected_pair_idx = st.selectbox("Select Benchmark Duel", range(len(pairs)), format_func=lambda i: pair_labels[i])
    algo_c, algo_q, env_name = pairs[selected_pair_idx]

    env_spec = ENVIRONMENTS[env_name]

    tab_side_by_side, tab_batch_eval = st.tabs(["📺 Side-by-Side Visual Playback", "📊 Statistical Batch Duel"])

    # --- Mode A: Side-by-Side Visual Playback ---
    with tab_side_by_side:
        st.markdown(f"#### Visual Rollout: {algo_c} (Classical) vs. {algo_q} (Quantum)")
        st.caption(f"Environment: {env_name} | Observation Dim: {env_spec['obs_dim']} | Action Domain: {env_spec['action_type'].capitalize()}")

        c_view_left, c_view_right = st.columns(2)
        with c_view_left:
            st.markdown(f"<h4 style='color: #38bdf8; text-align: center;'>{algo_c} (Classical)</h4>", unsafe_allow_html=True)
            screen_c = st.empty()
            metric_c = st.empty()

        with c_view_right:
            st.markdown(f"<h4 style='color: #c084fc; text-align: center;'>{algo_q} (4-Qubit Quantum)</h4>", unsafe_allow_html=True)
            screen_q = st.empty()
            metric_q = st.empty()

        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            run_duel = st.button("🏁 Start Side-by-Side Duel", use_container_width=True)
        with col_b2:
            duel_steps = st.slider("Max Episode Steps", 50, env_spec["max_episode_steps"], min(200, env_spec["max_episode_steps"]))

        if run_duel:
            agent_c = get_cached_agent(algo_c, env_name)
            agent_q = get_cached_agent(algo_q, env_name)

            env_c = gym.make(env_spec["gymnasium_id"], render_mode="rgb_array")
            env_q = gym.make(env_spec["gymnasium_id"], render_mode="rgb_array")

            # Reset both with identical starting seed
            obs_c, _ = env_c.reset(seed=42)
            obs_q, _ = env_q.reset(seed=42)

            ret_c = 0.0
            ret_q = 0.0
            done_c = False
            done_q = False

            model_info_c = {"algo": algo_c, "type": "benchmark", "is_quantum": False, "id": f"Benchmark_{algo_c}"}
            model_info_q = {"algo": algo_q, "type": "benchmark", "is_quantum": True, "id": f"Benchmark_{algo_q}"}

            for step in range(duel_steps):
                if done_c and done_q:
                    break

                # Step Classical
                if not done_c:
                    raw_c, step_c, insp_c = step_policy_with_inspection(env_name, model_info_c, obs_c, env_c, deterministic=True)
                    obs_c, r_c, term_c, trunc_c, _ = env_c.step(step_c)
                    ret_c += float(r_c)
                    done_c = term_c or trunc_c
                    frame_c = env_c.render()
                    img_c = render_annotated_frame(frame_c, env_name, algo_c, step, raw_c, float(r_c), ret_c)
                    screen_c.image(img_c, use_container_width=True)
                    metric_c.metric("Return", f"{ret_c:+.1f}", delta=f"{r_c:+.2f}")

                # Step Quantum
                if not done_q:
                    raw_q, step_q, insp_q = step_policy_with_inspection(env_name, model_info_q, obs_q, env_q, deterministic=True)
                    obs_q, r_q, term_q, trunc_q, _ = env_q.step(step_q)
                    ret_q += float(r_q)
                    done_q = term_q or trunc_q
                    frame_q = env_q.render()
                    img_q = render_annotated_frame(frame_q, env_name, algo_q, step, raw_q, float(r_q), ret_q, quantum_expvals=insp_q.get("quantum_expectations"))
                    screen_q.image(img_q, use_container_width=True)
                    metric_q.metric("Return", f"{ret_q:+.1f}", delta=f"{r_q:+.2f}")

                time.sleep(0.015)

            env_c.close()
            env_q.close()

            # Final Verdict
            if ret_c > ret_q:
                st.info(f"🏆 Classical {algo_c} won this rollout ({ret_c:+.1f} vs {ret_q:+.1f})")
            elif ret_q > ret_c:
                st.success(f"🏆 Hybrid Quantum {algo_q} won this rollout ({ret_q:+.1f} vs {ret_c:+.1f})")
            else:
                st.warning(f"🤝 Tie score ({ret_c:+.1f})")

    # --- Mode B: Statistical Batch Duel ---
    with tab_batch_eval:
        st.markdown(f"#### Deterministic Batch Evaluation: {algo_c} vs. {algo_q}")
        st.caption("Runs independent seeded evaluation episodes and compares the return distributions.")

        n_duel_episodes = st.slider("Evaluation Episodes", 5, 30, 15, key="batch_eval_n_eps")
        if st.button("🚀 Execute Batch Duel", key="run_batch_duel_btn"):
            with st.spinner("Evaluating agents..."):
                agent_c = get_cached_agent(algo_c, env_name)
                agent_q = get_cached_agent(algo_q, env_name)

                env_c = gym.make(env_spec["gymnasium_id"])
                env_q = gym.make(env_spec["gymnasium_id"])

                returns_c = []
                returns_q = []
                latencies_c = []
                latencies_q = []

                # Evaluate Classical
                for ep in range(n_duel_episodes):
                    obs, _ = env_c.reset(seed=3000 + ep)
                    done = False
                    ep_ret = 0
                    t_start = time.perf_counter()
                    while not done:
                        act = agent_c.select_action(obs, deterministic=True)
                        if isinstance(act, tuple): act = act[0]
                        step_act = act if env_spec["action_type"] == "continuous" else int(act)
                        obs, r, term, trunc, _ = env_c.step(step_act)
                        ep_ret += float(r)
                        done = term or trunc
                    latencies_c.append((time.perf_counter() - t_start) * 1000.0)
                    returns_c.append(ep_ret)

                # Evaluate Quantum
                for ep in range(n_duel_episodes):
                    obs, _ = env_q.reset(seed=3000 + ep)
                    done = False
                    ep_ret = 0
                    t_start = time.perf_counter()
                    while not done:
                        act = agent_q.select_action(obs, deterministic=True)
                        if isinstance(act, tuple): act = act[0]
                        step_act = act if env_spec["action_type"] == "continuous" else int(act)
                        obs, r, term, trunc, _ = env_q.step(step_act)
                        ep_ret += float(r)
                        done = term or trunc
                    latencies_q.append((time.perf_counter() - t_start) * 1000.0)
                    returns_q.append(ep_ret)

                env_c.close()
                env_q.close()

            # Display comparative stats
            st.markdown("##### 📊 Comparative Summary Table")
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                st.markdown(f"""
                <div class="research-card">
                    <h4 style="color: #38bdf8;">{algo_c} (Classical)</h4>
                    <p>• <strong>Mean Return</strong>: <code>{np.mean(returns_c):.2f} ± {np.std(returns_c):.2f}</code></p>
                    <p>• <strong>Median Return</strong>: <code>{np.median(returns_c):.2f}</code></p>
                    <p>• <strong>Min / Max</strong>: <code>{np.min(returns_c):.1f} / {np.max(returns_c):.1f}</code></p>
                    <p>• <strong>Inference / Ep</strong>: <code>{np.mean(latencies_c):.1f} ms</code></p>
                </div>
                """, unsafe_allow_html=True)

            with col_sc2:
                st.markdown(f"""
                <div class="quantum-card">
                    <h4 style="color: #c084fc;">{algo_q} (Hybrid Quantum)</h4>
                    <p>• <strong>Mean Return</strong>: <code>{np.mean(returns_q):.2f} ± {np.std(returns_q):.2f}</code></p>
                    <p>• <strong>Median Return</strong>: <code>{np.median(returns_q):.2f}</code></p>
                    <p>• <strong>Min / Max</strong>: <code>{np.min(returns_q):.1f} / {np.max(returns_q):.1f}</code></p>
                    <p>• <strong>Inference / Ep</strong>: <code>{np.mean(latencies_q):.1f} ms</code></p>
                </div>
                """, unsafe_allow_html=True)

            # Boxplot comparison
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=returns_c,
                name=f"{algo_c} (Classical)",
                marker_color="#38bdf8",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8
            ))
            fig.add_trace(go.Box(
                y=returns_q,
                name=f"{algo_q} (Hybrid Quantum)",
                marker_color="#c084fc",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8
            ))
            fig.update_layout(
                title=f"Evaluation Return Distribution: {algo_c} vs {algo_q} on {env_name} (n={n_duel_episodes})",
                yaxis_title="Episode Cumulative Return",
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1")
            )
            st.plotly_chart(fig, use_container_width=True)

