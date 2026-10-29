"""
Environment Playground: Interactive model execution, telemetry, action inspection, and human challenge mode.
"""

import time
import streamlit as st
import numpy as np
import gymnasium as gym
import plotly.graph_objects as go

from quantum_rl_benchmark.config.settings import ENVIRONMENTS
from quantum_rl_benchmark.demo.engine import (
    get_compatible_models, step_policy_with_inspection, render_annotated_frame
)


def render_playground():
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="badge-pill badge-demonstration">Interactive Arena</span>
        <span class="badge-pill badge-cyan">Real-Time Simulation</span>
    </div>
    <h2 style="font-weight: 700; margin-bottom: 0.25rem;">🎮 Environment Playground</h2>
    <p style="color: #94a3b8; font-size: 0.95rem;">
        Select an environment and algorithm to watch live policy execution, inspect quantum expectation values, or test your skills in human challenge mode.
    </p>
    """, unsafe_allow_html=True)

    # 1. Selection Row
    col_env, col_model, col_mode = st.columns([1.2, 1.8, 1.0])

    with col_env:
        env_names = list(ENVIRONMENTS.keys())
        selected_env = st.selectbox(
            "Select Environment",
            env_names,
            index=0,
            key="playground_env_select"
        )

    compatible_models = get_compatible_models(selected_env)
    model_options = {m["name"]: m for m in compatible_models}

    with col_model:
        selected_model_name = st.selectbox(
            "Select Agent / Baseline",
            list(model_options.keys()),
            index=0,
            key="playground_model_select"
        )
        selected_model_info = model_options[selected_model_name]

    with col_mode:
        control_mode = st.radio(
            "Control Mode",
            ["Autonomous Agent", "Human Challenge Mode"],
            horizontal=False,
            key="playground_control_mode"
        )

    env_spec = ENVIRONMENTS[selected_env]

    # Session state initialization for environment simulation
    state_key = f"sim_{selected_env}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "env": None,
            "obs": None,
            "step": 0,
            "cum_reward": 0.0,
            "done": False,
            "term": False,
            "trunc": False,
            "last_action": None,
            "last_reward": 0.0,
            "last_inspection": None,
            "history_rewards": [],
        }

    sim = st.session_state[state_key]

    # Ensure environment is initialized
    if sim["env"] is None:
        sim["env"] = gym.make(env_spec["gymnasium_id"], render_mode="rgb_array")
        sim["obs"], _ = sim["env"].reset(seed=42)
        sim["step"] = 0
        sim["cum_reward"] = 0.0
        sim["done"] = False
        sim["term"] = False
        sim["trunc"] = False
        sim["last_action"] = "Reset"
        sim["last_reward"] = 0.0
        sim["last_inspection"] = None
        sim["history_rewards"] = []

    st.markdown("---")

    # Layout: Screen on left (60%), Telemetry & Action Inspector on right (40%)
    col_screen, col_telemetry = st.columns([1.3, 1.0])

    with col_screen:
        screen_placeholder = st.empty()
        # Render initial or current frame
        frame = sim["env"].render()
        q_expvals = sim["last_inspection"].get("quantum_expectations") if sim["last_inspection"] else None
        annotated_img = render_annotated_frame(
            frame,
            selected_env,
            selected_model_info["algo"],
            sim["step"],
            sim["last_action"],
            sim["last_reward"],
            sim["cum_reward"],
            quantum_expvals=q_expvals
        )
        screen_placeholder.image(annotated_img, use_container_width=True)

        # Controls Bar
        if control_mode == "Autonomous Agent":
            st.markdown("##### ⏯️ Playback Controls")
            btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns(5)

            with btn_col1:
                step_clicked = st.button("▶ Step Once", use_container_width=True)
            with btn_col2:
                run_10_clicked = st.button("⏩ Run 10 Steps", use_container_width=True)
            with btn_col3:
                run_50_clicked = st.button("🚀 Run 50 Steps", use_container_width=True)
            with btn_col4:
                run_full_clicked = st.button("🏁 Run Episode", use_container_width=True)
            with btn_col5:
                reset_clicked = st.button("🔄 Reset Env", use_container_width=True)

            c_opt1, c_opt2 = st.columns(2)
            with c_opt1:
                deterministic = st.checkbox("Deterministic Policy", value=True, help="Disable stochastic sampling / epsilon exploration")
            with c_opt2:
                delay_ms = st.slider("Simulation Delay (ms)", 0, 100, 20, step=10)

            # Handle Actions
            if reset_clicked:
                sim["obs"], _ = sim["env"].reset()
                sim["step"] = 0
                sim["cum_reward"] = 0.0
                sim["done"] = False
                sim["term"] = False
                sim["trunc"] = False
                sim["last_action"] = "Reset"
                sim["last_reward"] = 0.0
                sim["last_inspection"] = None
                sim["history_rewards"] = []
                st.rerun()

            steps_to_run = 0
            if step_clicked and not sim["done"]:
                steps_to_run = 1
            elif run_10_clicked and not sim["done"]:
                steps_to_run = 10
            elif run_50_clicked and not sim["done"]:
                steps_to_run = 50
            elif run_full_clicked and not sim["done"]:
                steps_to_run = env_spec["max_episode_steps"]

            if steps_to_run > 0:
                for _ in range(steps_to_run):
                    if sim["done"]:
                        break
                    raw_act, step_act, insp = step_policy_with_inspection(
                        selected_env, selected_model_info, sim["obs"], sim["env"],
                        deterministic=deterministic
                    )
                    next_obs, r, term, trunc, _ = sim["env"].step(step_act)
                    sim["obs"] = next_obs
                    sim["step"] += 1
                    sim["last_reward"] = float(r)
                    sim["cum_reward"] += float(r)
                    sim["last_action"] = raw_act
                    sim["last_inspection"] = insp
                    sim["history_rewards"].append(float(r))
                    sim["done"] = term or trunc
                    sim["term"] = term
                    sim["trunc"] = trunc

                    # Update screen frame dynamically
                    frame = sim["env"].render()
                    q_exp = insp.get("quantum_expectations")
                    annotated_img = render_annotated_frame(
                        frame, selected_env, selected_model_info["algo"],
                        sim["step"], raw_act, float(r), sim["cum_reward"],
                        quantum_expvals=q_exp
                    )
                    screen_placeholder.image(annotated_img, use_container_width=True)

                    if delay_ms > 0 and steps_to_run > 1:
                        time.sleep(delay_ms / 1000.0)

                st.rerun()

        else:
            # Human Challenge Mode
            st.markdown("##### 🕹️ Human Challenge Controls")
            st.caption("Press action buttons to interact with the environment in real time:")

            h_action = None
            if selected_env == "CartPole-v1":
                h_c1, h_c2, h_c3 = st.columns([1, 1, 1])
                with h_c1:
                    if st.button("⬅️ Push Left (Action 0)", use_container_width=True):
                        h_action = 0
                with h_c2:
                    if st.button("➡️ Push Right (Action 1)", use_container_width=True):
                        h_action = 1
                with h_c3:
                    if st.button("🔄 Reset", use_container_width=True):
                        sim["obs"], _ = sim["env"].reset()
                        sim["step"] = 0
                        sim["cum_reward"] = 0.0
                        sim["done"] = False
                        st.rerun()

            elif selected_env == "MountainCar-v0":
                h_c1, h_c2, h_c3, h_c4 = st.columns(4)
                with h_c1:
                    if st.button("⬅️ Push Left (0)", use_container_width=True): h_action = 0
                with h_c2:
                    if st.button("⏸️ Coast (1)", use_container_width=True): h_action = 1
                with h_c3:
                    if st.button("➡️ Push Right (2)", use_container_width=True): h_action = 2
                with h_c4:
                    if st.button("🔄 Reset", use_container_width=True):
                        sim["obs"], _ = sim["env"].reset()
                        sim["step"] = 0
                        sim["cum_reward"] = 0.0
                        sim["done"] = False
                        st.rerun()

            elif selected_env == "LunarLander-v3":
                h_c1, h_c2, h_c3, h_c4 = st.columns(4)
                with h_c1:
                    if st.button("🔥 Main Engine (2)", use_container_width=True): h_action = 2
                with h_c2:
                    if st.button("⬅️ Left Thruster (1)", use_container_width=True): h_action = 1
                with h_c3:
                    if st.button("➡️ Right Thruster (3)", use_container_width=True): h_action = 3
                with h_c4:
                    if st.button("🔄 Reset", use_container_width=True):
                        sim["obs"], _ = sim["env"].reset()
                        sim["step"] = 0
                        sim["cum_reward"] = 0.0
                        sim["done"] = False
                        st.rerun()

            elif selected_env == "Pendulum-v1":
                torque_slider = st.slider("Torque Command", -2.0, 2.0, 0.0, step=0.2)
                p_c1, p_c2 = st.columns(2)
                with p_c1:
                    if st.button("⚡ Apply Torque", use_container_width=True):
                        h_action = np.array([torque_slider], dtype=np.float32)
                with p_c2:
                    if st.button("🔄 Reset", use_container_width=True):
                        sim["obs"], _ = sim["env"].reset()
                        sim["step"] = 0
                        sim["cum_reward"] = 0.0
                        sim["done"] = False
                        st.rerun()

            else:
                st.info("Interactive control for this high-dimensional environment is best explored via automated rollouts.")

            if h_action is not None and not sim["done"]:
                raw_act, step_act, insp = step_policy_with_inspection(
                    selected_env, selected_model_info, sim["obs"], sim["env"],
                    human_action=h_action
                )
                next_obs, r, term, trunc, _ = sim["env"].step(step_act)
                sim["obs"] = next_obs
                sim["step"] += 1
                sim["last_reward"] = float(r)
                sim["cum_reward"] += float(r)
                sim["last_action"] = raw_act
                sim["last_inspection"] = insp
                sim["done"] = term or trunc
                st.rerun()

    # --- Right Column: Telemetry & Deep Action Inspector ---
    with col_telemetry:
        st.markdown("##### 📈 Live Telemetry Panel")

        t1, t2 = st.columns(2)
        with t1:
            st.metric("Cumulative Return", f"{sim['cum_reward']:+.1f}")
        with t2:
            st.metric("Step / Horizon", f"{sim['step']} / {env_spec['max_episode_steps']}")

        # Progress bar
        progress_val = min(1.0, max(0.0, sim["step"] / env_spec["max_episode_steps"]))
        st.progress(progress_val)

        # Episode status indicator
        if sim["done"]:
            if sim["term"]:
                st.error("❌ Episode Terminated: Terminal Failure Condition Reached")
            elif sim["trunc"]:
                st.success("✅ Episode Completed: Horizon Time Limit Reached")
        else:
            st.info("🟢 Episode Active: Policy Stepping In-Progress")

        # Environment-Specific Telemetry Gauge
        st.markdown("##### 🧭 Observation State Readings")
        obs_array = np.array(sim["obs"])
        if selected_env == "CartPole-v1":
            x, x_dot, theta, theta_dot = obs_array
            theta_deg = np.degrees(theta)
            st.write(f"• **Cart Position**: `{x:+.3f}` m (Limit: ±2.4m)")
            st.write(f"• **Cart Velocity**: `{x_dot:+.3f}` m/s")
            st.write(f"• **Pole Angle**: `{theta_deg:+.2f}°` (Limit: ±12°)")
            st.write(f"• **Angular Velocity**: `{theta_dot:+.3f}` rad/s")

        elif selected_env == "MountainCar-v0":
            pos, vel = obs_array
            st.write(f"• **Car Position**: `{pos:+.3f}` (Goal: ≥ 0.50)")
            st.write(f"• **Car Velocity**: `{vel:+.4f}`")

        elif selected_env == "Pendulum-v1":
            cos_th, sin_th, th_dot = obs_array
            angle_deg = np.degrees(np.arctan2(sin_th, cos_th))
            st.write(f"• **Angle**: `{angle_deg:+.1f}°` (Upright: 0°)")
            st.write(f"• **Angular Velocity**: `{th_dot:+.2f}` rad/s")

        elif selected_env == "LunarLander-v3":
            x, y, vx, vy, ang, vang, lc, rc = obs_array
            st.write(f"• **Coordinates**: x=`{x:+.2f}`, y=`{y:+.2f}`")
            st.write(f"• **Velocity**: vx=`{vx:+.2f}`, vy=`{vy:+.2f}`")
            st.write(f"• **Tilt**: `{np.degrees(ang):+.1f}°`")
            st.write(f"• **Leg Contacts**: Left=`{bool(lc)}`, Right=`{bool(rc)}`")

        elif selected_env == "BipedalWalker-v3":
            hull_angle = obs_array[0]
            hull_vel = obs_array[2]
            st.write(f"• **Hull Angle**: `{np.degrees(hull_angle):+.1f}°`")
            st.write(f"• **Forward Speed**: `{hull_vel:+.2f}`")

        # --- Deep Action Inspector Panel ---
        st.markdown("##### 🔍 Deep Action & Quantum Activation Inspector")
        insp = sim["last_inspection"]

        if insp is not None:
            # 1. Quantum Expectation Values
            if insp.get("quantum_expectations") is not None:
                q_vals = insp["quantum_expectations"]
                st.markdown("**⚛ 4-Qubit Pauli-Z Expectation Values** $\\langle Z_i \\rangle \\in [-1, 1]$:")

                fig_q = go.Figure(go.Bar(
                    x=[f"Qubit {i}" for i in range(4)],
                    y=q_vals,
                    marker=dict(color=["#c084fc", "#a855f7", "#9333ea", "#7e22ce"]),
                    text=[f"{v:+.3f}" for v in q_vals],
                    textposition="auto",
                ))
                fig_q.update_layout(
                    height=180,
                    margin=dict(l=10, r=10, t=10, b=20),
                    yaxis=dict(range=[-1.1, 1.1], title="⟨Z⟩"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1", size=11),
                )
                st.plotly_chart(fig_q, use_container_width=True)

            # 2. Discrete Q-values
            if insp.get("q_values") is not None:
                q_vals = insp["q_values"]
                st.markdown("**Discrete Action Q-Values** $Q(s, a)$:")
                fig_act = go.Figure(go.Bar(
                    x=[f"Action {i}" for i in range(len(q_vals))],
                    y=q_vals,
                    marker=dict(color="#38bdf8"),
                    text=[f"{v:+.2f}" for v in q_vals],
                    textposition="auto",
                ))
                fig_act.update_layout(
                    height=160,
                    margin=dict(l=10, r=10, t=10, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1", size=11),
                )
                st.plotly_chart(fig_act, use_container_width=True)

            # 3. Action Probabilities
            elif insp.get("action_distribution") is not None and "probs" in insp["action_distribution"]:
                probs = insp["action_distribution"]["probs"]
                st.markdown("**Action Probability Distribution** $\\pi(a|s)$:")
                fig_act = go.Figure(go.Bar(
                    x=[f"Action {i}" for i in range(len(probs))],
                    y=probs,
                    marker=dict(color="#34d399"),
                    text=[f"{p*100:.1f}%" for p in probs],
                    textposition="auto",
                ))
                fig_act.update_layout(
                    height=160,
                    margin=dict(l=10, r=10, t=10, b=20),
                    yaxis=dict(range=[0, 1.05], title="Probability"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cbd5e1", size=11),
                )
                st.plotly_chart(fig_act, use_container_width=True)

            # 4. Continuous Action Gaussian Mean / Std
            elif insp.get("action_distribution") is not None and "mean" in insp["action_distribution"]:
                dist_data = insp["action_distribution"]
                st.markdown("**Continuous Policy Parameters**:")
                st.write(f"• **Mean Vector** $\\mu(s)$: `{np.array(dist_data['mean']).round(3)}`")
                st.write(f"• **Std Vector** $\\sigma(s)$: `{np.array(dist_data['std']).round(3)}`")
        else:
            st.caption("Step the policy to inspect internal network activations and quantum observables.")

