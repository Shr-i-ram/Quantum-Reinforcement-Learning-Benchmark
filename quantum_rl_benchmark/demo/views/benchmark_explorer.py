"""
Benchmark Explorer and Results Dashboard:
Interactive exploration of all recorded benchmark runs, figures, and data.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from quantum_rl_benchmark.results.experiment_registry import load_experiment_registry
from quantum_rl_benchmark.config.settings import ENVIRONMENTS


def render_benchmark_explorer():
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <span class="badge-pill badge-cyan">Manifest-Backed Data</span>
        <span class="badge-pill badge-quantum">Multi-Metric Explorer</span>
    </div>
    <h2 style="font-weight: 700; margin-bottom: 0.25rem;">📊 Benchmark Results Explorer</h2>
    <p style="color: #94a3b8; font-size: 0.95rem;">
        Filter, compare, and analyze all 20 primary benchmark evaluations directly from the verified experiment registry.
    </p>
    """, unsafe_allow_html=True)

    registry = load_experiment_registry()
    bench_data = registry.get("benchmark_experiments", {})

    # Convert to DataFrame
    rows = []
    for k, v in bench_data.items():
        m = v.get("metrics", {})
        q = v.get("quantum_resources", {})
        rows.append({
            "Experiment": k,
            "Environment": v["environment"],
            "Algorithm": v["algorithm"],
            "Paradigm": "Hybrid Quantum" if v["is_quantum"] else "Classical",
            "Regime": v["configuration_regime"].capitalize(),
            "Total Steps": v["training_budget_steps"],
            "Mean Return": m.get("mean_return"),
            "Std Return": m.get("std_return"),
            "Median Return": m.get("median_return"),
            "Best Return": m.get("max_return"),
            "Wall-Clock Time (s)": round(v.get("wall_clock_time_s", 0), 1),
            "Parameters": v.get("n_parameters", 0),
            "Quantum Qubits": q.get("n_qubits", 0),
            "Status": v["completion_status"],
        })

    df = pd.DataFrame(rows)

    # Filter Bar
    f1, f2, f3 = st.columns(3)
    with f1:
        envs_filter = st.multiselect("Filter Environments", options=df["Environment"].unique(), default=list(df["Environment"].unique()))
    with f2:
        paradigms_filter = st.multiselect("Filter Paradigm", options=df["Paradigm"].unique(), default=list(df["Paradigm"].unique()))
    with f3:
        regime_filter = st.multiselect("Filter Regime", options=df["Regime"].unique(), default=list(df["Regime"].unique()))

    filtered_df = df[
        (df["Environment"].isin(envs_filter)) &
        (df["Paradigm"].isin(paradigms_filter)) &
        (df["Regime"].isin(regime_filter))
    ]

    st.markdown("---")

    # Interactive Plots
    p_tab1, p_tab2, p_tab3 = st.tabs(["📈 Returns Comparison", "⚡ Runtime vs Performance Pareto", "🌡️ Cross-Environment Heatmap"])

    with p_tab1:
        fig_bar = px.bar(
            filtered_df,
            x="Environment",
            y="Mean Return",
            color="Algorithm",
            barmode="group",
            text="Mean Return",
            title="Deterministic Evaluation Mean Returns by Algorithm and Environment",
            hover_data=["Median Return", "Std Return", "Wall-Clock Time (s)", "Parameters"],
            color_discrete_map={
                "PPO": "#38bdf8",
                "Q-PPO": "#c084fc",
                "DQN": "#34d399",
                "Q-DQN": "#f472b6",
                "SAC": "#fbbf24",
                "Q-SAC": "#a855f7",
            }
        )
        fig_bar.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_bar.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with p_tab2:
        fig_pareto = px.scatter(
            filtered_df,
            x="Wall-Clock Time (s)",
            y="Mean Return",
            color="Algorithm",
            size=[max(10, p / 300) for p in filtered_df["Parameters"]],
            hover_name="Experiment",
            log_x=True,
            title="Computational Runtime (Log Scale) vs. Evaluation Performance",
            color_discrete_map={
                "PPO": "#38bdf8",
                "Q-PPO": "#c084fc",
                "DQN": "#34d399",
                "Q-DQN": "#f472b6",
                "SAC": "#fbbf24",
                "Q-SAC": "#a855f7",
            }
        )
        fig_pareto.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    with p_tab3:
        pivot_returns = df.pivot(index="Algorithm", columns="Environment", values="Mean Return")
        fig_heat = px.imshow(
            pivot_returns,
            labels=dict(x="Gymnasium Environment", y="Algorithm", color="Mean Return"),
            color_continuous_scale="Viridis",
            title="Performance Return Matrix across All 20 Experiments"
        )
        fig_heat.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Master Results Table
    st.markdown("### 📋 Complete Benchmark Experiments Table")
    st.dataframe(
        filtered_df.sort_values(by=["Environment", "Mean Return"], ascending=[True, False]),
        use_container_width=True,
        hide_index=True
    )
