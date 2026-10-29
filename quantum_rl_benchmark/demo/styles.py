"""
Dark Modern Research Laboratory CSS Theme for Streamlit Live Demo.
"""

CUSTOM_CSS = """
<style>
/* Global Dark Background & Modern Font */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

code, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 15% 15%, #0f172a 0%, #020617 100%);
}

/* Custom Card Container */
.research-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
    transition: border-color 0.2s ease;
}

.research-card:hover {
    border-color: rgba(56, 189, 248, 0.45);
}

.quantum-card {
    background: rgba(30, 27, 75, 0.55);
    border: 1px solid rgba(168, 85, 247, 0.35);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px 0 rgba(112, 26, 117, 0.2);
}

.quantum-card:hover {
    border-color: rgba(192, 132, 252, 0.6);
}

/* Header & Accent Text */
.accent-cyan {
    color: #38bdf8;
    font-weight: 600;
}

.accent-purple {
    color: #c084fc;
    font-weight: 600;
}

.accent-emerald {
    color: #34d399;
    font-weight: 600;
}

.accent-amber {
    color: #fbbf24;
    font-weight: 600;
}

/* Metric Pill Badges */
.badge-pill {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 0.4rem;
}

.badge-classical {
    background: rgba(14, 165, 233, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.4);
}

.badge-quantum {
    background: rgba(168, 85, 247, 0.15);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.4);
}

.badge-demonstration {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-baseline {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

/* Metric Display Values */
.metric-value-huge {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}

.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 0.25rem;
}

/* Telemetry readout bar */
.telemetry-bar {
    background: #020617;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    border: 1px solid #1e293b;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

/* Custom Scrollbars */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #090d16;
}
::-webkit-scrollbar-thumb {
    background: #1e293b;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #334155;
}
</style>
"""


def apply_custom_styles():
    """Apply CSS styles into Streamlit app."""
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

