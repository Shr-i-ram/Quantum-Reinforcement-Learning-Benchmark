"""
Quantum Reinforcement Learning Benchmark: Research Platform & Live Demonstration.
Top-level Streamlit Application.

Usage:
    streamlit run app.py
    # Or directly:
    python app.py
"""

import os
import sys

# Auto-dispatch to `streamlit run` if invoked directly via `python app.py`
try:
    from streamlit.runtime import exists as _st_exists
    _running_under_streamlit = _st_exists()
except Exception:
    _running_under_streamlit = False

if not _running_under_streamlit:
    import subprocess
    cmd = [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))

import platform
import streamlit as st
import torch
import pennylane as qml

# Configure Streamlit page layout
st.set_page_config(
    page_title="Quantum RL Benchmark | Research Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_rl_benchmark.demo.styles import apply_custom_styles
from quantum_rl_benchmark.demo.views.mission_control import render_mission_control
from quantum_rl_benchmark.demo.views.playground import render_playground
from quantum_rl_benchmark.demo.views.head_to_head import render_head_to_head
from quantum_rl_benchmark.demo.views.training_studio import render_training_studio
from quantum_rl_benchmark.demo.views.benchmark_explorer import render_benchmark_explorer
from quantum_rl_benchmark.demo.views.quantum_lab import render_quantum_lab
from quantum_rl_benchmark.demo.views.ablations_view import render_ablations_view
from quantum_rl_benchmark.demo.views.research_insights import render_research_insights
from quantum_rl_benchmark.demo.views.guided_tour import render_guided_tour


def main():
    apply_custom_styles()

    # --- Sidebar Navigation ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <h2 style="font-weight: 800; font-size: 1.5rem; margin-bottom: 0.2rem; color: #38bdf8;">
                ⚛️ Quantum RL
            </h2>
            <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;">
                Benchmark & Live Demo
            </div>
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "🚀 Mission Control": render_mission_control,
            "🎮 Environment Playground": render_playground,
            "⚔️ Head-to-Head Arena": render_head_to_head,
            "⚡ Live Training Studio": render_training_studio,
            "📊 Benchmark Explorer": render_benchmark_explorer,
            "🔬 Quantum Architecture Lab": render_quantum_lab,
            "🧩 Ablations & Robustness": render_ablations_view,
            "💡 Research Insights": render_research_insights,
            "🎓 Guided Presentation Tour": render_guided_tour,
        }

        selected_page = st.radio(
            "Navigation",
            list(pages.keys()),
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # System Diagnostics in Sidebar
        st.markdown("##### 💻 System Environment")
        cuda_ok = torch.cuda.is_available()
        st.caption(f"• **OS**: {platform.system()} {platform.release()}")
        st.caption(f"• **Python**: {platform.python_version()}")
        st.caption(f"• **PyTorch**: {torch.__version__} ({'CUDA' if cuda_ok else 'CPU'})")
        st.caption(f"• **PennyLane**: {qml.__version__} (lightning.qubit)")

        st.markdown("---")
        # Safe Demo Recovery Button
        if st.button("🔄 Reset Demo State", use_container_width=True, help="Clears cached session data if needed"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # --- Render Selected Page ---
    try:
        pages[selected_page]()
    except Exception as e:
        st.error(f"Application error on {selected_page}: {e}")
        st.info("You can use the 'Reset Demo State' button in the sidebar to restore the default state.")


if __name__ == "__main__":
    main()

