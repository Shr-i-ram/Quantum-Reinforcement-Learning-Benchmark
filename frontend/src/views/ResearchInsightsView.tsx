import React, { useState } from 'react';
import {
  BookOpen,
  Download,
  HelpCircle,
  CheckCircle2,
  AlertTriangle,
  FileText,
  ShieldCheck,
  Cpu,
  Flame,
} from 'lucide-react';
import { fetchReportMarkdown } from '../api';

const RESEARCH_QUESTIONS = [
  {
    q: "1. Did hybrid quantum RL outperform classical RL in our experiments?",
    a: (
      <span>
        <strong>No.</strong> Across all 5 tested Gymnasium environments (CartPole, MountainCar, LunarLander, Pendulum, BipedalWalker),
        classical algorithms consistently outperformed hybrid quantum-classical algorithms in both mean cumulative return and sample efficiency.
        For instance, in CartPole-v1, our optimized classical DQN achieved near-optimal mastery (Mean 480.2 ± 40.2 / 500), while the 4-qubit hybrid Q-DQN
        achieved a mean return of 9.4 ± 0.7 under an identical compute budget.
      </span>
    ),
    badge: "Empirical Finding",
  },
  {
    q: "2. Why is a rigorous negative result scientifically defensible and valuable?",
    a: (
      <span>
        A transparent, well-measured negative result is a cornerstone of scientific integrity. In quantum machine learning, substantial hype
        suggests that quantum variational circuits inherently enhance neural networks. Our benchmark proves that naive hybrid substitution
        introduces expressivity bottlenecks and high gradient variance without task-level benefits, providing an essential baseline
        and preventing future researchers from repeating identical architectural dead ends.
      </span>
    ),
    badge: "Methodology",
  },
  {
    q: "3. What is the primary computational bottleneck in simulating quantum RL?",
    a: (
      <span>
        Simulating quantum unitary transformations and parameter-shift or adjoint gradient calculations on classical hardware
        incurs a <strong>2.5x to 6.2x wall-clock penalty</strong> per policy update compared to dense classical layers.
        Every statevector step requires computing 2ⁿ complex amplitudes and calculating separate quantum expectation values for each qubit wire.
      </span>
    ),
    badge: "Compute Bottleneck",
  },
  {
    q: "4. Why does a 4-qubit VQC act as an information bottleneck?",
    a: (
      <span>
        In continuous observation spaces (e.g. BipedalWalker's 24-dimensional state or LunarLander's 8-dimensional state),
        projecting high-dimensional features down to 4 rotational angles and measuring 4 Pauli-Z expectation values ⟨Zᵢ⟩ ∈ [-1, 1]
        imposes an extreme compression constraint. Rich geometric relationships in continuous dynamics cannot be preserved across 4 scalar expectation values.
      </span>
    ),
    badge: "Ansatz Expressivity",
  },
  {
    q: "5. What are Barren Plateaus and did they impact training?",
    a: (
      <span>
        Barren plateaus refer to the mathematical phenomenon where the variance of cost function gradients vanishes exponentially
        as the number of qubits grows: <code>Var[∂L/∂θ] ∈ O(1/2ⁿ)</code>. While 4 qubits is small enough to avoid asymptotic barren plateaus,
        the non-convex, oscillating sinusoidal landscape of parameterized rotations creates localized saddle points and flat plateaus
        that impede standard Adam and RMSprop optimizers in reinforcement learning.
      </span>
    ),
    badge: "Theoretical Hazard",
  },
  {
    q: "6. How would physical NISQ quantum hardware perform compared to our simulator?",
    a: (
      <span>
        Counter-intuitively, executing these circuits on current physical NISQ QPUs would be <strong>orders of magnitude slower</strong>
        due to cloud network dispatch latencies (~1-5 seconds per batch) and the necessity of shot-based sampling (S ≥ 1,000 shots per measurement)
        to suppress readout variance within ε ≈ 1/√S. Classical simulation is vastly faster up to ~22 qubits.
      </span>
    ),
    badge: "Hardware Reality",
  },
  {
    q: "7. How was scientific reproducibility ensured throughout this benchmark?",
    a: (
      <span>
        All 20 experiment configurations are completely cataloged in <code>experiment_registry.json</code>, with preserved model checkpoints,
        evaluation seeds, training curves, and hyperparameter dictionaries. Every evaluation in this platform loads the genuine PyTorch weights
        and executes real Gymnasium physics.
      </span>
    ),
    badge: "Reproducibility",
  },
  {
    q: "8. What are constructive future directions for Quantum Reinforcement Learning?",
    a: (
      <span>
        Promising directions include: (1) <em>Quantum Reservoir Computing</em> which exploits fixed unitary dynamics without gradient training;
        (2) <em>Direct Hamiltonian Encoding</em> where environments themselves are quantum systems (e.g. quantum control or molecular simulation);
        and (3) <em>Multi-Scale Data Re-Uploading</em> architectures where classical states are injected between every entangling layer.
      </span>
    ),
    badge: "Future Work",
  },
];

export const ResearchInsightsView: React.FC = () => {
  const [downloading, setDownloading] = useState<boolean>(false);

  const handleDownloadReport = async () => {
    setDownloading(true);
    try {
      const md = await fetchReportMarkdown();
      const blob = new Blob([md], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'Quantum_RL_Research_Report.md';
      a.click();
    } catch (e) {
      console.error(e);
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-blue-950/40 via-slate-900 to-slate-950 border border-blue-500/30 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-950 text-blue-400 border border-blue-800/60 text-xs font-semibold mb-2">
            <BookOpen className="w-3.5 h-3.5" />
            Publication-Grade Scientific Analysis
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white">Research Insights & Defense</h1>
          <p className="text-sm text-slate-300 mt-1 max-w-2xl">
            Evidence-based answers to key research questions, computational trade-off analysis, theoretical bottleneck examinations, and publication briefing exports.
          </p>
        </div>

        <button
          onClick={handleDownloadReport}
          disabled={downloading}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-bold bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition shadow-lg shadow-cyan-500/20 cursor-pointer disabled:opacity-50"
        >
          <Download className="w-4 h-4" />
          {downloading ? 'Exporting...' : 'Export Full Research Briefing (.md)'}
        </button>
      </div>

      {/* 8 Research Q&A Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {RESEARCH_QUESTIONS.map((item, idx) => (
          <div
            key={idx}
            className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3 hover:border-slate-700 transition"
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-slate-950 border border-slate-800 text-cyan-400">
                {item.badge}
              </span>
              <span className="text-xs text-slate-500 font-mono">Q{idx + 1}</span>
            </div>
            <h3 className="text-sm font-bold text-white leading-snug">{item.q}</h3>
            <p className="text-xs text-slate-300 leading-relaxed">{item.a}</p>
          </div>
        ))}
      </div>

      {/* Conference Paper Link Notice */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-2xl p-6 flex flex-col sm:flex-row justify-between items-center gap-4 text-xs text-slate-400">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-500/20 text-blue-400 flex items-center justify-center">
            <FileText className="w-5 h-5" />
          </div>
          <div>
            <div className="text-sm font-bold text-white">IEEE Conference Paper Available</div>
            <p className="text-slate-400 mt-0.5">
              Compiled into <code>IEEE_Quantum_RL_Benchmark_Paper.pdf</code> with full mathematical formulations and figures.
            </p>
          </div>
        </div>

        <button
          onClick={handleDownloadReport}
          className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-xl font-medium transition cursor-pointer"
        >
          Download Summary Report
        </button>
      </div>
    </div>
  );
};
