import React, { useState } from 'react';
import { X, ChevronLeft, ChevronRight, Award, Zap, Cpu, Play, CheckCircle } from 'lucide-react';

interface GuidedTourModalProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigateTab: (tabId: string) => void;
}

const TOUR_STEPS = [
  {
    step: 1,
    title: "Project Mission & Research Question",
    tabTarget: "mission",
    badge: "Introduction",
    content: (
      <div className="space-y-3 text-slate-300 text-sm">
        <p>
          Welcome to the <strong>Quantum Reinforcement Learning Arena</strong> — an interactive research platform
          evaluating the empirical performance of <strong>hybrid quantum-classical RL algorithms</strong> against classical baselines.
        </p>
        <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-700/60">
          <p className="text-cyan-400 font-medium">Core Research Question:</p>
          <p className="italic text-slate-200 mt-1">
            "Does replacing classical neural network layers with variational quantum circuits (VQCs) provide an empirical advantage
            in sample efficiency, policy stability, or final return across Gymnasium control environments?"
          </p>
        </div>
        <p>
          This platform was built to demonstrate our experiments live, with working physics, real neural networks,
          PennyLane quantum circuits, and verified empirical metrics.
        </p>
      </div>
    ),
  },
  {
    step: 2,
    title: "Classical Baseline Mastery: CartPole-v1 Solved",
    tabTarget: "playground",
    badge: "Demonstration Agent",
    content: (
      <div className="space-y-3 text-slate-300 text-sm">
        <p>
          To establish a rigorous benchmark standard, we trained an optimized classical Deep Q-Network (DQN) agent
          specifically tuned on CartPole-v1.
        </p>
        <div className="grid grid-cols-2 gap-2 text-center">
          <div className="bg-emerald-950/40 border border-emerald-500/40 p-2.5 rounded-lg">
            <div className="text-emerald-400 text-xs font-semibold uppercase tracking-wider">Mean Return</div>
            <div className="text-2xl font-bold text-emerald-300 mt-0.5">480.2 <span className="text-xs text-slate-400">/ 500</span></div>
            <div className="text-[11px] text-emerald-400/80">Near-Optimal Mastery</div>
          </div>
          <div className="bg-emerald-950/40 border border-emerald-500/40 p-2.5 rounded-lg">
            <div className="text-emerald-400 text-xs font-semibold uppercase tracking-wider">Median Return</div>
            <div className="text-2xl font-bold text-emerald-300 mt-0.5">500.0 <span className="text-xs text-slate-400">max</span></div>
            <div className="text-[11px] text-emerald-400/80">Solves Environment</div>
          </div>
        </div>
        <p className="text-xs text-slate-400">
          You can test this agent live in the <strong>Interactive Playground</strong> tab to watch it hold the pole vertically upright indefinitely.
        </p>
      </div>
    ),
  },
  {
    step: 3,
    title: "The Hybrid Quantum Architecture",
    tabTarget: "quantum",
    badge: "Ansatz Design",
    content: (
      <div className="space-y-3 text-slate-300 text-sm">
        <p>
          Our hybrid architecture replaces inner feedforward layers with a <strong>4-qubit Variational Quantum Circuit (VQC)</strong>:
        </p>
        <ul className="list-disc pl-5 space-y-1 text-xs text-slate-300">
          <li><strong>Classical Preprocessing:</strong> Linear projection with <code className="text-purple-300">Tanh</code> bounding features to <code className="text-purple-300">[-1, 1]⁴</code>.</li>
          <li><strong>Angle Encoding:</strong> State vector encoded via parameterized rotations <code className="text-cyan-300">RY(xᵢ)</code> on each wire.</li>
          <li><strong>Variational Entanglement:</strong> Parameterized <code className="text-cyan-300">RY(θ)</code>, <code className="text-cyan-300">RZ(ϕ)</code> rotations with circular CNOT entangling ladders.</li>
          <li><strong>Expectation Value Readout:</strong> Instantaneous Pauli-Z expectation values <code className="text-cyan-300">⟨Zᵢ⟩ ∈ [-1, 1]</code> for each qubit.</li>
        </ul>
        <p className="text-xs text-slate-400">
          In the <strong>Quantum Lab</strong>, you can manipulate feature angle sliders and watch PennyLane compute expectation values in real time.
        </p>
      </div>
    ),
  },
  {
    step: 4,
    title: "Head-to-Head Duel: Classical vs Quantum Reality",
    tabTarget: "duel",
    badge: "Empirical Comparison",
    content: (
      <div className="space-y-3 text-slate-300 text-sm">
        <p>
          In the <strong>Head-to-Head Arena</strong>, you can pit Classical and Quantum models against each other
          on identical environment seeds and compare their trajectories.
        </p>
        <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-700/60 text-xs space-y-1">
          <div className="flex justify-between items-center text-emerald-400 font-semibold">
            <span>Classical DQN (Demonstration):</span>
            <span>480.2 ± 40.2 Return</span>
          </div>
          <div className="flex justify-between items-center text-amber-400 font-semibold">
            <span>Classical DQN (Original Benchmark):</span>
            <span>154.6 ± 2.7 Return</span>
          </div>
          <div className="flex justify-between items-center text-rose-400 font-semibold">
            <span>Hybrid Q-DQN (4 Qubits):</span>
            <span>9.4 ± 0.7 Return</span>
          </div>
        </div>
        <p className="text-xs text-slate-400">
          The statistical batch duel runner allows you to execute 15-30 synchronous evaluation trials live to verify this variance distribution.
        </p>
      </div>
    ),
  },
  {
    step: 5,
    title: "Human Challenge Arena: Beat the AI",
    tabTarget: "challenges",
    badge: "Interactive Play",
    content: (
      <div className="space-y-3 text-slate-300 text-sm">
        <p>
          Can a human operator outperform our reinforcement learning agents?
        </p>
        <p className="text-xs text-slate-300">
          The <strong>Human Challenge Arena</strong> lets professors and reviewers take direct control of the Gymnasium environments
          using keyboard arrows or on-screen buttons.
        </p>
        <div className="bg-cyan-950/30 border border-cyan-500/30 p-2.5 rounded-lg text-xs">
          <span className="text-cyan-400 font-semibold">Challenges Available:</span>
          <p className="text-slate-300 mt-1">
            • CartPole: Maintain balance for &gt; 300 steps.<br />
            • MountainCar: Reach the mountain summit in &lt; 160 steps.<br />
            • LunarLander: Land between flags with score &gt; 100.
          </p>
        </div>
        <p className="text-xs text-slate-400">
          Scores are recorded to a live, persistent leaderboard for the presentation session!
        </p>
      </div>
    ),
  },
  {
    step: 6,
    title: "Computational Bottleneck & Simulation Scaling",
    tabTarget: "quantum",
    badge: "Complexity Analysis",
    content: (
      <div className="space-y-3 text-slate-300 text-sm">
        <p>
          A central finding of our research is the <strong>simulation bottleneck</strong>:
        </p>
        <ul className="list-disc pl-5 space-y-1 text-xs text-slate-300">
          <li>Simulating a 4-qubit circuit with statevectors on CPU/GPU is fast (~0.05ms) but requires <strong>2.5x to 6.2x more wall-clock time</strong> than standard matrix multiplications.</li>
          <li>On physical NISQ quantum hardware, shot-based sampling (<code className="text-cyan-300">S ≥ 1000</code>) and cloud API queue latencies make RL gradient updates orders of magnitude slower.</li>
          <li><strong>Barren Plateaus:</strong> As qubit count increases, gradient variance vanishes exponentially: <code className="text-rose-400">Var[∂L/∂θ] ∝ 1/2ⁿ</code>.</li>
        </ul>
      </div>
    ),
  },
  {
    step: 7,
    title: "Live Training Studio: Transparent Training",
    tabTarget: "training",
    badge: "Real Background Training",
    content: (
      <div className="space-y-3 text-slate-300 text-sm">
        <p>
          Unlike static dashboards with fabricated curves, the <strong>Live Training Studio</strong> initiates a real background training thread
          on your machine.
        </p>
        <p className="text-xs text-slate-300">
          You can launch a 1,000 to 5,000 step run of DQN, Q-DQN, or PPO and monitor real-time policy updates, episode returns, and FPS throughput.
        </p>
        <p className="text-xs text-slate-400">
          Once finished or stopped, the resulting model checkpoint can be immediately loaded into the Playground for inspection!
        </p>
      </div>
    ),
  },
  {
    step: 8,
    title: "Conclusion & Scientific Defense of the Negative Result",
    tabTarget: "insights",
    badge: "Research Defense",
    content: (
      <div className="space-y-3 text-slate-300 text-sm">
        <p>
          <strong>Why this project is scientifically valuable:</strong>
        </p>
        <div className="space-y-2 text-xs">
          <div className="bg-slate-900/80 p-2 rounded border border-slate-700">
            <span className="text-purple-400 font-semibold">1. Scientific Integrity:</span> We transparently document the performance gap rather than cherry-picking artificial successes.
          </div>
          <div className="bg-slate-900/80 p-2 rounded border border-slate-700">
            <span className="text-purple-400 font-semibold">2. Architectural Insights:</span> Demonstrates that naive angle-encoding and 4-qubit expressivity cannot replace classical representational capacity in continuous state spaces.
          </div>
          <div className="bg-slate-900/80 p-2 rounded border border-slate-700">
            <span className="text-purple-400 font-semibold">3. Comprehensive Benchmarking:</span> 20 rigorous experimental configurations, ablations on circuit depth, noise robustness tests, and publication-ready IEEE LaTeX documentation.
          </div>
        </div>
      </div>
    ),
  },
];

export const GuidedTourModal: React.FC<GuidedTourModalProps> = ({
  isOpen,
  onClose,
  onNavigateTab,
}) => {
  const [currentIdx, setCurrentIdx] = useState(0);

  if (!isOpen) return null;

  const currentStep = TOUR_STEPS[currentIdx];

  const handleNext = () => {
    if (currentIdx < TOUR_STEPS.length - 1) {
      const nextIdx = currentIdx + 1;
      setCurrentIdx(nextIdx);
      onNavigateTab(TOUR_STEPS[nextIdx].tabTarget);
    }
  };

  const handlePrev = () => {
    if (currentIdx > 0) {
      const prevIdx = currentIdx - 1;
      setCurrentIdx(prevIdx);
      onNavigateTab(TOUR_STEPS[prevIdx].tabTarget);
    }
  };

  const handleJumpToTab = () => {
    onNavigateTab(currentStep.tabTarget);
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
      <div className="relative w-full max-w-2xl bg-slate-950 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Top Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/60">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold text-sm border border-cyan-500/30">
              {currentStep.step}
            </div>
            <div>
              <div className="text-xs uppercase tracking-wider text-cyan-400 font-semibold">
                Guided Review Presentation • Step {currentIdx + 1} of {TOUR_STEPS.length}
              </div>
              <h3 className="text-base font-semibold text-white mt-0.5">{currentStep.title}</h3>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="px-6 py-5 overflow-y-auto max-h-[60vh]">
          {currentStep.content}
        </div>

        {/* Bottom Navigation */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-slate-800 bg-slate-900/40">
          <div className="flex items-center gap-1.5">
            {TOUR_STEPS.map((_, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setCurrentIdx(idx);
                  onNavigateTab(TOUR_STEPS[idx].tabTarget);
                }}
                className={`h-2 rounded-full transition-all ${
                  idx === currentIdx
                    ? 'w-6 bg-cyan-400'
                    : 'w-2 bg-slate-700 hover:bg-slate-600'
                }`}
              />
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleJumpToTab}
              className="px-3 py-1.5 text-xs text-slate-300 hover:text-white bg-slate-800 hover:bg-slate-700 rounded-lg transition border border-slate-700"
            >
              Explore This Tab Directly
            </button>

            <button
              onClick={handlePrev}
              disabled={currentIdx === 0}
              className="p-1.5 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:hover:bg-transparent transition"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>

            {currentIdx < TOUR_STEPS.length - 1 ? (
              <button
                onClick={handleNext}
                className="flex items-center gap-1 px-4 py-1.5 text-xs font-semibold text-slate-950 bg-cyan-400 hover:bg-cyan-300 rounded-lg transition"
              >
                Next Step
                <ChevronRight className="w-4 h-4" />
              </button>
            ) : (
              <button
                onClick={onClose}
                className="flex items-center gap-1 px-4 py-1.5 text-xs font-semibold text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg transition"
              >
                Complete Tour
                <CheckCircle className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
