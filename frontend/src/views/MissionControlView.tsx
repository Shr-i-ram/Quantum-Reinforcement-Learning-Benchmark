import React from 'react';
import {
  Gamepad2,
  Swords,
  Cpu,
  Flame,
  BarChart3,
  Trophy,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  ShieldCheck,
  Binary,
  Layers,
  Sparkles,
} from 'lucide-react';

interface MissionControlViewProps {
  onNavigateTab: (tabId: string) => void;
  onOpenTour: () => void;
}

export const MissionControlView: React.FC<MissionControlViewProps> = ({
  onNavigateTab,
  onOpenTour,
}) => {
  return (
    <div className="space-y-8 pb-12 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-b from-slate-900 via-slate-900/80 to-slate-950 border border-slate-800 p-8 sm:p-12 shadow-2xl">
        <div className="absolute top-0 right-0 -mr-20 -mt-20 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -ml-20 -mb-20 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-950/60 border border-cyan-800/60 text-cyan-400 text-xs font-semibold mb-4">
            <Sparkles className="w-3.5 h-3.5" />
            University Research Review & Interactive Demonstration Platform
          </div>

          <h1 className="text-3xl sm:text-5xl font-extrabold text-white tracking-tight leading-tight">
            Comparison of <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">Quantum Reinforcement Learning</span> in Gymnasium Environments
          </h1>

          <p className="mt-4 text-base sm:text-lg text-slate-300 leading-relaxed">
            An empirical investigation into hybrid quantum-classical policy optimization across increasing environmental complexity.
            Featuring live simulation, deep Pauli-Z quantum introspection, real model stepping, and transparent benchmark analysis.
          </p>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <button
              onClick={() => onNavigateTab('playground')}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition shadow-lg shadow-cyan-500/25 cursor-pointer"
            >
              <Gamepad2 className="w-4 h-4" />
              Launch Interactive Playground
              <ArrowRight className="w-4 h-4" />
            </button>

            <button
              onClick={onOpenTour}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 transition cursor-pointer"
            >
              Start Guided Presentation Tour
            </button>
          </div>
        </div>
      </div>

      {/* Primary Research Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium uppercase tracking-wider">
            <span>Environments Tested</span>
            <Layers className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-3xl font-extrabold text-white mt-2">5 Suites</div>
          <p className="text-xs text-slate-400 mt-1">CartPole, MountainCar, LunarLander, Pendulum, BipedalWalker</p>
        </div>

        <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium uppercase tracking-wider">
            <span>Verified Baseline</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-extrabold text-emerald-400 mt-2">480.2 <span className="text-sm font-normal text-slate-400">/ 500</span></div>
          <p className="text-xs text-emerald-400/80 mt-1">DQN Demonstration Solved (Median: 500.0)</p>
        </div>

        <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium uppercase tracking-wider">
            <span>Quantum VQC Width</span>
            <Binary className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-3xl font-extrabold text-purple-400 mt-2">4 Qubits</div>
          <p className="text-xs text-slate-400 mt-1">Angle Embedding + Circular CNOT Entanglement</p>
        </div>

        <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-5 hover:border-slate-700 transition">
          <div className="flex items-center justify-between text-slate-400 text-xs font-medium uppercase tracking-wider">
            <span>Benchmark Experiments</span>
            <BarChart3 className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-3xl font-extrabold text-amber-400 mt-2">20 Configs</div>
          <p className="text-xs text-slate-400 mt-1">Preserved Checkpoints, Ablations, & Robustness</p>
        </div>
      </div>

      {/* Architectural Flow Diagram */}
      <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 sm:p-8">
        <h2 className="text-xl font-bold text-white mb-2 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-cyan-400" />
          Hybrid Quantum-Classical Dataflow Pipeline
        </h2>
        <p className="text-sm text-slate-400 mb-6">
          How observation vectors from Gymnasium environments are translated through variational quantum states to produce agent decisions:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-center text-center">
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <span className="text-xs text-slate-400 uppercase font-semibold">1. Observation</span>
            <div className="text-base font-bold text-white mt-1">s ∈ ℝᵈ</div>
            <div className="text-[11px] text-slate-400 mt-1">Classic or Box2D state (e.g. 4 dims)</div>
          </div>

          <div className="text-cyan-400 font-bold hidden md:block">→</div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <span className="text-xs text-purple-400 uppercase font-semibold">2. Feature Projection</span>
            <div className="text-base font-bold text-purple-300 mt-1">Tanh(W s + b)</div>
            <div className="text-[11px] text-slate-400 mt-1">Normalized angles x ∈ [-1, 1]⁴</div>
          </div>

          <div className="text-cyan-400 font-bold hidden md:block">→</div>

          <div className="bg-slate-950 p-4 rounded-xl border border-cyan-500/40 shadow-lg shadow-cyan-500/10">
            <span className="text-xs text-cyan-400 uppercase font-semibold">3. 4-Qubit Ansatz</span>
            <div className="text-base font-bold text-cyan-300 mt-1">RY(x) ⊗ U(θ) ⊗ CNOT</div>
            <div className="text-[11px] text-cyan-400/80 mt-1">PennyLane QNode Simulation</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 items-center text-center mt-3">
          <div className="hidden md:block col-span-2"></div>
          <div className="text-cyan-400 font-bold hidden md:block">↓</div>
          <div className="hidden md:block col-span-2"></div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 items-center text-center mt-2 max-w-2xl mx-auto">
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <span className="text-xs text-cyan-400 uppercase font-semibold">4. Quantum Readout</span>
            <div className="text-base font-bold text-white mt-1">⟨Zᵢ⟩ ∈ [-1, 1]</div>
            <div className="text-[11px] text-slate-400 mt-1">4 Pauli-Z Expectation Values</div>
          </div>

          <div className="text-cyan-400 font-bold hidden md:block">→</div>

          <div className="bg-slate-950 p-4 rounded-xl border border-emerald-500/40">
            <span className="text-xs text-emerald-400 uppercase font-semibold">5. Action Output</span>
            <div className="text-base font-bold text-emerald-300 mt-1">Q(s, a) or π(a|s)</div>
            <div className="text-[11px] text-emerald-400/80 mt-1">Action Selection & Environment Step</div>
          </div>
        </div>
      </div>

      {/* Interactive Laboratories Directory */}
      <div>
        <h2 className="text-xl font-bold text-white mb-4">Interactive Research Laboratories</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div
            onClick={() => onNavigateTab('playground')}
            className="group bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-cyan-500/50 p-5 rounded-2xl cursor-pointer transition shadow-lg"
          >
            <div className="w-10 h-10 rounded-xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center mb-3 group-hover:scale-110 transition">
              <Gamepad2 className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-white group-hover:text-cyan-400 transition">Interactive Playground</h3>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              Step through environments with real physics, toggle human or AI control, and inspect live Pauli-Z expectation values.
            </p>
          </div>

          <div
            onClick={() => onNavigateTab('challenges')}
            className="group bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-amber-500/50 p-5 rounded-2xl cursor-pointer transition shadow-lg"
          >
            <div className="w-10 h-10 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center mb-3 group-hover:scale-110 transition">
              <Trophy className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-white group-hover:text-amber-400 transition">Human Challenge Arena</h3>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              Directly control CartPole, MountainCar, or LunarLander via keyboard arrows to test if you can beat the AI scorecard.
            </p>
          </div>

          <div
            onClick={() => onNavigateTab('duel')}
            className="group bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-rose-500/50 p-5 rounded-2xl cursor-pointer transition shadow-lg"
          >
            <div className="w-10 h-10 rounded-xl bg-rose-500/20 text-rose-400 flex items-center justify-center mb-3 group-hover:scale-110 transition">
              <Swords className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-white group-hover:text-rose-400 transition">Head-to-Head Duel</h3>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              Run synchronized classical vs quantum duels across identical seeds to observe divergence in actions and return distributions.
            </p>
          </div>

          <div
            onClick={() => onNavigateTab('quantum')}
            className="group bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-purple-500/50 p-5 rounded-2xl cursor-pointer transition shadow-lg"
          >
            <div className="w-10 h-10 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center mb-3 group-hover:scale-110 transition">
              <Cpu className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-white group-hover:text-purple-400 transition">Quantum Lab</h3>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              Adjust feature angle sliders and execute live PennyLane QNodes to calculate Pauli-Z expectation values and state probabilities.
            </p>
          </div>

          <div
            onClick={() => onNavigateTab('training')}
            className="group bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-orange-500/50 p-5 rounded-2xl cursor-pointer transition shadow-lg"
          >
            <div className="w-10 h-10 rounded-xl bg-orange-500/20 text-orange-400 flex items-center justify-center mb-3 group-hover:scale-110 transition">
              <Flame className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-white group-hover:text-orange-400 transition">Live Training Studio</h3>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              Launch real background training sessions, monitor live policy losses, and immediately deploy newly trained models to the arena.
            </p>
          </div>

          <div
            onClick={() => onNavigateTab('benchmarks')}
            className="group bg-slate-900/60 hover:bg-slate-900 border border-slate-800 hover:border-emerald-500/50 p-5 rounded-2xl cursor-pointer transition shadow-lg"
          >
            <div className="w-10 h-10 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center mb-3 group-hover:scale-110 transition">
              <BarChart3 className="w-5 h-5" />
            </div>
            <h3 className="text-base font-semibold text-white group-hover:text-emerald-400 transition">Benchmark Explorer</h3>
            <p className="text-xs text-slate-400 mt-1.5 leading-relaxed">
              Explore the complete 20-configuration benchmark manifest, Pareto frontiers (return vs time), ablations, and robustness.
            </p>
          </div>
        </div>
      </div>

      {/* Authorship & University Review Footer Note */}
      <div className="p-6 rounded-2xl bg-slate-900/30 border border-slate-800 text-xs text-slate-400 flex flex-col sm:flex-row justify-between items-center gap-3">
        <div>
          <span className="font-semibold text-slate-300">Research Authors:</span> Shriram Narayanan, Evelyn Maria Paulraj, Koppesh P.
        </div>
        <div>
          <span>Department of Computer Science & Artificial Intelligence</span>
        </div>
      </div>
    </div>
  );
};
