import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Sliders,
  Binary,
  Layers,
  Sparkles,
  Zap,
  Clock,
  HardDrive,
  AlertTriangle,
  Activity,
} from 'lucide-react';
import { simulateQuantum, fetchQuantumScaling } from '../api';
import { QuantumSimResponse, QuantumScalingData } from '../types';

export const QuantumLabView: React.FC = () => {
  // Feature angles x0, x1, x2, x3 in [-1, 1]
  const [features, setFeatures] = useState<number[]>([0.35, -0.6, 0.8, -0.25]);
  const [depth, setDepth] = useState<number>(2);

  const [simResult, setSimResult] = useState<QuantumSimResponse | null>(null);
  const [scalingData, setScalingData] = useState<QuantumScalingData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [shots, setShots] = useState<number>(1000);

  // Run live PennyLane circuit simulation
  const runSimulation = async (feats: number[], d: number) => {
    try {
      const res = await simulateQuantum(feats, d);
      setSimResult(res);
    } catch (e) {
      console.error(e);
    }
  };

  // Run on mount or when sliders change
  useEffect(() => {
    runSimulation(features, depth);
  }, [features, depth]);

  // Load scaling data
  useEffect(() => {
    fetchQuantumScaling(shots, depth).then(setScalingData).catch(console.error);
  }, [shots, depth]);

  const handleSliderChange = (idx: number, val: number) => {
    const updated = [...features];
    updated[idx] = val;
    setFeatures(updated);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-8">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-purple-950/40 via-slate-900 to-slate-950 border border-purple-500/30 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-purple-950 text-purple-400 border border-purple-800/60 text-xs font-semibold mb-2">
            <Cpu className="w-3.5 h-3.5" />
            Live PennyLane QNode Explorer
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white">Quantum Circuit & Measurement Lab</h1>
          <p className="text-sm text-slate-300 mt-1 max-w-2xl">
            Directly manipulate continuous state feature projections and observe instantaneous Pauli-Z expectation values and 16-basis state probabilities computed via real PennyLane quantum simulations.
          </p>
        </div>

        <div className="flex items-center gap-2 bg-slate-950 px-3 py-2 rounded-xl border border-slate-800 text-xs">
          <span className="w-2 h-2 rounded-full bg-purple-400 animate-pulse" />
          <span className="text-slate-300">Backend: PennyLane default.qubit</span>
        </div>
      </div>

      {/* Interactive Circuit Studio */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Sliders & Circuit Diagram (6 cols) */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Sliders className="w-4 h-4 text-purple-400" />
                Input Feature Angles xᵢ ∈ [-1.0, 1.0]
              </h3>
              <div className="flex items-center gap-1 text-xs">
                <span className="text-slate-400">Depth:</span>
                {[1, 2, 3, 4].map((d) => (
                  <button
                    key={d}
                    onClick={() => setDepth(d)}
                    className={`px-2 py-0.5 rounded font-semibold text-xs ${
                      depth === d ? 'bg-purple-500 text-slate-950' : 'bg-slate-800 text-slate-400 hover:text-white'
                    }`}
                  >
                    {d}L
                  </button>
                ))}
              </div>
            </div>

            {/* 4 Interactive Feature Sliders */}
            <div className="space-y-3">
              {features.map((val, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-purple-300 font-semibold">Qubit {idx} Input (x_{idx})</span>
                    <span className="font-mono text-cyan-300">{val.toFixed(3)} rad</span>
                  </div>
                  <input
                    type="range"
                    min="-1.0"
                    max="1.0"
                    step="0.01"
                    value={val}
                    onChange={(e) => handleSliderChange(idx, parseFloat(e.target.value))}
                    className="w-full accent-purple-400 cursor-pointer"
                  />
                </div>
              ))}
            </div>

            <div className="pt-2 border-t border-slate-800">
              <div className="flex justify-between items-center text-xs text-slate-400">
                <span>Angle Encoding Formula:</span>
                <span className="font-mono text-cyan-300 bg-slate-950 px-2 py-1 rounded">|ψ₀⟩ = ⊗ᵢ RY(xᵢ)|0⟩</span>
              </div>
            </div>
          </div>

          {/* Circuit ASCII Diagram */}
          {simResult && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-2">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">
                4-Qubit Variational Architecture Layout
              </h3>
              <pre className="bg-slate-950 p-4 rounded-xl border border-slate-800 text-[11px] font-mono text-cyan-300 overflow-x-auto leading-relaxed">
                {simResult.circuit_diagram}
              </pre>
            </div>
          )}
        </div>

        {/* Expectation Values & Probability Distribution (6 cols) */}
        <div className="lg:col-span-6 space-y-4">
          {/* Pauli-Z Expectation Values Readout */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-cyan-400" />
              Instantaneous Pauli-Z Expectation Values ⟨Zᵢ⟩
            </h3>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-center">
              {simResult?.expectation_values.map((ev, idx) => (
                <div key={idx} className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <span className="text-[10px] text-purple-400 uppercase font-semibold">Qubit {ev.qubit}</span>
                  <div className="text-lg font-bold font-mono text-cyan-300 mt-0.5">
                    {ev.expval.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-slate-400 mt-0.5">{ev.pauli}</div>
                </div>
              ))}
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              These 4 expectation values form the bottleneck representation feed directly into the classical policy/value head to produce action distributions.
            </p>
          </div>

          {/* 16 Basis State Probability Distribution */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Binary className="w-4 h-4 text-purple-400" />
              Computational Basis Probabilities |⟨s|ψ⟩|² (2⁴ = 16 States)
            </h3>

            <div className="grid grid-cols-4 sm:grid-cols-8 gap-1.5">
              {simResult?.basis_probabilities.map((item, idx) => {
                const heightPct = Math.max(5, item.probability * 100);
                return (
                  <div key={idx} className="bg-slate-950 p-1.5 rounded-lg border border-slate-800 text-center flex flex-col justify-end h-24">
                    <div
                      className="w-full bg-gradient-to-t from-purple-600 to-cyan-400 rounded-t transition-all duration-150"
                      style={{ height: `${heightPct}%` }}
                    />
                    <div className="text-[9px] font-mono text-slate-400 mt-1 truncate">{item.basis}</div>
                    <div className="text-[9px] font-mono text-cyan-300 font-semibold">
                      {(item.probability * 100).toFixed(0)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Quantum vs Classical Scaling & Barren Plateau Analysis */}
      {scalingData && (
        <div className="bg-slate-900/50 border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-6">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div>
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <HardDrive className="w-5 h-5 text-cyan-400" />
                Computational Scaling & Physical QPU Bottlenecks
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Rigorous complexity comparison: Classical statevector simulation (O(2ⁿ)) vs NISQ QPU shot-based sampling.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">Shots (S):</span>
              <select
                value={shots}
                onChange={(e) => setShots(Number(e.target.value))}
                className="bg-slate-950 border border-slate-700 text-white text-xs rounded-xl px-2.5 py-1.5 focus:outline-none focus:border-cyan-500"
              >
                <option value="500">500 Shots</option>
                <option value="1000">1000 Shots (Standard)</option>
                <option value="5000">5000 Shots</option>
              </select>
            </div>
          </div>

          {/* Scaling Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                <tr>
                  <th className="py-2.5 px-3 font-semibold">Qubits (n)</th>
                  <th className="py-2.5 px-3 font-semibold">State Dim (2ⁿ)</th>
                  <th className="py-2.5 px-3 font-semibold text-cyan-400">Classical Statevector Memory</th>
                  <th className="py-2.5 px-3 font-semibold text-cyan-400">Classical Sim Time</th>
                  <th className="py-2.5 px-3 font-semibold text-purple-400">Physical QPU Runtime</th>
                  <th className="py-2.5 px-3 font-semibold text-rose-400">Barren Plateau Var[∂L]</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {scalingData.qubits.slice(0, 9).map((n, idx) => {
                  const memMb = scalingData.classical_memory_mb[idx];
                  const simTime = scalingData.classical_sim_time_ms[idx];
                  const qpuTime = scalingData.qpu_runtime_ms[idx];
                  const bpVar = scalingData.barren_plateau_variance[idx];
                  const isCrossover = n >= scalingData.crossover_qubit_estimate;

                  return (
                    <tr key={idx} className={n === 4 ? 'bg-cyan-950/20 font-medium' : 'hover:bg-slate-800/30'}>
                      <td className="py-2 px-3">
                        <span className="font-bold text-white">{n} Qubits</span>
                        {n === 4 && <span className="ml-2 text-[10px] text-cyan-400 font-semibold">(Our Model)</span>}
                      </td>
                      <td className="py-2 px-3 font-mono text-slate-300">{(2 ** n).toLocaleString()}</td>
                      <td className="py-2 px-3 font-mono text-cyan-300">
                        {memMb < 1024 ? `${memMb.toFixed(2)} MB` : `${(memMb / 1024).toFixed(2)} GB`}
                      </td>
                      <td className="py-2 px-3 font-mono text-cyan-300">
                        {simTime < 1000 ? `${simTime.toFixed(2)} ms` : `${(simTime / 1000).toFixed(2)} s`}
                      </td>
                      <td className="py-2 px-3 font-mono text-purple-300">
                        {qpuTime.toFixed(1)} ms
                      </td>
                      <td className="py-2 px-3 font-mono text-rose-400">
                        {bpVar.toExponential(2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Research Insight Callouts */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-1">
              <span className="text-cyan-400 font-semibold">The 4-Qubit Simulation Paradox:</span>
              <p className="text-slate-400">{scalingData.insights.simulation_bottleneck}</p>
            </div>
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-1">
              <span className="text-purple-400 font-semibold">The Classical Memory Wall:</span>
              <p className="text-slate-400">{scalingData.insights.memory_wall}</p>
            </div>
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 space-y-1">
              <span className="text-rose-400 font-semibold">Barren Plateau Phenomenon:</span>
              <p className="text-slate-400">{scalingData.insights.barren_plateau_hazard}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
