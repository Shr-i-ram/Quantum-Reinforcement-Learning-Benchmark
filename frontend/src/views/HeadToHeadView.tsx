import React, { useState } from 'react';
import {
  Swords,
  Play,
  RotateCcw,
  BarChart2,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Activity,
  Award,
  Layers,
} from 'lucide-react';
import { runBatchDuel } from '../api';
import { BatchDuelResult } from '../types';

export const HeadToHeadView: React.FC = () => {
  const [selectedEnvId, setSelectedEnvId] = useState<string>('CartPole-v1');
  const [algoA, setAlgoA] = useState<string>('DQN (Demonstration Solved)');
  const [algoB, setAlgoB] = useState<string>('Q-DQN');
  const [numEpisodes, setNumEpisodes] = useState<number>(15);

  const [loading, setLoading] = useState<boolean>(false);
  const [duelResult, setDuelResult] = useState<BatchDuelResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleLaunchDuel = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await runBatchDuel(selectedEnvId, algoA, algoB, numEpisodes);
      setDuelResult(res);
    } catch (err: any) {
      setErrorMsg(`Failed to run batch duel: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-rose-950/40 via-slate-900 to-slate-950 border border-rose-500/30 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-950 text-rose-400 border border-rose-800/60 text-xs font-semibold mb-2">
            <Swords className="w-3.5 h-3.5" />
            Rigorous Statistical Duel Runner
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white">Classical vs Quantum Head-to-Head</h1>
          <p className="text-sm text-slate-300 mt-1 max-w-2xl">
            Evaluate two algorithms simultaneously across identical initial states and random seeds to measure performance variance, win rates, and trajectory divergence.
          </p>
        </div>

        <button
          onClick={handleLaunchDuel}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-bold bg-rose-500 hover:bg-rose-400 text-slate-950 transition shadow-lg shadow-rose-500/25 disabled:opacity-50 cursor-pointer"
        >
          {loading ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              <span>Simulating Duels...</span>
            </>
          ) : (
            <>
              <Swords className="w-4 h-4" />
              <span>Execute Batch Duel</span>
            </>
          )}
        </button>
      </div>

      {/* Configuration Bar */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Environment:</label>
          <select
            value={selectedEnvId}
            onChange={(e) => {
              const env = e.target.value;
              setSelectedEnvId(env);
              if (env === 'CartPole-v1') {
                setAlgoA('DQN (Demonstration Solved)');
                setAlgoB('Q-DQN');
              } else if (env === 'Pendulum-v1') {
                setAlgoA('PPO');
                setAlgoB('Q-PPO');
              } else {
                setAlgoA('DQN');
                setAlgoB('Q-DQN');
              }
            }}
            className="bg-slate-950 border border-slate-700 text-white text-sm rounded-xl px-3 py-2 focus:outline-none focus:border-rose-500"
          >
            <option value="CartPole-v1">CartPole-v1 (Discrete)</option>
            <option value="MountainCar-v0">MountainCar-v0 (Discrete)</option>
            <option value="LunarLander-v3">LunarLander-v3 (Discrete)</option>
            <option value="Pendulum-v1">Pendulum-v1 (Continuous)</option>
          </select>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold uppercase tracking-wider text-cyan-400">Model A (Classical):</label>
          <select
            value={algoA}
            onChange={(e) => setAlgoA(e.target.value)}
            className="bg-slate-950 border border-slate-700 text-white text-sm rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500"
          >
            {selectedEnvId === 'CartPole-v1' && <option value="DQN (Demonstration Solved)">DQN (Demonstration Solved)</option>}
            <option value="DQN">Classical DQN</option>
            <option value="PPO">Classical PPO</option>
            <option value="SAC">Classical SAC</option>
            <option value="Heuristic">Heuristic Rule</option>
            <option value="Random Baseline">Random Baseline</option>
          </select>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold uppercase tracking-wider text-purple-400">Model B (Quantum):</label>
          <select
            value={algoB}
            onChange={(e) => setAlgoB(e.target.value)}
            className="bg-slate-950 border border-slate-700 text-white text-sm rounded-xl px-3 py-2 focus:outline-none focus:border-purple-500"
          >
            <option value="Q-DQN">Hybrid Q-DQN (4 Qubits)</option>
            <option value="Q-PPO">Hybrid Q-PPO (4 Qubits)</option>
            <option value="Q-SAC">Hybrid Q-SAC (4 Qubits)</option>
          </select>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Trials:</label>
          <select
            value={numEpisodes}
            onChange={(e) => setNumEpisodes(Number(e.target.value))}
            className="bg-slate-950 border border-slate-700 text-white text-sm rounded-xl px-3 py-2 focus:outline-none focus:border-rose-500"
          >
            <option value="10">10 Episodes</option>
            <option value="15">15 Episodes</option>
            <option value="25">25 Episodes</option>
          </select>
        </div>
      </div>

      {errorMsg && (
        <div className="bg-rose-950/50 border border-rose-500/50 text-rose-300 p-3 rounded-xl text-xs">
          {errorMsg}
        </div>
      )}

      {/* Duel Scoreboard & Statistical Outcome */}
      {duelResult && (
        <div className="space-y-6">
          {/* Win/Loss Big Metric Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 sm:p-8">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 mb-6 text-center">
              Batch Duel Empirical Outcome ({duelResult.n_episodes} Synchronized Episodes)
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
              {/* Model A Summary */}
              <div className="bg-slate-950 p-6 rounded-2xl border border-cyan-500/40 text-center space-y-2">
                <span className="text-xs uppercase font-semibold text-cyan-400 tracking-wider">Model A (Classical)</span>
                <div className="text-xl font-bold text-white">{duelResult.algo_a}</div>
                <div className="text-3xl font-extrabold text-cyan-400 mt-2">
                  {duelResult.results_a.wins} <span className="text-sm font-normal text-slate-400">Wins</span>
                </div>
                <div className="text-xs text-slate-300">
                  Mean Return: <span className="font-mono text-cyan-300 font-bold">{duelResult.results_a.mean_return.toFixed(1)}</span> ± {duelResult.results_a.std_return.toFixed(1)}
                </div>
                <div className="text-[11px] text-slate-400">
                  Range: [{duelResult.results_a.min_return.toFixed(0)}, {duelResult.results_a.max_return.toFixed(0)}] | Median: {duelResult.results_a.median_return.toFixed(0)}
                </div>
              </div>

              {/* VS Split Graphic */}
              <div className="text-center space-y-3">
                <div className="text-2xl font-black text-rose-400 tracking-wider">VS</div>
                <div className="text-xs text-slate-400">
                  Ties / Identical: <span className="font-bold text-white">{duelResult.ties}</span>
                </div>
                {/* Win rate bar */}
                <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800 flex">
                  <div
                    className="bg-cyan-500 transition-all duration-300"
                    style={{
                      width: `${(duelResult.results_a.wins / duelResult.n_episodes) * 100}%`,
                    }}
                  />
                  <div
                    className="bg-slate-700 transition-all duration-300"
                    style={{
                      width: `${(duelResult.ties / duelResult.n_episodes) * 100}%`,
                    }}
                  />
                  <div
                    className="bg-purple-500 transition-all duration-300"
                    style={{
                      width: `${(duelResult.results_b.wins / duelResult.n_episodes) * 100}%`,
                    }}
                  />
                </div>
                <div className="flex justify-between text-[11px] text-slate-400">
                  <span className="text-cyan-400 font-semibold">{((duelResult.results_a.wins / duelResult.n_episodes) * 100).toFixed(0)}% A</span>
                  <span className="text-purple-400 font-semibold">{((duelResult.results_b.wins / duelResult.n_episodes) * 100).toFixed(0)}% B</span>
                </div>
              </div>

              {/* Model B Summary */}
              <div className="bg-slate-950 p-6 rounded-2xl border border-purple-500/40 text-center space-y-2">
                <span className="text-xs uppercase font-semibold text-purple-400 tracking-wider">Model B (Quantum)</span>
                <div className="text-xl font-bold text-white">{duelResult.algo_b}</div>
                <div className="text-3xl font-extrabold text-purple-400 mt-2">
                  {duelResult.results_b.wins} <span className="text-sm font-normal text-slate-400">Wins</span>
                </div>
                <div className="text-xs text-slate-300">
                  Mean Return: <span className="font-mono text-purple-300 font-bold">{duelResult.results_b.mean_return.toFixed(1)}</span> ± {duelResult.results_b.std_return.toFixed(1)}
                </div>
                <div className="text-[11px] text-slate-400">
                  Range: [{duelResult.results_b.min_return.toFixed(0)}, {duelResult.results_b.max_return.toFixed(0)}] | Median: {duelResult.results_b.median_return.toFixed(0)}
                </div>
              </div>
            </div>
          </div>

          {/* Episode-by-Episode Returns Table */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-cyan-400" />
              Episode-by-Episode Seed Returns Comparison
            </h3>

            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-950 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-4 font-semibold">Trial #</th>
                    <th className="py-2.5 px-4 font-semibold text-cyan-400">{duelResult.algo_a} Return</th>
                    <th className="py-2.5 px-4 font-semibold text-cyan-400">Steps</th>
                    <th className="py-2.5 px-4 font-semibold text-purple-400">{duelResult.algo_b} Return</th>
                    <th className="py-2.5 px-4 font-semibold text-purple-400">Steps</th>
                    <th className="py-2.5 px-4 font-semibold">Delta (A - B)</th>
                    <th className="py-2.5 px-4 font-semibold">Winner</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60">
                  {duelResult.results_a.returns.map((retA, idx) => {
                    const retB = duelResult.results_b.returns[idx];
                    const lenA = duelResult.results_a.lengths[idx];
                    const lenB = duelResult.results_b.lengths[idx];
                    const delta = retA - retB;
                    const winner = delta > 0 ? duelResult.algo_a : delta < 0 ? duelResult.algo_b : 'Tie';
                    return (
                      <tr key={idx} className="hover:bg-slate-800/30">
                        <td className="py-2 px-4 font-medium text-slate-300">Trial {idx + 1}</td>
                        <td className="py-2 px-4 font-mono text-cyan-300 font-semibold">{retA.toFixed(1)}</td>
                        <td className="py-2 px-4 text-slate-400">{lenA}</td>
                        <td className="py-2 px-4 font-mono text-purple-300 font-semibold">{retB.toFixed(1)}</td>
                        <td className="py-2 px-4 text-slate-400">{lenB}</td>
                        <td className={`py-2 px-4 font-mono font-semibold ${delta > 0 ? 'text-emerald-400' : delta < 0 ? 'text-rose-400' : 'text-slate-400'}`}>
                          {delta > 0 ? `+${delta.toFixed(1)}` : delta.toFixed(1)}
                        </td>
                        <td className="py-2 px-4">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            winner === duelResult.algo_a
                              ? 'bg-cyan-950 text-cyan-300 border border-cyan-800'
                              : winner === duelResult.algo_b
                              ? 'bg-purple-950 text-purple-300 border border-purple-800'
                              : 'bg-slate-800 text-slate-400'
                          }`}>
                            {winner}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
