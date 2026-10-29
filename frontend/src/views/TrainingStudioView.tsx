import React, { useState, useEffect } from 'react';
import {
  Flame,
  Play,
  Square,
  Activity,
  CheckCircle2,
  Clock,
  Zap,
  RotateCcw,
  Sliders,
  TrendingUp,
  Cpu,
} from 'lucide-react';
import { startTraining, stopTraining, fetchTrainingStatus } from '../api';
import { TrainingStatus } from '../types';

interface TrainingStudioViewProps {
  onNavigateTab: (tabId: string) => void;
}

export const TrainingStudioView: React.FC<TrainingStudioViewProps> = ({
  onNavigateTab,
}) => {
  const [selectedEnv, setSelectedEnv] = useState<string>('CartPole-v1');
  const [selectedAlgo, setSelectedAlgo] = useState<string>('DQN');
  const [totalSteps, setTotalSteps] = useState<number>(3000);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [seed, setSeed] = useState<number>(42);

  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [actionMsg, setActionMsg] = useState<string | null>(null);

  // Poll status every 1.2 seconds
  useEffect(() => {
    let timer: any = null;
    const poll = async () => {
      try {
        const s = await fetchTrainingStatus();
        setStatus(s);
      } catch (e) {
        console.error(e);
      }
    };

    poll();
    timer = setInterval(poll, 1200);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  const handleStart = async () => {
    setActionMsg(null);
    try {
      const res = await startTraining(selectedAlgo, selectedEnv, totalSteps, learningRate, seed);
      if (!res.success) {
        setActionMsg(`Failed to start: ${res.message}`);
      } else {
        setActionMsg(res.message);
      }
    } catch (err: any) {
      setActionMsg(`Error: ${err.message}`);
    }
  };

  const handleStop = async () => {
    try {
      const res = await stopTraining();
      setActionMsg(res.message);
    } catch (err: any) {
      setActionMsg(`Error: ${err.message}`);
    }
  };

  const progressPct = status
    ? Math.min(100, (status.current_step / (status.total_steps || 1)) * 100)
    : 0;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-orange-950/40 via-slate-900 to-slate-950 border border-orange-500/30 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-orange-950 text-orange-400 border border-orange-800/60 text-xs font-semibold mb-2">
            <Flame className="w-3.5 h-3.5" />
            Background Live Policy Optimization
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white">Live Training Studio</h1>
          <p className="text-sm text-slate-300 mt-1 max-w-2xl">
            Execute real background training sessions for classical or quantum agents directly on your machine.
            Watch policy updates occur in real-time without simulated or pre-recorded metrics.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {status?.is_running ? (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs font-bold bg-rose-600 hover:bg-rose-500 text-white transition shadow-lg shadow-rose-600/25 cursor-pointer"
            >
              <Square className="w-4 h-4 fill-white" />
              Stop Training Session
            </button>
          ) : (
            <button
              onClick={handleStart}
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl text-xs font-bold bg-orange-500 hover:bg-orange-400 text-slate-950 transition shadow-lg shadow-orange-500/25 cursor-pointer"
            >
              <Play className="w-4 h-4 fill-slate-950" />
              Start Real Training
            </button>
          )}
        </div>
      </div>

      {actionMsg && (
        <div className="bg-slate-900 border border-slate-700 text-slate-300 p-3 rounded-xl text-xs">
          {actionMsg}
        </div>
      )}

      {/* Hyperparameter Configuration Bar */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 items-center">
        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">Environment:</label>
          <select
            value={selectedEnv}
            disabled={status?.is_running}
            onChange={(e) => setSelectedEnv(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-orange-500 disabled:opacity-50"
          >
            <option value="CartPole-v1">CartPole-v1 (Fastest)</option>
            <option value="MountainCar-v0">MountainCar-v0</option>
            <option value="Pendulum-v1">Pendulum-v1</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">Algorithm:</label>
          <select
            value={selectedAlgo}
            disabled={status?.is_running}
            onChange={(e) => setSelectedAlgo(e.target.value)}
            className="w-full bg-slate-950 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-orange-500 disabled:opacity-50"
          >
            <option value="DQN">Classical DQN</option>
            <option value="Q-DQN">Hybrid Q-DQN (4 Qubits)</option>
            <option value="PPO">Classical PPO</option>
            <option value="Q-PPO">Hybrid Q-PPO (4 Qubits)</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">Total Steps:</label>
          <select
            value={totalSteps}
            disabled={status?.is_running}
            onChange={(e) => setTotalSteps(Number(e.target.value))}
            className="w-full bg-slate-950 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-orange-500 disabled:opacity-50"
          >
            <option value="1500">1,500 Steps (Fastest Demo)</option>
            <option value="3000">3,000 Steps (Standard)</option>
            <option value="6000">6,000 Steps (Extended)</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">Learning Rate:</label>
          <select
            value={learningRate}
            disabled={status?.is_running}
            onChange={(e) => setLearningRate(Number(e.target.value))}
            className="w-full bg-slate-950 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-orange-500 disabled:opacity-50"
          >
            <option value="0.001">1e-3 (Standard)</option>
            <option value="0.0005">5e-4</option>
            <option value="0.0001">1e-4</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-1">Random Seed:</label>
          <input
            type="number"
            value={seed}
            disabled={status?.is_running}
            onChange={(e) => setSeed(Number(e.target.value))}
            className="w-full bg-slate-950 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-orange-500 disabled:opacity-50"
          />
        </div>
      </div>

      {/* Live Training Telemetry */}
      {status && (
        <div className="space-y-6">
          {/* Progress Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
              <div className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${status.is_running ? 'bg-orange-500 animate-ping' : 'bg-slate-600'}`} />
                <h3 className="text-sm font-bold text-white">
                  {status.is_running
                    ? `Training ${status.algorithm} on ${status.environment}...`
                    : status.current_step > 0
                    ? `Session Completed (${status.algorithm} on ${status.environment})`
                    : 'Ready to Train'}
                </h3>
              </div>

              <div className="text-xs text-slate-400">
                Elapsed: <span className="font-mono text-white font-semibold">{status.elapsed_time_s.toFixed(1)}s</span> | Throughput:{' '}
                <span className="font-mono text-cyan-300 font-semibold">{status.fps} steps/s</span>
              </div>
            </div>

            {/* Progress Bar */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-slate-400">
                <span>Progress: {progressPct.toFixed(1)}%</span>
                <span className="font-mono">
                  {status.current_step.toLocaleString()} / {status.total_steps.toLocaleString()} steps
                </span>
              </div>
              <div className="h-3 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                <div
                  className="h-full bg-gradient-to-r from-orange-500 to-amber-400 transition-all duration-300"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>

            {/* Metric Counters */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">Episodes Completed</span>
                <div className="text-xl font-bold text-white mt-0.5">{status.episodes}</div>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">Latest Ep Return</span>
                <div className="text-xl font-bold text-orange-400 mt-0.5">{status.latest_reward.toFixed(1)}</div>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">Mean Return (Last 10)</span>
                <div className="text-xl font-bold text-emerald-400 mt-0.5">{status.avg_reward_10.toFixed(1)}</div>
              </div>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">Quantum VQC Status</span>
                <div className="text-sm font-bold text-purple-400 mt-1">
                  {status.is_quantum ? '4 Qubits Active' : 'Classical Dense'}
                </div>
              </div>
            </div>
          </div>

          {/* Episode Reward History Chart */}
          {status.history?.episode_rewards?.length > 0 && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                Live Episode Reward Learning Curve
              </h3>

              <div className="h-44 flex items-end gap-1.5 pt-4 pb-2 px-2 bg-slate-950 rounded-xl border border-slate-800 overflow-x-auto">
                {status.history.episode_rewards.map((r, idx) => {
                  const maxR = Math.max(...status.history.episode_rewards, 20);
                  const minR = Math.min(...status.history.episode_rewards, 0);
                  const range = maxR - minR || 1;
                  const hPct = Math.max(8, ((r - minR) / range) * 100);

                  return (
                    <div
                      key={idx}
                      title={`Episode ${idx + 1}: Return ${r.toFixed(1)}`}
                      className="flex-1 min-w-[12px] bg-emerald-500 hover:bg-emerald-400 rounded-t transition-all group relative cursor-pointer"
                      style={{ height: `${hPct}%` }}
                    >
                      <div className="hidden group-hover:block absolute bottom-full mb-1 left-1/2 -translate-x-1/2 bg-slate-900 border border-slate-700 text-white text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap z-20">
                        Ep {idx + 1}: {r.toFixed(0)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Test in Playground Action */}
          {status.saved_checkpoint_path && (
            <div className="bg-emerald-950/30 border border-emerald-500/40 p-4 rounded-2xl flex flex-col sm:flex-row justify-between items-center gap-3">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                <span className="text-xs text-emerald-300 font-semibold">
                  Checkpoint saved! Ready for live testing in Playground.
                </span>
              </div>
              <button
                onClick={() => onNavigateTab('playground')}
                className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-bold rounded-xl transition cursor-pointer"
              >
                Inspect in Playground →
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
