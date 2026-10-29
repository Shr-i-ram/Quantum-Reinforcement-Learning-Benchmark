import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  Cpu,
  Activity,
  Layers,
  Sparkles,
  Info,
  ChevronRight,
  Gauge,
  Sliders,
} from 'lucide-react';
import {
  fetchEnvironments,
  createSession,
  stepSession,
  resetSession,
} from '../api';
import { EnvironmentMeta, SimSession } from '../types';

export const PlaygroundView: React.FC = () => {
  const [environments, setEnvironments] = useState<EnvironmentMeta[]>([]);
  const [selectedEnvId, setSelectedEnvId] = useState<string>('CartPole-v1');
  const [selectedAlgo, setSelectedAlgo] = useState<string>('DQN (Demonstration Solved)');
  
  const [session, setSession] = useState<SimSession | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(30); // FPS
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const playingRef = useRef(isPlaying);
  playingRef.current = isPlaying;

  const sessionRef = useRef(session);
  sessionRef.current = session;

  // Load environments on mount
  useEffect(() => {
    fetchEnvironments().then((envs) => {
      setEnvironments(envs);
      if (envs.length > 0) {
        setSelectedEnvId(envs[0].id);
        if (envs[0].compatible_algorithms.length > 0) {
          setSelectedAlgo(envs[0].compatible_algorithms[0]);
        }
      }
    }).catch((err) => {
      setErrorMsg(`Failed to load environments: ${err.message}`);
    });
  }, []);

  const currentEnv = environments.find((e) => e.id === selectedEnvId);

  // Initialize or recreate simulation session
  const initSession = async (envId: string, algo: string) => {
    setLoading(true);
    setIsPlaying(false);
    setErrorMsg(null);
    try {
      const sess = await createSession(envId, algo);
      setSession(sess);
    } catch (err: any) {
      setErrorMsg(`Failed to initialize session: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedEnvId && selectedAlgo) {
      initSession(selectedEnvId, selectedAlgo);
    }
  }, [selectedEnvId, selectedAlgo]);

  // Execute a single simulation step
  const executeStep = async (actionOverride?: any) => {
    const cur = sessionRef.current;
    if (!cur || cur.done) {
      setIsPlaying(false);
      return;
    }
    try {
      const nextSess = await stepSession(cur.session_id, actionOverride);
      setSession(nextSess);
      if (nextSess.done) {
        setIsPlaying(false);
      }
    } catch (err: any) {
      setIsPlaying(false);
      setErrorMsg(`Step error: ${err.message}`);
    }
  };

  // Reset simulation session
  const handleReset = async () => {
    if (!session) return;
    setIsPlaying(false);
    try {
      const resetSess = await resetSession(session.session_id);
      setSession(resetSess);
    } catch (err: any) {
      setErrorMsg(`Reset error: ${err.message}`);
    }
  };

  // Auto-play loop
  useEffect(() => {
    let timer: any = null;
    if (isPlaying) {
      timer = setInterval(() => {
        executeStep();
      }, 1000 / playbackSpeed);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlaying, playbackSpeed]);

  // Keyboard controls for human manual play
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!currentEnv) return;
      if (selectedAlgo !== 'Human Manual') return;

      const code = e.code;
      if (code in currentEnv.human_keys) {
        e.preventDefault();
        const action = currentEnv.human_keys[code];
        executeStep(action);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentEnv, selectedAlgo]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Top Controls Header */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 flex flex-wrap items-center justify-between gap-4">
        {/* Environment Selector */}
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Environment:</label>
          <select
            value={selectedEnvId}
            onChange={(e) => {
              const newEnvId = e.target.value;
              setSelectedEnvId(newEnvId);
              const env = environments.find((x) => x.id === newEnvId);
              if (env && env.compatible_algorithms.length > 0) {
                setSelectedAlgo(env.compatible_algorithms[0]);
              }
            }}
            className="bg-slate-950 border border-slate-700 text-white text-sm rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500"
          >
            {environments.map((env) => (
              <option key={env.id} value={env.id}>
                {env.name} ({env.id})
              </option>
            ))}
          </select>
        </div>

        {/* Algorithm Selector */}
        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold uppercase tracking-wider text-slate-400">Agent Controller:</label>
          <select
            value={selectedAlgo}
            onChange={(e) => setSelectedAlgo(e.target.value)}
            className="bg-slate-950 border border-slate-700 text-white text-sm rounded-xl px-3 py-2 focus:outline-none focus:border-cyan-500"
          >
            <option value="Human Manual">🎮 Human Manual Control</option>
            {currentEnv?.compatible_algorithms.map((algo) => (
              <option key={algo} value={algo}>
                {algo.includes('Demonstration') ? `⭐ ${algo}` : algo}
              </option>
            ))}
          </select>
        </div>

        {/* Playback Controls */}
        <div className="flex items-center gap-2">
          {selectedAlgo !== 'Human Manual' && (
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              disabled={loading || session?.done}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold transition ${
                isPlaying
                  ? 'bg-amber-500 hover:bg-amber-400 text-slate-950'
                  : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
              } disabled:opacity-40`}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? 'Pause' : 'Auto Play'}
            </button>
          )}

          <button
            onClick={() => executeStep()}
            disabled={loading || session?.done || isPlaying}
            className="flex items-center gap-1 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 transition disabled:opacity-40"
          >
            <SkipForward className="w-4 h-4" />
            Step
          </button>

          <button
            onClick={handleReset}
            disabled={loading}
            className="flex items-center gap-1 px-3 py-2 rounded-xl text-xs font-medium bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 transition"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>

          {/* Speed Selector */}
          <div className="flex items-center gap-1 ml-2 bg-slate-950 px-2 py-1 rounded-xl border border-slate-800 text-xs">
            <span className="text-slate-400 text-[10px]">Speed:</span>
            {[10, 30, 60].map((fps) => (
              <button
                key={fps}
                onClick={() => setPlaybackSpeed(fps)}
                className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                  playbackSpeed === fps ? 'bg-cyan-500 text-slate-950' : 'text-slate-400 hover:text-white'
                }`}
              >
                {fps === 10 ? '0.3x' : fps === 30 ? '1x' : '2x'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="bg-rose-950/50 border border-rose-500/50 text-rose-300 p-3 rounded-xl text-xs">
          {errorMsg}
        </div>
      )}

      {/* Main Simulation View & Telemetry Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Simulation Frame / Canvas (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="relative bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl flex items-center justify-center min-h-[380px]">
            {session?.frame_base64 ? (
              <img
                src={session.frame_base64}
                alt="Gymnasium Environment Simulation"
                className="w-full h-auto object-contain max-h-[460px]"
              />
            ) : (
              <div className="flex flex-col items-center justify-center text-slate-500 gap-2 p-8">
                <Activity className="w-8 h-8 animate-spin" />
                <span className="text-xs">Initializing Gymnasium simulation...</span>
              </div>
            )}

            {/* Episode Done Alert Overlay */}
            {session?.done && (
              <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center">
                <span className="px-3 py-1 rounded-full bg-cyan-950 text-cyan-400 border border-cyan-800 text-xs font-semibold mb-2">
                  Episode Completed
                </span>
                <div className="text-2xl font-bold text-white">
                  Final Return: <span className="text-cyan-400">{session.cumulative_reward.toFixed(1)}</span>
                </div>
                <p className="text-xs text-slate-400 mt-1 max-w-sm">
                  {session.termination_reason || 'Environment termination reached.'}
                </p>
                <button
                  onClick={handleReset}
                  className="mt-4 px-4 py-2 rounded-xl text-xs font-semibold bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition cursor-pointer"
                >
                  Reset & Run Again
                </button>
              </div>
            )}
          </div>

          {/* Human Control Keypad (when Human Manual is active) */}
          {selectedAlgo === 'Human Manual' && currentEnv && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-cyan-400 flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5" />
                  Manual Human Keypad & Guide
                </span>
                <span className="text-[11px] text-slate-400">{currentEnv.key_guide}</span>
              </div>

              <div className="flex flex-wrap gap-2">
                {currentEnv.action_labels.map((lbl, idx) => (
                  <button
                    key={idx}
                    onClick={() => executeStep(idx)}
                    disabled={session?.done}
                    className="flex-1 min-w-[120px] py-2.5 px-3 bg-slate-950 hover:bg-slate-800 active:bg-cyan-950 border border-slate-700 hover:border-cyan-500/50 rounded-xl text-xs font-semibold text-white transition disabled:opacity-40"
                  >
                    {lbl}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Environment Specs & Solved Threshold */}
          {currentEnv && (
            <div className="bg-slate-900/40 border border-slate-800/80 rounded-2xl p-4 text-xs text-slate-400 flex flex-wrap justify-between items-center gap-3">
              <div>
                <span className="text-slate-300 font-medium">Goal / Solved Threshold:</span>{' '}
                <span className="text-cyan-400 font-semibold">{currentEnv.solved_threshold}</span>{' '}
                ({currentEnv.reward_metric})
              </div>
              <div>
                <span className="text-slate-300 font-medium">Action Space:</span>{' '}
                <span className="text-white capitalize">{currentEnv.action_type} ({currentEnv.action_dim} dims)</span>
              </div>
            </div>
          )}
        </div>

        {/* Telemetry & Quantum Introspection (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Live Gauges Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Gauge className="w-4 h-4 text-cyan-400" />
              Live Episode Telemetry
            </h3>

            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">Step</span>
                <div className="text-lg font-bold text-white mt-0.5">{session?.step_count || 0}</div>
              </div>
              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">Step Reward</span>
                <div className="text-lg font-bold text-cyan-400 mt-0.5">
                  {session?.reward !== undefined ? session.reward.toFixed(1) : '0.0'}
                </div>
              </div>
              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase font-semibold">Return</span>
                <div className="text-lg font-bold text-emerald-400 mt-0.5">
                  {session?.cumulative_reward !== undefined ? session.cumulative_reward.toFixed(1) : '0.0'}
                </div>
              </div>
            </div>

            {/* State Variables Vector Display */}
            <div className="space-y-1.5">
              <span className="text-[11px] text-slate-400 font-medium">State Observation Vector:</span>
              <div className="space-y-1">
                {currentEnv?.obs_labels.slice(0, 4).map((label, idx) => {
                  const val = session?.state?.[idx] ?? 0;
                  return (
                    <div key={idx} className="flex justify-between items-center text-xs bg-slate-950 px-2.5 py-1.5 rounded-lg border border-slate-800/80">
                      <span className="text-slate-400">{label}</span>
                      <span className="font-mono text-cyan-300 font-semibold">{val.toFixed(4)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Deep Quantum Introspection Panel (if quantum) */}
          {session?.telemetry?.is_quantum ? (
            <div className="bg-gradient-to-b from-purple-950/20 to-slate-900/60 border border-purple-500/30 rounded-2xl p-5 space-y-4 shadow-lg shadow-purple-500/5">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-purple-400 flex items-center gap-1.5">
                  <Cpu className="w-4 h-4 text-purple-400" />
                  Quantum Introspection (Pauli-Z Readout)
                </h3>
                <span className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded-full bg-purple-950 text-purple-300 border border-purple-800/50">
                  4 Qubits Active
                </span>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed">
                Real-time expectation values ⟨Zᵢ⟩ measured from the 4-qubit circuit before linear projection into action space:
              </p>

              {/* 4 Pauli-Z Expectation Meters */}
              <div className="space-y-2">
                {session.telemetry.expectation_values?.map((expVal, qIdx) => {
                  const percentage = ((expVal + 1) / 2) * 100; // Map [-1, 1] to [0, 100]%
                  return (
                    <div key={qIdx} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-purple-300 font-semibold">Qubit {qIdx} (⟨Z_{qIdx}⟩)</span>
                        <span className="font-mono text-cyan-300">{expVal.toFixed(4)}</span>
                      </div>
                      <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                        <div
                          className="h-full bg-gradient-to-r from-purple-500 to-cyan-400 transition-all duration-150"
                          style={{ width: `${Math.max(5, Math.min(100, percentage))}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Projected Feature Vector Angles */}
              {session.telemetry.projected_features && (
                <div className="pt-2 border-t border-slate-800 space-y-1">
                  <span className="text-[11px] text-slate-400">Angle Encodings x ∈ [-1, 1]⁴:</span>
                  <div className="flex gap-1.5">
                    {session.telemetry.projected_features.map((fVal, fIdx) => (
                      <div key={fIdx} className="flex-1 bg-slate-950 py-1 text-center rounded font-mono text-[11px] text-slate-300 border border-slate-800">
                        {fVal.toFixed(2)}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Classical Action Telemetry (Q-values or probabilities) */
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Activity className="w-4 h-4 text-cyan-400" />
                Classical Decision Introspection
              </h3>

              {session?.telemetry?.q_values && (
                <div className="space-y-2">
                  <span className="text-[11px] text-slate-400 font-medium">Estimated Q-Values Q(s, a):</span>
                  <div className="space-y-1.5">
                    {session.telemetry.q_values.map((qVal, aIdx) => (
                      <div key={aIdx} className="flex justify-between items-center text-xs bg-slate-950 px-2.5 py-1.5 rounded-lg border border-slate-800">
                        <span className="text-slate-400">Action {aIdx}</span>
                        <span className="font-mono text-emerald-400 font-semibold">{qVal.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {session?.telemetry?.action_probabilities && (
                <div className="space-y-2">
                  <span className="text-[11px] text-slate-400 font-medium">Policy Distribution π(a|s):</span>
                  <div className="space-y-1.5">
                    {session.telemetry.action_probabilities.map((prob, aIdx) => (
                      <div key={aIdx} className="space-y-0.5">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">Action {aIdx}</span>
                          <span className="font-mono text-cyan-300 font-semibold">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                          <div
                            className="h-full bg-cyan-400"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
