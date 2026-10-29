import React, { useState, useEffect, useRef } from 'react';
import confetti from 'canvas-confetti';
import {
  Trophy,
  Target,
  RotateCcw,
  Sparkles,
  Award,
  CheckCircle2,
  Sliders,
  ChevronRight,
  User,
  Send,
} from 'lucide-react';
import {
  fetchChallenges,
  recordChallenge,
  createSession,
  stepSession,
  resetSession,
  fetchEnvironments,
} from '../api';
import { ChallengePreset, LeaderboardEntry, SimSession, EnvironmentMeta } from '../types';

export const ChallengeArenaView: React.FC = () => {
  const [presets, setPresets] = useState<Record<string, ChallengePreset>>({});
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [selectedEnvId, setSelectedEnvId] = useState<string>('CartPole-v1');
  const [playerName, setPlayerName] = useState<string>('');
  
  const [session, setSession] = useState<SimSession | null>(null);
  const [environments, setEnvironments] = useState<EnvironmentMeta[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [hasCelebrated, setHasCelebrated] = useState<boolean>(false);
  const [submissionMsg, setSubmissionMsg] = useState<string | null>(null);

  const sessionRef = useRef(session);
  sessionRef.current = session;

  // Load challenges and environments
  const loadData = async () => {
    try {
      const data = await fetchChallenges();
      setPresets(data.presets);
      setLeaderboard(data.leaderboard);
      const envs = await fetchEnvironments();
      setEnvironments(envs);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const currentPreset = presets[selectedEnvId];
  const currentEnv = environments.find((e) => e.id === selectedEnvId);

  // Initialize session for human play
  const initChallengeSession = async (envId: string) => {
    setLoading(true);
    setHasCelebrated(false);
    setSubmissionMsg(null);
    try {
      const sess = await createSession(envId, 'Human Manual');
      setSession(sess);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedEnvId) {
      initChallengeSession(selectedEnvId);
    }
  }, [selectedEnvId]);

  // Step action
  const handleAction = async (act: any) => {
    const cur = sessionRef.current;
    if (!cur || cur.done) return;

    try {
      const nextSess = await stepSession(cur.session_id, act);
      setSession(nextSess);

      // Check if target score is met
      if (currentPreset && !hasCelebrated) {
        let reached = false;
        if (selectedEnvId === 'MountainCar-v0') {
          reached = nextSess.cumulative_reward >= currentPreset.target_score;
        } else {
          reached = nextSess.cumulative_reward >= currentPreset.target_score;
        }

        if (reached) {
          setHasCelebrated(true);
          confetti({
            particleCount: 80,
            spread: 70,
            origin: { y: 0.6 },
          });
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!currentEnv) return;
      const code = e.code;
      if (code in currentEnv.human_keys) {
        e.preventDefault();
        handleAction(currentEnv.human_keys[code]);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentEnv, currentPreset, hasCelebrated]);

  const handleReset = async () => {
    if (!session) return;
    setHasCelebrated(false);
    setSubmissionMsg(null);
    try {
      const resetSess = await resetSession(session.session_id);
      setSession(resetSess);
    } catch (e) {
      console.error(e);
    }
  };

  const handleSubmitScore = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!session) return;

    const name = playerName.trim() || 'Review Committee Member';
    try {
      const res = await recordChallenge(
        name,
        selectedEnvId,
        session.cumulative_reward,
        session.step_count,
        session.done
      );
      setSubmissionMsg(`Recorded score ${session.cumulative_reward.toFixed(1)} for ${name}!`);
      const updated = await fetchChallenges();
      setLeaderboard(updated.leaderboard);
    } catch (err: any) {
      setSubmissionMsg(`Error recording score: ${err.message}`);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-amber-950/40 via-slate-900 to-slate-950 border border-amber-500/30 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-950 text-amber-400 border border-amber-800/60 text-xs font-semibold mb-2">
            <Trophy className="w-3.5 h-3.5" />
            Human vs AI Benchmark Challenge
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white">Can You Beat the AI Agents?</h1>
          <p className="text-sm text-slate-300 mt-1 max-w-2xl">
            Take direct manual control of the physical environments. Attempt to beat the classical solved score or the 4-qubit quantum hybrid benchmark!
          </p>
        </div>

        {/* Challenge Environment Selection Buttons */}
        <div className="flex flex-wrap gap-2">
          {Object.keys(presets).map((envId) => (
            <button
              key={envId}
              onClick={() => setSelectedEnvId(envId)}
              className={`px-3 py-2 rounded-xl text-xs font-semibold transition ${
                selectedEnvId === envId
                  ? 'bg-amber-500 text-slate-950 shadow-lg shadow-amber-500/20'
                  : 'bg-slate-900 hover:bg-slate-800 text-slate-300 border border-slate-700'
              }`}
            >
              {envId.replace('-v1', '').replace('-v0', '').replace('-v3', '')}
            </button>
          ))}
        </div>
      </div>

      {/* Main Challenge Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Playable Canvas & Keypad (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="relative bg-slate-950 rounded-2xl border border-slate-800 overflow-hidden shadow-2xl flex items-center justify-center min-h-[380px]">
            {session?.frame_base64 ? (
              <img
                src={session.frame_base64}
                alt="Gymnasium Environment Challenge"
                className="w-full h-auto object-contain max-h-[440px]"
              />
            ) : (
              <div className="text-xs text-slate-500">Loading challenge frame...</div>
            )}

            {/* Target Reached Banner */}
            {hasCelebrated && (
              <div className="absolute top-4 left-4 right-4 bg-emerald-500 text-slate-950 font-bold px-4 py-2 rounded-xl shadow-lg flex items-center justify-center gap-2 text-xs sm:text-sm animate-bounce">
                <Sparkles className="w-4 h-4 fill-slate-950" />
                Target Score Achieved! You beat the challenge benchmark!
              </div>
            )}

            {/* Episode Done Alert */}
            {session?.done && (
              <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center">
                <div className="text-xl font-bold text-white">Challenge Attempt Finished</div>
                <div className="text-3xl font-extrabold text-amber-400 mt-2">
                  {session.cumulative_reward.toFixed(1)} <span className="text-sm font-normal text-slate-400">Score</span>
                </div>
                <p className="text-xs text-slate-400 mt-1 max-w-xs">
                  {session.step_count} total steps taken.
                </p>
                <button
                  onClick={handleReset}
                  className="mt-4 px-5 py-2 rounded-xl text-xs font-bold bg-amber-500 hover:bg-amber-400 text-slate-950 transition cursor-pointer"
                >
                  Try Again
                </button>
              </div>
            )}
          </div>

          {/* Action Buttons */}
          {currentEnv && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-amber-400 flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5" />
                  Keyboard / Touch Action Pad
                </span>
                <span className="text-[11px] text-slate-400">{currentEnv.key_guide}</span>
              </div>

              <div className="flex flex-wrap gap-2">
                {currentEnv.action_labels.map((lbl, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleAction(idx)}
                    disabled={session?.done}
                    className="flex-1 min-w-[130px] py-3 px-3 bg-slate-950 hover:bg-slate-800 active:bg-amber-950 border border-slate-700 hover:border-amber-500/50 rounded-xl text-xs font-semibold text-white transition disabled:opacity-40"
                  >
                    {lbl}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Challenge Scorecard & Leaderboard (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Target Scorecard Card */}
          {currentPreset && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Challenge Scorecard</h3>
                <span className="text-xs font-semibold text-amber-400">{currentPreset.title}</span>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed">
                {currentPreset.description}
              </p>

              <div className="grid grid-cols-2 gap-2 text-center">
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <span className="text-[10px] text-slate-400 uppercase font-semibold">Your Score</span>
                  <div className="text-2xl font-extrabold text-amber-400 mt-0.5">
                    {session?.cumulative_reward !== undefined ? session.cumulative_reward.toFixed(1) : '0.0'}
                  </div>
                </div>
                <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
                  <span className="text-[10px] text-slate-400 uppercase font-semibold">Target Score</span>
                  <div className="text-2xl font-extrabold text-white mt-0.5">
                    {currentPreset.target_score}
                  </div>
                </div>
              </div>

              {/* Benchmarks Comparison Reference */}
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 space-y-1.5 text-xs">
                <div className="flex justify-between items-center text-emerald-400">
                  <span className="text-slate-400">Classical Benchmark:</span>
                  <span className="font-semibold">{currentPreset.agent_to_beat}</span>
                </div>
                <div className="flex justify-between items-center text-purple-400">
                  <span className="text-slate-400">Quantum Hybrid Score:</span>
                  <span className="font-semibold">{currentPreset.quantum_agent_score}</span>
                </div>
              </div>

              {/* Submit to Leaderboard Form */}
              <form onSubmit={handleSubmitScore} className="space-y-2 pt-2 border-t border-slate-800">
                <label className="text-[11px] text-slate-400 font-medium">Record this attempt to Leaderboard:</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={playerName}
                    onChange={(e) => setPlayerName(e.target.value)}
                    placeholder="Enter player name / reviewer"
                    className="flex-1 bg-slate-950 border border-slate-700 text-white text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-amber-500"
                  />
                  <button
                    type="submit"
                    className="flex items-center gap-1 px-3 py-2 rounded-xl text-xs font-semibold bg-amber-500 hover:bg-amber-400 text-slate-950 transition cursor-pointer"
                  >
                    <Send className="w-3.5 h-3.5" />
                    Record
                  </button>
                </div>
                {submissionMsg && (
                  <p className="text-[11px] text-emerald-400">{submissionMsg}</p>
                )}
              </form>
            </div>
          )}

          {/* Persistent Leaderboard Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Trophy className="w-4 h-4 text-amber-400" />
              Arena Leaderboard
            </h3>

            <div className="max-h-[220px] overflow-y-auto space-y-1.5">
              {leaderboard.length === 0 ? (
                <div className="text-xs text-slate-500 text-center py-4">
                  No challenge attempts recorded yet. Play an episode above and submit your score!
                </div>
              ) : (
                leaderboard.slice().reverse().map((entry, idx) => (
                  <div
                    key={idx}
                    className="flex justify-between items-center bg-slate-950 px-3 py-2 rounded-lg border border-slate-800 text-xs"
                  >
                    <div>
                      <span className="font-semibold text-white">{entry.player_name}</span>
                      <span className="text-[10px] text-slate-400 ml-2">({entry.env_id})</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-bold text-amber-400">{entry.score} pts</span>
                      {entry.success && (
                        <span className="text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 px-1.5 py-0.5 rounded">
                          Target Met
                        </span>
                      )}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
