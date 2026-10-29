import {
  EnvironmentMeta,
  SimSession,
  QuantumSimResponse,
  QuantumScalingData,
  TrainingStatus,
  ChallengePreset,
  LeaderboardEntry,
  BatchDuelResult
} from './types';

const API_BASE = '/api';

export async function fetchEnvironments(): Promise<EnvironmentMeta[]> {
  const res = await fetch(`${API_BASE}/environments`);
  if (!res.ok) throw new Error('Failed to fetch environments');
  return res.json();
}

export async function createSession(env_id: string, algo: string, seed: number = 42): Promise<SimSession> {
  const res = await fetch(`${API_BASE}/sim/session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ env_id, algo, seed }),
  });
  if (!res.ok) throw new Error('Failed to create session');
  return res.json();
}

export async function stepSession(session_id: string, action?: any): Promise<SimSession> {
  const res = await fetch(`${API_BASE}/sim/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, action }),
  });
  if (!res.ok) throw new Error('Failed to step session');
  return res.json();
}

export async function resetSession(session_id: string, seed?: number): Promise<SimSession> {
  const res = await fetch(`${API_BASE}/sim/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, seed }),
  });
  if (!res.ok) throw new Error('Failed to reset session');
  return res.json();
}

export async function closeSession(session_id: string): Promise<void> {
  await fetch(`${API_BASE}/sim/close`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id }),
  });
}

export async function runBatchDuel(
  env_id: string,
  algo_a: string,
  algo_b: string,
  n_episodes: number = 15
): Promise<BatchDuelResult> {
  const res = await fetch(`${API_BASE}/sim/batch_duel`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ env_id, algo_a, algo_b, n_episodes }),
  });
  if (!res.ok) throw new Error('Failed to execute batch duel');
  return res.json();
}

export async function simulateQuantum(
  features: number[],
  depth: number = 2
): Promise<QuantumSimResponse> {
  const res = await fetch(`${API_BASE}/quantum/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features, depth }),
  });
  if (!res.ok) throw new Error('Failed to simulate quantum circuit');
  return res.json();
}

export async function fetchQuantumScaling(
  shots: number = 1000,
  depth: number = 2
): Promise<QuantumScalingData> {
  const res = await fetch(`${API_BASE}/quantum/scaling?shots=${shots}&depth=${depth}`);
  if (!res.ok) throw new Error('Failed to fetch quantum scaling data');
  return res.json();
}

export async function startTraining(
  algo: string,
  env_id: string,
  total_steps: number = 3000,
  learning_rate: number = 0.001,
  seed: number = 42
): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_BASE}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algo, env_id, total_steps, learning_rate, seed }),
  });
  return res.json();
}

export async function stopTraining(): Promise<{ success: boolean; message: string }> {
  const res = await fetch(`${API_BASE}/training/stop`, { method: 'POST' });
  return res.json();
}

export async function fetchTrainingStatus(): Promise<TrainingStatus> {
  const res = await fetch(`${API_BASE}/training/status`);
  if (!res.ok) throw new Error('Failed to fetch training status');
  return res.json();
}

export async function fetchBenchmarkRegistry(): Promise<any> {
  const res = await fetch(`${API_BASE}/benchmark/registry`);
  if (!res.ok) throw new Error('Failed to fetch benchmark registry');
  return res.json();
}

export async function fetchParetoFrontier(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/benchmark/pareto`);
  if (!res.ok) throw new Error('Failed to fetch Pareto data');
  return res.json();
}

export async function fetchAblations(): Promise<any> {
  const res = await fetch(`${API_BASE}/benchmark/ablations`);
  return res.json();
}

export async function fetchRobustness(): Promise<any> {
  const res = await fetch(`${API_BASE}/benchmark/robustness`);
  return res.json();
}

export async function fetchBaselines(): Promise<any> {
  const res = await fetch(`${API_BASE}/benchmark/baselines`);
  return res.json();
}

export async function fetchReportMarkdown(): Promise<string> {
  const res = await fetch(`${API_BASE}/benchmark/report`);
  return res.text();
}

export async function fetchChallenges(): Promise<{
  presets: Record<string, ChallengePreset>;
  leaderboard: LeaderboardEntry[];
}> {
  const res = await fetch(`${API_BASE}/challenges`);
  return res.json();
}

export async function recordChallenge(
  player_name: string,
  env_id: string,
  score: number,
  steps: number,
  completed: boolean
): Promise<any> {
  const res = await fetch(`${API_BASE}/challenges/record`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ player_name, env_id, score, steps, completed }),
  });
  return res.json();
}
