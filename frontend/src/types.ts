export interface EnvironmentMeta {
  id: string;
  name: string;
  category: string;
  description: string;
  difficulty: string;
  difficulty_level: number;
  obs_dim: number;
  action_type: "discrete" | "continuous";
  action_dim: number;
  action_labels: string[];
  obs_labels: string[];
  obs_ranges: number[][];
  reward_metric: string;
  solved_threshold: number;
  max_steps: number;
  human_keys: Record<string, any>;
  key_guide: string;
  compatible_algorithms: string[];
}

export interface SimSession {
  session_id: string;
  env_id: string;
  algo: string;
  step_count: number;
  reward: number;
  cumulative_reward: number;
  done: boolean;
  state: number[];
  action?: string;
  termination_reason?: string;
  frame_base64: string;
  telemetry: {
    is_quantum?: boolean;
    expectation_values?: number[];
    projected_features?: number[];
    q_values?: number[];
    action_probabilities?: number[];
    continuous_mean?: number[];
  };
}

export interface QuantumSimResponse {
  features: number[];
  depth: number;
  expectation_values: {
    qubit: number;
    expval: number;
    pauli: string;
  }[];
  basis_probabilities: {
    basis: string;
    probability: number;
  }[];
  circuit_diagram: string;
}

export interface QuantumScalingData {
  qubits: number[];
  shots: number;
  depth: number;
  classical_memory_mb: number[];
  classical_sim_time_ms: number[];
  qpu_runtime_ms: number[];
  barren_plateau_variance: number[];
  crossover_qubit_estimate: number;
  insights: {
    simulation_bottleneck: string;
    memory_wall: string;
    barren_plateau_hazard: string;
  };
}

export interface TrainingStatus {
  is_running: boolean;
  algorithm: string | null;
  environment: string | null;
  current_step: number;
  total_steps: number;
  episodes: number;
  latest_reward: number;
  avg_reward_10: number;
  elapsed_time_s: number;
  is_quantum: boolean;
  fps: number;
  saved_checkpoint_path: string | null;
  error: string | null;
  history: {
    steps: number[];
    episode_rewards: number[];
  };
}

export interface ChallengePreset {
  title: string;
  description: string;
  target_score: number;
  max_score: number;
  agent_to_beat: string;
  quantum_agent_score: string;
}

export interface LeaderboardEntry {
  player_name: string;
  env_id: string;
  score: number;
  steps: number;
  success: boolean;
  timestamp: string;
}

export interface BatchDuelResult {
  env_id: string;
  algo_a: string;
  algo_b: string;
  n_episodes: number;
  results_a: {
    returns: number[];
    lengths: number[];
    mean_return: number;
    std_return: number;
    median_return: number;
    min_return: number;
    max_return: number;
    wins: number;
  };
  results_b: {
    returns: number[];
    lengths: number[];
    mean_return: number;
    std_return: number;
    median_return: number;
    min_return: number;
    max_return: number;
    wins: number;
  };
  ties: number;
}
