// Pure state -> backend-payload builders. No OpenTUI renderer is touched here,
// so the JSON contract with `core.tui_backend` is covered by plain unit tests.

import type { Catalog } from "./backend.ts";

export type WorkflowId =
  | "train"
  | "evaluate"
  | "benchmark"
  | "human_play"
  | "trajectory"
  | "multi_train"
  | "quit";

export const MAIN_MENU_OPTIONS: { id: WorkflowId; label: string }[] = [
  { id: "train", label: "Train an agent" },
  { id: "evaluate", label: "Evaluate an agent" },
  { id: "benchmark", label: "Run a benchmark" },
  { id: "human_play", label: "Human play (BeamNG)" },
  { id: "trajectory", label: "Generate trajectories (BeamNG)" },
  { id: "multi_train", label: "Multi-agent training (BeamNG)" },
  { id: "quit", label: "Quit" },
];

export const BEAMNG_MAPS = ["gridmap_v2", "italy", "west_coast_usa"] as const;

export const MULTI_COLORS = [
  "Yellow",
  "Red",
  "Blue",
  "Green",
  "Orange",
  "White",
  "Black",
] as const;

export interface BeamNGFields {
  map_name: string;
  vehicle_id: string;
  trajectory_hints: number;
  body_orientation: boolean;
  wheel_terrain: boolean;
  random_path?: boolean;
  dense_episodes?: number;
}

export const BEAMNG_DEFAULTS: BeamNGFields = {
  map_name: "gridmap_v2",
  vehicle_id: "taxi",
  trajectory_hints: 0,
  body_orientation: false,
  wheel_terrain: false,
};

export interface TrainState {
  algo_name: string;
  env_name: string;
  n_episodes: number;
  save_path?: string;
  agent_params: Record<string, number>;
  beamng?: BeamNGFields;
  checkpoint_policy: "resume" | "reset";
}

export interface EvaluateState {
  algo_name: string;
  env_name: string;
  model_path?: string;
  n_episodes: number;
  beamng?: BeamNGFields;
}

export interface BenchmarkState {
  benchmark_name: string;
  seeds: number[];
  eval_episodes: number;
  success_threshold: number;
  max_episodes: number;
  reward_threshold?: number;
  algo_name?: string;
  env_name?: string;
  algos?: string[];
  param_grid?: Record<string, unknown[]>;
}

export interface HumanPlayState {
  map_name: string;
  vehicle_id: string;
  sensor: string;
  random_path: boolean;
}

export interface TrajectoryState {
  map_name: string;
  overwrite: boolean;
}

export interface MultiSpecState {
  algo: string;
  env: string;
  vehicle_id: string;
  color: string;
  save_path: string;
  trajectory_hints: number;
  body_orientation: boolean;
  wheel_terrain: boolean;
}

export interface MultiTrainState {
  map_name: string;
  random_path: boolean;
  n_episodes: number;
  time_limit_minutes: number;
  specs: MultiSpecState[];
  checkpoint_policy: "resume" | "reset";
}

// Encode the beamng options that change what a checkpoint represents into the
// file name, so different configs don't overwrite each other: "_h<n>" for
// checkpoint hints (>0) and "_ori" when body orientation is on.
export function beamngPathSuffix(beamng?: { trajectory_hints: number; body_orientation: boolean }): string {
  if (!beamng) return "";
  let suffix = "";
  if (beamng.trajectory_hints > 0) suffix += `_h${beamng.trajectory_hints}`;
  if (beamng.body_orientation) suffix += "_ori";
  return suffix;
}

export function trainSavePath(
  algoName: string,
  envName: string,
  beamng?: { trajectory_hints: number; body_orientation: boolean },
): string {
  return `outputs/${algoName}_${envName}${beamngPathSuffix(beamng)}.pth`;
}

export function buildTrainPayload(_catalog: Catalog, state: TrainState): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    algo_name: state.algo_name,
    env_name: state.env_name,
    n_episodes: state.n_episodes,
    save_path: state.save_path ?? trainSavePath(state.algo_name, state.env_name, state.beamng),
    agent_params: state.agent_params,
    reset_existing: state.checkpoint_policy === "reset",
  };
  if (state.env_name.startsWith("beamng")) {
    payload.beamng = { ...BEAMNG_DEFAULTS, random_path: false, ...state.beamng };
  }
  return payload;
}

export function buildEvaluatePayload(
  _catalog: Catalog,
  state: EvaluateState,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    algo_name: state.algo_name,
    env_name: state.env_name,
    model_path: state.model_path ?? trainSavePath(state.algo_name, state.env_name, state.beamng),
    n_episodes: state.n_episodes,
  };
  if (state.env_name.startsWith("beamng")) {
    // Evaluation has no random_path field.
    const { map_name, vehicle_id, trajectory_hints, body_orientation, wheel_terrain } = {
      ...BEAMNG_DEFAULTS,
      ...state.beamng,
    };
    payload.beamng = { map_name, vehicle_id, trajectory_hints, body_orientation, wheel_terrain };
  }
  return payload;
}

export function buildBenchmarkPayload(
  _catalog: Catalog,
  state: BenchmarkState,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    benchmark_name: state.benchmark_name,
    seeds: state.seeds,
    eval_episodes: state.eval_episodes,
    success_threshold: state.success_threshold,
    max_episodes: state.max_episodes,
  };
  if (state.benchmark_name === "comparison") {
    payload.algos = state.algos;
    payload.env_name = state.env_name;
  } else if (state.benchmark_name === "gridsearch") {
    payload.algo_name = state.algo_name;
    payload.env_name = state.env_name;
    payload.param_grid = state.param_grid;
  } else {
    payload.algo_name = state.algo_name;
    payload.env_name = state.env_name;
    payload.reward_threshold = state.reward_threshold ?? 7.0;
  }
  return payload;
}

export function buildHumanPlayPayload(state: HumanPlayState): Record<string, unknown> {
  return {
    map_name: state.map_name,
    vehicle_id: state.vehicle_id,
    sensor: state.sensor,
    random_path: state.random_path,
  };
}

export function buildTrajectoryPayload(state: TrajectoryState): Record<string, unknown> {
  return { map_name: state.map_name, overwrite: state.overwrite };
}

export function buildMultiTrainPayload(
  _catalog: Catalog,
  state: MultiTrainState,
): Record<string, unknown> {
  return {
    map_name: state.map_name,
    random_path: state.random_path,
    n_episodes: state.n_episodes,
    time_limit_minutes: state.time_limit_minutes,
    reset_existing: state.checkpoint_policy === "reset",
    specs: state.specs,
  };
}
