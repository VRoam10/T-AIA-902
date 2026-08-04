// Pure state -> backend-payload builders. No OpenTUI renderer is touched here,
// so the JSON contract with `core.tui_backend` is covered by plain unit tests.

import type { Catalog } from "./backend.ts";

export type WorkflowId = "train" | "multi_train" | "human_play" | "course" | "trajectory" | "quit";

export const MAIN_MENU_OPTIONS: { id: WorkflowId; label: string }[] = [
  { id: "train", label: "Train an agent" },
  { id: "multi_train", label: "Multi-agent training" },
  { id: "human_play", label: "Human play" },
  { id: "course", label: "Course mode (race)" },
  { id: "trajectory", label: "Generate trajectories" },
  { id: "quit", label: "Quit" },
];

export const BEAMNG_MAPS = ["gridmap_v2", "italy", "west_coast_usa", "east_coast_usa"] as const;

// The perception axis. The output axis is NOT offered: the algorithm determines it
// (dqn/dqn_per drive the discrete action table, ddpg/td3 drive continuous controls),
// so a second field could only ever disagree with the first.
export const BEAMNG_SENSORS = ["lidar", "adv_lidar", "camera"] as const;
export type BeamNGSensor = (typeof BEAMNG_SENSORS)[number];

export const MULTI_COLORS = [
  "Yellow",
  "Red",
  "Blue",
  "Green",
  "Orange",
  "White",
  "Black",
] as const;

// Two entrants, so two distinct liveries. One car model for everyone — only the
// paint differs, so a race result reflects the policies and not the machinery.
export const RACE_COLORS = ["Red", "Blue"] as const;

export type CourseOpponent = "algo" | "human";

// What a run drives: the road-network paths built by "Generate trajectories", or
// one of the game's own race tracks. The two track kinds are the game's own
// distinction — a sprint runs point to point, a lap returns to its start.
export const TRACK_KINDS = ["generated", "sprint", "lap"] as const;
export type TrackKind = (typeof TRACK_KINDS)[number];

export interface BeamNGFields {
  map_name: string;
  sensor: string;
  trajectory_hints: number;
  body_orientation: boolean;
  road_info: boolean;
  wheel_info: boolean;
  random_path?: boolean;
  dense_episodes?: number;
  // A game-track key, or "" for the generated paths.
  track?: string;
}

export const BEAMNG_DEFAULTS: BeamNGFields = {
  map_name: "gridmap_v2",
  sensor: "lidar",
  trajectory_hints: 0,
  body_orientation: false,
  road_info: false,
  wheel_info: false,
  track: "",
};

/** Game-track keys for a map, filtered to one kind, longest first. */
export function tracksFor(catalog: Catalog, mapName: string, kind: TrackKind): string[] {
  if (kind === "generated") return [];
  return (catalog.beamng_tracks?.[mapName] ?? [])
    .filter((t) => t.kind === kind)
    .map((t) => t.key);
}

/** The track a form's two fields resolve to: "" for generated paths. */
export function resolveTrack(
  catalog: Catalog,
  mapName: string,
  kind: string,
  selected: string,
): string {
  if (kind === "generated") return "";
  const keys = tracksFor(catalog, mapName, kind as TrackKind);
  // Guard against a stale selection: the map may have changed under the field.
  return keys.includes(selected) ? selected : (keys[0] ?? "");
}

export interface TrainState {
  algo_name: string;
  n_episodes: number;
  save_path?: string;
  agent_params: Record<string, number>;
  beamng?: BeamNGFields;
  checkpoint_policy: "resume" | "reset";
}

export interface HumanPlayState {
  map_name: string;
  sensor: string;
  random_path: boolean;
  road_info: boolean;
  wheel_info: boolean;
  track?: string;
}

// Trajectory generation is a per-map probe, so the only inputs are which map (or
// "all", expanded by the form) and whether an existing cache is replaced.
export interface TrajectoryState {
  map_name: string;
  overwrite: boolean;
}

export interface MultiSpecState {
  algo: string;
  sensor: string;
  color: string;
  save_path: string;
  trajectory_hints: number;
  body_orientation: boolean;
  road_info: boolean;
  wheel_info: boolean;
}

export interface MultiTrainState {
  map_name: string;
  random_path: boolean;
  n_episodes: number;
  time_limit_minutes: number;
  specs: MultiSpecState[];
  checkpoint_policy: "resume" | "reset";
  track?: string;
}

export interface RacerState {
  algo: string;
  sensor: string;
  model_path: string;
  color: string;
  trajectory_hints: number;
  body_orientation: boolean;
  road_info: boolean;
  wheel_info: boolean;
  human?: boolean;
}

export interface CourseState {
  map_name: string;
  opponent: CourseOpponent;
  laps: number;
  races: number;
  learning: boolean;
  racers: RacerState[];
  track?: string;
}

// Encode the beamng options that change what a checkpoint represents into the file
// name, so different configs cannot overwrite each other: "_h<n>" for checkpoint
// hints (>0), "_ori" for body orientation, "_road" for road position, "_whl" for
// wheel performance. The order is fixed so a path is reproducible.
export function beamngPathSuffix(beamng?: {
  trajectory_hints: number;
  body_orientation: boolean;
  road_info?: boolean;
  wheel_info?: boolean;
}): string {
  if (!beamng) return "";
  let suffix = "";
  if (beamng.trajectory_hints > 0) suffix += `_h${beamng.trajectory_hints}`;
  if (beamng.body_orientation) suffix += "_ori";
  if (beamng.road_info) suffix += "_road";
  if (beamng.wheel_info) suffix += "_whl";
  return suffix;
}

// The sensor replaces the env name in checkpoint paths: it is what determines the
// observation width, so two sensors must never share a file.
export function trainSavePath(
  algoName: string,
  sensor: string,
  beamng?: {
    trajectory_hints: number;
    body_orientation: boolean;
    road_info?: boolean;
    wheel_info?: boolean;
  },
): string {
  return `outputs/${algoName}_${sensor}${beamngPathSuffix(beamng)}.pth`;
}

export function buildTrainPayload(_catalog: Catalog, state: TrainState): Record<string, unknown> {
  const beamng = { ...BEAMNG_DEFAULTS, random_path: false, ...state.beamng };
  return {
    algo_name: state.algo_name,
    // One registered environment now; the sensor/output axes live in the options.
    env_name: "beamng",
    n_episodes: state.n_episodes,
    save_path: state.save_path ?? trainSavePath(state.algo_name, beamng.sensor, beamng),
    agent_params: state.agent_params,
    reset_existing: state.checkpoint_policy === "reset",
    beamng,
  };
}

export function buildHumanPlayPayload(state: HumanPlayState): Record<string, unknown> {
  return {
    map_name: state.map_name,
    sensor: state.sensor,
    random_path: state.random_path,
    track: state.track ?? "",
    road_info: state.road_info,
    wheel_info: state.wheel_info,
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
    track: state.track ?? "",
  };
}

export function buildCoursePayload(state: CourseState): Record<string, unknown> {
  // A human opponent replaces racer 2 entirely: the player needs no algorithm,
  // checkpoint or sensor, so only the livery carries over.
  const racers: Record<string, unknown>[] = state.racers.map((r, i) => {
    if (state.opponent === "human" && i === 1) {
      return { human: true, color: r.color };
    }
    return {
      algo: r.algo,
      sensor: r.sensor,
      model_path: r.model_path || trainSavePath(r.algo, r.sensor, r),
      color: r.color,
      trajectory_hints: r.trajectory_hints,
      body_orientation: r.body_orientation,
      road_info: r.road_info,
      wheel_info: r.wheel_info,
    };
  });

  return {
    map_name: state.map_name,
    laps: state.laps,
    races: state.races,
    learning: state.learning,
    racers,
    track: state.track ?? "",
  };
}
