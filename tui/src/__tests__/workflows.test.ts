import { describe, expect, test } from "bun:test";

import type { Catalog } from "../backend.ts";
import {
  BEAMNG_DEFAULTS,
  MAIN_MENU_OPTIONS,
  buildTrainPayload,
  trainSavePath,
  type TrainState,
} from "../workflows.ts";

// The old interactive menu (core/cli.py main_menu) used these exact labels.
const OLD_MENU_LABELS = [
  "Train an agent",
  "Evaluate an agent",
  "Run a benchmark",
  "Human play (BeamNG)",
  "Generate trajectories (BeamNG)",
  "Multi-agent training (BeamNG)",
];

const EMPTY_CATALOG: Catalog = {
  algorithms: [],
  environments: [],
  compatible_envs: {},
  benchmarks: [],
  beamng_maps: [],
  beamng_vehicles: [],
  multi_algos: [],
};

describe("main menu", () => {
  test("labels match old CLI plus Quit", () => {
    expect(MAIN_MENU_OPTIONS.map((o) => o.label)).toEqual([...OLD_MENU_LABELS, "Quit"]);
  });
});

describe("train payload", () => {
  test("default save path for dqn + beamng_lidar", () => {
    expect(trainSavePath("dqn", "beamng_lidar")).toBe("outputs/dqn_beamng_lidar.pth");

    const state: TrainState = {
      algo_name: "dqn",
      env_name: "beamng_lidar",
      n_episodes: 500,
      agent_params: {},
      checkpoint_policy: "resume",
    };
    const payload = buildTrainPayload(EMPTY_CATALOG, state);
    expect(payload.save_path).toBe("outputs/dqn_beamng_lidar.pth");
    expect(payload.reset_existing).toBe(false);
  });

  test("beamng option defaults are gridmap_v2 and taxi", () => {
    expect(BEAMNG_DEFAULTS.map_name).toBe("gridmap_v2");
    expect(BEAMNG_DEFAULTS.vehicle_id).toBe("taxi");

    const state: TrainState = {
      algo_name: "dqn",
      env_name: "beamng_lidar",
      n_episodes: 500,
      agent_params: {},
      checkpoint_policy: "resume",
    };
    const payload = buildTrainPayload(EMPTY_CATALOG, state) as { beamng: Record<string, unknown> };
    expect(payload.beamng.map_name).toBe("gridmap_v2");
    expect(payload.beamng.vehicle_id).toBe("taxi");
    expect(payload.beamng.random_path).toBe(false);
  });

  test("non-beamng env omits beamng block", () => {
    const state: TrainState = {
      algo_name: "q_learning",
      env_name: "taxi",
      n_episodes: 500,
      agent_params: {},
      checkpoint_policy: "reset",
    };
    const payload = buildTrainPayload(EMPTY_CATALOG, state);
    expect(payload.beamng).toBeUndefined();
    expect(payload.reset_existing).toBe(true);
  });
});
