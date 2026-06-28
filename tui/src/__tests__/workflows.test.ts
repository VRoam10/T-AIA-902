import { describe, expect, test } from "bun:test";

import type { Catalog } from "../backend.ts";
import {
  BEAMNG_DEFAULTS,
  MAIN_MENU_OPTIONS,
  buildBenchmarkPayload,
  buildEvaluatePayload,
  buildHumanPlayPayload,
  buildMultiTrainPayload,
  buildTrainPayload,
  buildTrajectoryPayload,
  trainSavePath,
  type BenchmarkState,
  type EvaluateState,
  type MultiTrainState,
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

  test("agent_params are forwarded", () => {
    const state: TrainState = {
      algo_name: "dqn",
      env_name: "taxi",
      n_episodes: 100,
      agent_params: { lr: 0.001, gamma: 0.99 },
      checkpoint_policy: "resume",
    };
    expect(buildTrainPayload(EMPTY_CATALOG, state).agent_params).toEqual({ lr: 0.001, gamma: 0.99 });
  });
});

describe("evaluate payload", () => {
  test("defaults model_path and omits beamng for taxi", () => {
    const state: EvaluateState = { algo_name: "dqn", env_name: "taxi", n_episodes: 10 };
    const payload = buildEvaluatePayload(EMPTY_CATALOG, state);
    expect(payload.model_path).toBe("outputs/dqn_taxi.pth");
    expect(payload.n_episodes).toBe(10);
    expect(payload.beamng).toBeUndefined();
  });

  test("beamng env includes a beamng block WITHOUT random_path", () => {
    const state: EvaluateState = {
      algo_name: "dqn",
      env_name: "beamng_lidar",
      n_episodes: 5,
      beamng: { ...BEAMNG_DEFAULTS, map_name: "italy", vehicle_id: "ibishu_pigeon" },
    };
    const payload = buildEvaluatePayload(EMPTY_CATALOG, state) as {
      beamng: Record<string, unknown>;
    };
    expect(payload.beamng.map_name).toBe("italy");
    expect(payload.beamng.vehicle_id).toBe("ibishu_pigeon");
    expect("random_path" in payload.beamng).toBe(false);
  });
});

describe("benchmark payload", () => {
  const base: BenchmarkState = {
    benchmark_name: "convergence",
    seeds: [0, 1, 2],
    eval_episodes: 100,
    success_threshold: 0,
    max_episodes: 2000,
    reward_threshold: 7,
    algo_name: "dqn",
    env_name: "taxi",
    algos: ["dqn", "q_learning"],
    param_grid: { lr: [0.1, 0.5] },
  };

  test("convergence: single algo + env + reward_threshold, no algos/grid", () => {
    const p = buildBenchmarkPayload(EMPTY_CATALOG, base);
    expect(p.algo_name).toBe("dqn");
    expect(p.env_name).toBe("taxi");
    expect(p.reward_threshold).toBe(7);
    expect(p.algos).toBeUndefined();
    expect(p.param_grid).toBeUndefined();
    expect(p.seeds).toEqual([0, 1, 2]);
  });

  test("comparison: sends algos + env, no algo_name/reward_threshold", () => {
    const p = buildBenchmarkPayload(EMPTY_CATALOG, { ...base, benchmark_name: "comparison" });
    expect(p.algos).toEqual(["dqn", "q_learning"]);
    expect(p.env_name).toBe("taxi");
    expect(p.algo_name).toBeUndefined();
    expect(p.reward_threshold).toBeUndefined();
  });

  test("gridsearch: sends algo_name + env + param_grid, no reward_threshold", () => {
    const p = buildBenchmarkPayload(EMPTY_CATALOG, { ...base, benchmark_name: "gridsearch" });
    expect(p.algo_name).toBe("dqn");
    expect(p.param_grid).toEqual({ lr: [0.1, 0.5] });
    expect(p.reward_threshold).toBeUndefined();
    expect(p.algos).toBeUndefined();
  });

  test("reward_threshold defaults to 7.0 when absent (convergence)", () => {
    const { reward_threshold, ...noThreshold } = base;
    void reward_threshold;
    expect(buildBenchmarkPayload(EMPTY_CATALOG, noThreshold as BenchmarkState).reward_threshold).toBe(7.0);
  });
});

describe("human-play payload", () => {
  test("passes map / vehicle / sensor / random_path through", () => {
    expect(
      buildHumanPlayPayload({
        map_name: "italy",
        vehicle_id: "taxi",
        sensor: "LiDAR",
        random_path: true,
      }),
    ).toEqual({ map_name: "italy", vehicle_id: "taxi", sensor: "LiDAR", random_path: true });
  });
});

describe("trajectory payload", () => {
  test("passes map_name and overwrite through", () => {
    expect(buildTrajectoryPayload({ map_name: "gridmap_v2", overwrite: true })).toEqual({
      map_name: "gridmap_v2",
      overwrite: true,
    });
  });
});

describe("multi-train payload", () => {
  test("maps checkpoint_policy to reset_existing and forwards specs", () => {
    const spec = {
      algo: "dqn",
      env: "beamng",
      vehicle_id: "taxi",
      color: "Yellow",
      save_path: "outputs/multi-agents/dqn_beamng_0.pth",
      trajectory_hints: 0,
      body_orientation: false,
      wheel_terrain: false,
    };
    const state: MultiTrainState = {
      map_name: "gridmap_v2",
      random_path: true,
      n_episodes: 500,
      time_limit_minutes: 0,
      checkpoint_policy: "reset",
      specs: [spec],
    };
    const p = buildMultiTrainPayload(EMPTY_CATALOG, state);
    expect(p.reset_existing).toBe(true);
    expect(p.random_path).toBe(true);
    expect(p.specs).toEqual([spec]);

    const resume = buildMultiTrainPayload(EMPTY_CATALOG, { ...state, checkpoint_policy: "resume" });
    expect(resume.reset_existing).toBe(false);
  });
});
