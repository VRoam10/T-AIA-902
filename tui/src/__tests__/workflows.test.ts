import { describe, expect, test } from "bun:test";

import type { Catalog } from "../backend.ts";
import {
  BEAMNG_DEFAULTS,
  BEAMNG_SENSORS,
  MAIN_MENU_OPTIONS,
  RACE_COLORS,
  beamngPathSuffix,
  buildCoursePayload,
  buildHumanPlayPayload,
  buildMultiTrainPayload,
  buildTrainPayload,
  buildTrajectoryPayload,
  resolveTrack,
  trainSavePath,
  tracksFor,
  type CourseState,
  type MultiTrainState,
  type RacerState,
  type TrainState,
} from "../workflows.ts";

// The four modes this branch exists to expose, plus trajectory generation (the
// caches every BeamNG run reads have to be producible from the menu), plus Quit.
const MENU_LABELS = [
  "Train an agent",
  "Multi-agent training",
  "Human play",
  "Course mode (race)",
  "Generate trajectories",
];

const EMPTY_CATALOG: Catalog = {
  algorithms: [],
  environments: [],
  compatible_envs: {},
  benchmarks: [],
  beamng_maps: [],
  beamng_sensors: [],
  multi_algos: [],
};

describe("main menu", () => {
  test("exposes exactly the five modes plus Quit", () => {
    expect(MAIN_MENU_OPTIONS.map((o) => o.label)).toEqual([...MENU_LABELS, "Quit"]);
  });

  test("the dropped modes are gone", () => {
    const ids = MAIN_MENU_OPTIONS.map((o) => o.id as string);
    for (const gone of ["evaluate", "benchmark"]) {
      expect(ids).not.toContain(gone);
    }
  });

  test("trajectory generation is reachable from the menu", () => {
    expect(MAIN_MENU_OPTIONS.some((o) => o.id === "trajectory")).toBe(true);
  });
});

describe("trajectory payload", () => {
  test("carries the map and the overwrite flag", () => {
    expect(buildTrajectoryPayload({ map_name: "italy", overwrite: true })).toEqual({
      map_name: "italy",
      overwrite: true,
    });
  });
});

describe("checkpoint paths encode the config", () => {
  test("no suffix for defaults; _h<n> for hints; _ori for body orientation", () => {
    expect(beamngPathSuffix()).toBe("");
    expect(beamngPathSuffix({ trajectory_hints: 0, body_orientation: false })).toBe("");
    expect(beamngPathSuffix({ trajectory_hints: 2, body_orientation: false })).toBe("_h2");
    expect(beamngPathSuffix({ trajectory_hints: 0, body_orientation: true })).toBe("_ori");
    expect(beamngPathSuffix({ trajectory_hints: 3, body_orientation: true })).toBe("_h3_ori");
  });

  test("the sensor is part of the path, since it sets the observation width", () => {
    expect(trainSavePath("dqn", "lidar")).toBe("outputs/dqn_lidar.pth");
    expect(trainSavePath("dqn", "adv_lidar")).toBe("outputs/dqn_adv_lidar.pth");
    expect(trainSavePath("td3", "camera")).toBe("outputs/td3_camera.pth");
  });

  test("two sensors never share a checkpoint file", () => {
    const paths = BEAMNG_SENSORS.map((s) => trainSavePath("dqn", s));
    expect(new Set(paths).size).toBe(BEAMNG_SENSORS.length);
  });

  test("options are appended after the sensor", () => {
    expect(trainSavePath("dqn", "adv_lidar", { trajectory_hints: 2, body_orientation: true })).toBe(
      "outputs/dqn_adv_lidar_h2_ori.pth",
    );
  });
});

describe("train payload", () => {
  const state: TrainState = {
    algo_name: "dqn",
    n_episodes: 500,
    agent_params: {},
    checkpoint_policy: "resume",
  };

  test("always targets the single beamng env", () => {
    expect(buildTrainPayload(EMPTY_CATALOG, state).env_name).toBe("beamng");
  });

  test("derives the save path from algo + sensor", () => {
    const payload = buildTrainPayload(EMPTY_CATALOG, state);
    expect(payload.save_path).toBe("outputs/dqn_lidar.pth");
    expect(payload.reset_existing).toBe(false);
  });

  test("a chosen sensor flows into both the block and the path", () => {
    const payload = buildTrainPayload(EMPTY_CATALOG, {
      ...state,
      beamng: { ...BEAMNG_DEFAULTS, sensor: "camera" },
    }) as { beamng: Record<string, unknown>; save_path: string };
    expect(payload.beamng.sensor).toBe("camera");
    expect(payload.save_path).toBe("outputs/dqn_camera.pth");
  });

  test("defaults are gridmap_v2 + lidar, and random_path off", () => {
    expect(BEAMNG_DEFAULTS.map_name).toBe("gridmap_v2");
    expect(BEAMNG_DEFAULTS.sensor).toBe("lidar");
    const payload = buildTrainPayload(EMPTY_CATALOG, state) as { beamng: Record<string, unknown> };
    expect(payload.beamng.map_name).toBe("gridmap_v2");
    expect(payload.beamng.random_path).toBe(false);
  });

  test("the beamng block is always present — there is no non-beamng env", () => {
    expect(buildTrainPayload(EMPTY_CATALOG, state).beamng).toBeDefined();
  });

  test("no vehicle is sent: one car for everyone", () => {
    const payload = buildTrainPayload(EMPTY_CATALOG, state) as { beamng: Record<string, unknown> };
    expect("vehicle_id" in payload.beamng).toBe(false);
  });

  test("checkpoint_policy maps to reset_existing", () => {
    expect(
      buildTrainPayload(EMPTY_CATALOG, { ...state, checkpoint_policy: "reset" }).reset_existing,
    ).toBe(true);
  });

  test("agent_params are forwarded", () => {
    expect(
      buildTrainPayload(EMPTY_CATALOG, { ...state, agent_params: { lr: 0.001, gamma: 0.99 } })
        .agent_params,
    ).toEqual({ lr: 0.001, gamma: 0.99 });
  });
});

describe("human-play payload", () => {
  test("passes map / sensor / random_path through", () => {
    expect(
      buildHumanPlayPayload({ map_name: "italy", sensor: "adv_lidar", random_path: true }),
    ).toEqual({ map_name: "italy", sensor: "adv_lidar", random_path: true, track: "" });
  });

  test("an unset track means the generated paths", () => {
    const payload = buildHumanPlayPayload({
      map_name: "italy",
      sensor: "lidar",
      random_path: false,
    });
    expect(payload.track).toBe("");
  });

  test("a chosen game track is forwarded", () => {
    const payload = buildHumanPlayPayload({
      map_name: "italy",
      sensor: "lidar",
      random_path: false,
      track: "mixedCircuit1",
    });
    expect(payload.track).toBe("mixedCircuit1");
  });
});

describe("game track selection", () => {
  const catalog: Catalog = {
    ...EMPTY_CATALOG,
    beamng_tracks: {
      italy: [
        { key: "mixedCircuit1", kind: "lap", checkpoints: 12, length_m: 3831 },
        { key: "cliffRoad1", kind: "sprint", checkpoints: 13, length_m: 6329 },
      ],
      gridmap_v2: [],
    },
  };

  test("tracks are filtered by kind, and generated selects none", () => {
    expect(tracksFor(catalog, "italy", "lap")).toEqual(["mixedCircuit1"]);
    expect(tracksFor(catalog, "italy", "sprint")).toEqual(["cliffRoad1"]);
    expect(tracksFor(catalog, "italy", "generated")).toEqual([]);
  });

  test("a map with no tracks, or a backend without the list, yields none", () => {
    expect(tracksFor(catalog, "gridmap_v2", "lap")).toEqual([]);
    expect(tracksFor(EMPTY_CATALOG, "italy", "lap")).toEqual([]);
  });

  test("resolveTrack returns the empty string for generated paths", () => {
    expect(resolveTrack(catalog, "italy", "generated", "mixedCircuit1")).toBe("");
  });

  test("resolveTrack keeps a valid selection", () => {
    expect(resolveTrack(catalog, "italy", "lap", "mixedCircuit1")).toBe("mixedCircuit1");
  });

  test("resolveTrack falls back when the selection is stale for the map", () => {
    // The map changed under the field: italy's circuit is not on gridmap_v2, and
    // sending it would fail deep in the env instead of racing something valid.
    expect(resolveTrack(catalog, "italy", "sprint", "mixedCircuit1")).toBe("cliffRoad1");
    expect(resolveTrack(catalog, "gridmap_v2", "lap", "mixedCircuit1")).toBe("");
  });
});

describe("multi-train payload", () => {
  test("maps checkpoint_policy to reset_existing and forwards specs", () => {
    const spec = {
      algo: "dqn",
      sensor: "lidar",
      color: "Yellow",
      save_path: "outputs/multi-agents/dqn_lidar_0.pth",
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

describe("course payload", () => {
  const racer = (overrides: Partial<RacerState> = {}): RacerState => ({
    algo: "dqn",
    sensor: "lidar",
    model_path: "outputs/dqn_lidar.pth",
    color: "Red",
    trajectory_hints: 0,
    body_orientation: false,
    ...overrides,
  });

  const base: CourseState = {
    map_name: "gridmap_v2",
    opponent: "algo",
    laps: 1,
    races: 1,
    learning: false,
    racers: [racer(), racer({ algo: "td3", sensor: "adv_lidar", color: "Blue" })],
  };

  test("sends both entrants for an algo-vs-algo race", () => {
    const p = buildCoursePayload(base) as { racers: Record<string, unknown>[] };
    expect(p.racers).toHaveLength(2);
    expect(p.racers[0].algo).toBe("dqn");
    expect(p.racers[1].algo).toBe("td3");
  });

  test("each entrant carries its own sensor", () => {
    const p = buildCoursePayload(base) as { racers: Record<string, unknown>[] };
    expect(p.racers[0].sensor).toBe("lidar");
    expect(p.racers[1].sensor).toBe("adv_lidar");
  });

  test("a human opponent replaces racer 2 entirely", () => {
    const p = buildCoursePayload({ ...base, opponent: "human" }) as {
      racers: Record<string, unknown>[];
    };
    expect(p.racers[1].human).toBe(true);
    // No algorithm, checkpoint or sensor to configure — the player drives.
    expect(p.racers[1].algo).toBeUndefined();
    expect(p.racers[1].model_path).toBeUndefined();
    expect(p.racers[1].sensor).toBeUndefined();
  });

  test("racer 1 is unaffected by a human opponent", () => {
    const p = buildCoursePayload({ ...base, opponent: "human" }) as {
      racers: Record<string, unknown>[];
    };
    expect(p.racers[0].algo).toBe("dqn");
    expect(p.racers[0].human).toBeUndefined();
  });

  test("an empty checkpoint falls back to the derived path", () => {
    const p = buildCoursePayload({
      ...base,
      racers: [racer({ model_path: "" }), racer({ algo: "td3", sensor: "camera", model_path: "" })],
    }) as { racers: Record<string, unknown>[] };
    expect(p.racers[0].model_path).toBe("outputs/dqn_lidar.pth");
    expect(p.racers[1].model_path).toBe("outputs/td3_camera.pth");
  });

  test("an explicit checkpoint is preserved", () => {
    const p = buildCoursePayload({
      ...base,
      racers: [racer({ model_path: "outputs/custom.pth" }), racer()],
    }) as { racers: Record<string, unknown>[] };
    expect(p.racers[0].model_path).toBe("outputs/custom.pth");
  });

  test("laps, races and learning are forwarded", () => {
    const p = buildCoursePayload({ ...base, races: 5, learning: true });
    expect(p.laps).toBe(1);
    expect(p.races).toBe(5);
    expect(p.learning).toBe(true);
  });

  test("the two liveries are distinct so the cars are tellable apart", () => {
    expect(RACE_COLORS[0]).not.toBe(RACE_COLORS[1]);
  });
});
