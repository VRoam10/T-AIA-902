// Integration coverage for the interactive form layer (controller + form +
// forms), driven through a real OpenTUI test renderer. This exercises the paths
// the pure unit tests can't: building a workflow form, the algorithm-switch
// rebuild (resolveAlgoEnv), and returning to the menu.

import { beforeAll, describe, expect, test } from "bun:test";
import { createTestRenderer } from "@opentui/core/testing";

import type { Catalog } from "../backend.ts";
import { createState, type Ctx } from "../context.ts";
import { buildScene } from "../scene.ts";
import { backToMenu, onChoiceChanged, openWorkflow, rebuildActiveForm } from "../controller.ts";
import { cycleChoice, readValues } from "../form.ts";
import { moveMenu, selectedWorkflow } from "../menu.ts";
import { appendLog } from "../status.ts";
import { LOG_PREVIEW_LIMIT } from "../widgets.ts";
import { BEAMNG_MAPS, MAIN_MENU_OPTIONS, resolveTrack } from "../workflows.ts";
import { makeWelcomeSyntax } from "../welcome.ts";

const catalog: Catalog = {
  algorithms: [
    { name: "dqn", default_config: { lr: 0.001, gamma: 0.99, batch_size: 64 }, compatible_envs: ["beamng"] },
    { name: "td3", default_config: { actor_lr: 0.0003, tau: 0.005 }, compatible_envs: ["beamng"] },
  ],
  environments: [{ name: "beamng", metadata: {} }],
  compatible_envs: { dqn: ["beamng"], td3: ["beamng"] },
  benchmarks: ["convergence", "comparison", "gridsearch"],
  beamng_maps: ["gridmap_v2", "italy"],
  beamng_sensors: ["lidar", "adv_lidar", "camera"],
  multi_algos: ["dqn"],
  beamng_tracks: {
    gridmap_v2: [
      { key: "highspeed_ring", kind: "lap", checkpoints: 11, length_m: 7338 },
      { key: "halfpipes_short", kind: "sprint", checkpoints: 1, length_m: 720 },
    ],
    italy: [{ key: "mixedCircuit1", kind: "lap", checkpoints: 12, length_m: 3831 }],
  },
};

let ctx: Ctx;
const keys = () => ctx.state.fields.map((f) => f.key);

beforeAll(async () => {
  const setup = await createTestRenderer({ width: 110, height: 40 });
  const scene = buildScene(setup.renderer, catalog);
  ctx = {
    renderer: setup.renderer,
    catalog,
    scene,
    state: createState(),
    algoNames: catalog.algorithms.map((a) => a.name),
    benchNames: catalog.benchmarks,
    sensors: catalog.beamng_sensors,
    readme: "# RL Pipeline\n",
    welcomeSyntax: makeWelcomeSyntax(),
    onChoiceChanged: (f) => onChoiceChanged(ctx, f),
    rebuildActiveForm: (k) => rebuildActiveForm(ctx, k),
  };
});

describe("train form", () => {
  test("offers a sensor but no environment: the axes replaced the env name", () => {
    openWorkflow(ctx, "train");
    expect(keys()).toContain("algo_name");
    expect(keys()).toContain("sensor");
    expect(keys()).not.toContain("env_name");
  });

  test("offers no output field: the algorithm determines the action head", () => {
    openWorkflow(ctx, "train");
    expect(keys()).not.toContain("output");
  });

  test("offers no vehicle field: one car for everyone", () => {
    openWorkflow(ctx, "train");
    expect(keys()).not.toContain("vehicle_id");
  });

  test("builds the first algorithm's hyperparameters by default", () => {
    openWorkflow(ctx, "train");
    expect(keys()).toContain("param:gamma"); // dqn's
    expect(keys()).not.toContain("param:tau"); // td3's, must be absent
  });

  test("switching algorithm rebuilds with the new algo's params", () => {
    openWorkflow(ctx, "train");
    ctx.state.focusIndex = ctx.state.fields.findIndex((f) => f.key === "algo_name");
    cycleChoice(ctx, 1); // dqn -> td3

    const algo = ctx.state.fields.find((f) => f.key === "algo_name")!;
    expect(algo.options![algo.index ?? 0]).toBe("td3");
    expect(keys()).toContain("param:tau"); // td3's now present
    expect(keys()).not.toContain("param:gamma"); // dqn's gone
  });

  test("the sensor list is the perception axis", () => {
    openWorkflow(ctx, "train");
    const sensor = ctx.state.fields.find((f) => f.key === "sensor")!;
    expect(sensor.options).toEqual(["lidar", "adv_lidar", "camera"]);
  });
});

describe("human play form", () => {
  test("offers map + sensor + random path, and no vehicle", () => {
    openWorkflow(ctx, "human_play");
    expect(keys()).toContain("map_name");
    expect(keys()).toContain("sensor");
    expect(keys()).toContain("random_path");
    expect(keys()).not.toContain("vehicle_id");
  });
});

describe("course form", () => {
  test("builds two racer blocks for an algo-vs-algo race", () => {
    openWorkflow(ctx, "course");
    expect(keys()).toContain("r1_algo");
    expect(keys()).toContain("r1_model_path");
    expect(keys()).toContain("r2_algo");
    expect(keys()).toContain("r2_model_path");
  });

  test("exposes the race controls", () => {
    openWorkflow(ctx, "course");
    expect(keys()).toContain("opponent");
    expect(keys()).toContain("races");
    expect(keys()).toContain("learning");
  });

  test("choosing a human opponent removes racer 2's whole block", () => {
    openWorkflow(ctx, "course");
    ctx.state.focusIndex = ctx.state.fields.findIndex((f) => f.key === "opponent");
    cycleChoice(ctx, 1); // algo -> human

    const opponent = ctx.state.fields.find((f) => f.key === "opponent")!;
    expect(opponent.options![opponent.index ?? 0]).toBe("human");
    expect(keys()).toContain("r1_algo");
    expect(keys()).not.toContain("r2_algo");
    expect(keys()).not.toContain("r2_model_path");
  });

  test("each racer's checkpoint path follows its own algorithm and sensor", () => {
    openWorkflow(ctx, "course");
    const path = ctx.state.fields.find((f) => f.key === "r1_model_path")!;
    expect(path.input!.value).toBe("outputs/dqn_lidar.pth");

    ctx.state.focusIndex = ctx.state.fields.findIndex((f) => f.key === "r1_sensor");
    cycleChoice(ctx, 1); // lidar -> adv_lidar
    expect(ctx.state.fields.find((f) => f.key === "r1_model_path")!.input!.value).toBe(
      "outputs/dqn_adv_lidar.pth",
    );
  });
});

describe("other workflows", () => {
  test("multi-train gates the start button until a vehicle is added", () => {
    openWorkflow(ctx, "multi_train");
    expect(ctx.state.fieldErrors.has("_vehicles")).toBe(true); // gated: no vehicle yet
    expect(ctx.state.multiSpecs.length).toBe(0);
  });

  test("the dropped workflows are absent from the menu", () => {
    for (const gone of ["evaluate", "benchmark"]) {
      expect(MAIN_MENU_OPTIONS.some((o) => (o.id as string) === gone)).toBe(false);
    }
  });

  // Selecting a track has to survive the form layer, not just the payload
  // builders: the field set changes when the kind changes, so a rebuild happens
  // mid-selection and the choice must come back with it.
  const selectKind = (ctx_: Ctx, kind: string) => {
    const i = ctx_.state.fields.findIndex((f) => f.key === "track_kind");
    ctx_.state.focusIndex = i;
    const field = ctx_.state.fields[i];
    const target = field.options!.indexOf(kind);
    cycleChoice(ctx_, target - (field.index ?? 0));
  };

  test("choosing a track kind reveals the matching track names", () => {
    openWorkflow(ctx, "train");
    expect(keys()).toContain("track_kind");
    expect(keys()).not.toContain("track"); // "generated" needs no name

    selectKind(ctx, "lap");
    expect(keys()).toContain("track");
    const track = ctx.state.fields.find((f) => f.key === "track")!;
    expect(track.options).toEqual(["highspeed_ring"]); // gridmap_v2's only lap
    expect(readValues(ctx).track).toBe("highspeed_ring");
  });

  test("the chosen kind survives the rebuild it triggers", () => {
    openWorkflow(ctx, "train");
    selectKind(ctx, "sprint");
    expect(readValues(ctx).track_kind).toBe("sprint");
    expect(readValues(ctx).track).toBe("halfpipes_short");
  });

  test("a track selection reaches the payload the backend receives", () => {
    // The end of the chain the bug hid in: option present everywhere, dropped in
    // transit, run silently falling back to the generated paths.
    openWorkflow(ctx, "train");
    selectKind(ctx, "lap");
    const v = readValues(ctx);
    const track = resolveTrack(ctx.catalog, v.map_name as string, v.track_kind as string, v.track as string);
    expect(track).toBe("highspeed_ring");
  });

  test("going back to generated drops the name field and the track", () => {
    openWorkflow(ctx, "train");
    selectKind(ctx, "lap");
    selectKind(ctx, "generated");
    expect(keys()).not.toContain("track");
    const v = readValues(ctx);
    expect(resolveTrack(ctx.catalog, v.map_name as string, v.track_kind as string, "")).toBe("");
  });

  test("the trajectory form offers every map plus 'all'", () => {
    openWorkflow(ctx, "trajectory");
    const mapField = ctx.state.fields.find((f) => f.key === "map_name");
    expect(mapField?.options).toEqual([...BEAMNG_MAPS, "all"]);
    expect(ctx.state.fields.some((f) => f.key === "overwrite")).toBe(true);
  });
});

describe("log rendering (prevents opentui.dll segfault / exit 3 on long runs)", () => {
  // The assertion is a node-count invariant, not a latency bound: 1000 appends
  // re-layout the joined log text natively, which can exceed bun's 5 s default
  // on slower machines. Generous explicit timeout keeps the guard deterministic.
  test("appending many lines never adds a renderable per line", () => {
    backToMenu(ctx); // start clean
    const before = ctx.scene.outputBox.content.getChildrenCount();
    for (let i = 0; i < LOG_PREVIEW_LIMIT * 5; i++) appendLog(ctx, `log line ${i}`);
    // The fix: logs render through ONE in-place text node, so the child count
    // stays constant no matter how many lines stream in (no native buffer churn).
    expect(ctx.scene.outputBox.content.getChildrenCount()).toBe(before);
  }, 20_000);
});

describe("workflows menu", () => {
  test("moveMenu wraps around and selectedWorkflow tracks the cursor", () => {
    backToMenu(ctx);
    ctx.state.menuIndex = 0;
    expect(selectedWorkflow(ctx)).toBe("train");
    moveMenu(ctx, -1); // wrap to last item
    expect(ctx.state.menuIndex).toBe(MAIN_MENU_OPTIONS.length - 1);
    expect(selectedWorkflow(ctx)).toBe("quit");
    moveMenu(ctx, 1); // wrap back to the first
    expect(selectedWorkflow(ctx)).toBe("train");
  });

  test("opening a workflow moves the cursor onto it", () => {
    backToMenu(ctx);
    openWorkflow(ctx, "course");
    expect(selectedWorkflow(ctx)).toBe("course");
  });
});

describe("navigation", () => {
  test("backToMenu clears the active workflow and the form", () => {
    openWorkflow(ctx, "train");
    expect(ctx.state.activeWorkflow).toBe("train");
    backToMenu(ctx);
    expect(ctx.state.activeWorkflow).toBeNull();
    // Only the welcome markdown node remains; no focusable fields.
    expect(ctx.state.fields.length).toBe(0);
  });
});
