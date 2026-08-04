// Regression coverage for three reported TUI bugs:
//   1. beamng option rows overlapping (long labels / values word-wrapped in a
//      height-1 row, so the 2nd line painted over the next row).
//   2. the train save path not following the selected algorithm/sensor.
//   3. multi-agent training offering no per-vehicle algorithm selection.
// Driven through a real OpenTUI test renderer so the layout bug is caught by
// inspecting the actual rendered frame.

import { beforeEach, describe, expect, test } from "bun:test";
import { createTestRenderer } from "@opentui/core/testing";

import type { Catalog } from "../backend.ts";
import { createState, type Ctx } from "../context.ts";
import { buildScene } from "../scene.ts";
import { onChoiceChanged, openWorkflow, rebuildActiveForm } from "../controller.ts";
import { cycleChoice } from "../form.ts";
import { installKeymap } from "../keymap.ts";
import { refreshDerivedPaths } from "../forms.ts";
import { makeWelcomeSyntax } from "../welcome.ts";

const catalog: Catalog = {
  algorithms: [
    { name: "dqn", default_config: { lr: 0.001, gamma: 0.99, target_update_freq: 100 }, compatible_envs: ["beamng"] },
    { name: "ddpg", default_config: { actor_lr: 0.0005, gamma: 0.99 }, compatible_envs: ["beamng"] },
  ],
  environments: [{ name: "beamng", metadata: {} }],
  compatible_envs: { dqn: ["beamng"], ddpg: ["beamng"] },
  benchmarks: ["convergence", "comparison", "gridsearch"],
  beamng_maps: ["gridmap_v2", "italy"],
  beamng_sensors: ["lidar", "adv_lidar", "camera"],
  multi_algos: ["dqn", "ddpg"],
};

async function makeCtx(width = 120, height = 44): Promise<{ ctx: Ctx; setup: Awaited<ReturnType<typeof createTestRenderer>> }> {
  const setup = await createTestRenderer({ width, height });
  const scene = buildScene(setup.renderer, catalog);
  const ctx: Ctx = {
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
  return { ctx, setup };
}

const fieldVal = (ctx: Ctx, key: string) => ctx.state.fields.find((f) => f.key === key)!.read() as string;
const focusKey = (ctx: Ctx, key: string) => {
  ctx.state.focusIndex = ctx.state.fields.findIndex((f) => f.key === key);
};
const setInput = (ctx: Ctx, key: string, value: string) => {
  ctx.state.fields.find((f) => f.key === key)!.input!.value = value;
};

describe("bug 1: beamng option rows must not wrap/overlap", () => {
  test("long option labels each render intact on a single line", async () => {
    const { ctx, setup } = await makeCtx();
    openWorkflow(ctx, "train");

    await setup.renderOnce();
    const frame = setup.captureCharFrame();

    // When a label word-wraps in a height-1 row, "Body" and "orientation" land
    // on separate lines, so the contiguous string never appears in the frame.
    expect(frame).toContain("Checkpoint hints");
    expect(frame).toContain("Body orientation");
    expect(frame).toContain("Warm-up episodes");
  });

  test("a long derived save path renders intact (not split across lines)", async () => {
    const { ctx, setup } = await makeCtx();
    openWorkflow(ctx, "train");
    focusKey(ctx, "sensor");
    cycleChoice(ctx, 1); // lidar -> adv_lidar
    setInput(ctx, "trajectory_hints", "3");
    refreshDerivedPaths(ctx);
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_adv_lidar_h3.pth");

    await setup.renderOnce();
    expect(setup.captureCharFrame()).toContain("outputs/dqn_adv_lidar_h3.pth");
  });
});

describe("bug 2: save path follows the selected algorithm/sensor", () => {
  test("changing the algorithm updates the default save path", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train"); // default dqn / lidar
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_lidar.pth");

    focusKey(ctx, "algo_name");
    cycleChoice(ctx, 1); // dqn -> ddpg
    expect(fieldVal(ctx, "algo_name")).toBe("ddpg");
    expect(fieldVal(ctx, "save_path")).toBe("outputs/ddpg_lidar.pth");
  });

  test("changing the sensor updates the default save path", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train");
    focusKey(ctx, "sensor");
    cycleChoice(ctx, 1); // lidar -> adv_lidar
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_adv_lidar.pth");
    cycleChoice(ctx, 1); // adv_lidar -> camera
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_camera.pth");
  });

  test("each course racer's checkpoint path follows its own selection", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "course");
    expect(fieldVal(ctx, "r1_model_path")).toBe("outputs/dqn_lidar.pth");

    focusKey(ctx, "r2_sensor");
    cycleChoice(ctx, 2); // lidar -> camera
    expect(fieldVal(ctx, "r2_model_path")).toBe("outputs/ddpg_camera.pth");
    // Racer 1 must be untouched by racer 2's change.
    expect(fieldVal(ctx, "r1_model_path")).toBe("outputs/dqn_lidar.pth");
  });
});

describe("bug 3: multi-agent training can select a per-vehicle algorithm", () => {
  let ctx: Ctx;
  beforeEach(async () => {
    ctx = (await makeCtx()).ctx;
    ctx.state.multiSpecs = [];
  });

  test("the form exposes an algorithm + sensor choice (and no vehicle)", () => {
    openWorkflow(ctx, "multi_train");
    const keys = ctx.state.fields.map((f) => f.key);
    expect(keys).toContain("multi_algo");
    expect(keys).toContain("multi_sensor");
    // The env name is gone (one env), and so is the vehicle (one car).
    expect(keys).not.toContain("multi_env");
    expect(keys).not.toContain("multi_vehicle");
  });

  test("the algorithm choice offers the catalog's multi_algos", () => {
    openWorkflow(ctx, "multi_train");
    const algo = ctx.state.fields.find((f) => f.key === "multi_algo")!;
    expect(algo.options).toEqual(["dqn", "ddpg"]);
  });

  test("the sensor choice offers the full perception axis for any algorithm", () => {
    openWorkflow(ctx, "multi_train");
    focusKey(ctx, "multi_algo");
    cycleChoice(ctx, 1); // dqn -> ddpg
    const sensor = ctx.state.fields.find((f) => f.key === "multi_sensor")!;
    // Every sensor works with every algorithm now; only the action head differs,
    // and that is derived rather than chosen.
    expect(sensor.options).toEqual(["lidar", "adv_lidar", "camera"]);
  });

  test("Add vehicle snapshots the currently selected algorithm + sensor", () => {
    openWorkflow(ctx, "multi_train");
    focusKey(ctx, "multi_algo");
    cycleChoice(ctx, 1); // dqn -> ddpg
    focusKey(ctx, "multi_sensor");
    cycleChoice(ctx, 1); // lidar -> adv_lidar
    ctx.state.fields.find((f) => f.key === "add")!.onAction!();
    expect(ctx.state.multiSpecs.length).toBe(1);
    expect(ctx.state.multiSpecs[0].algo).toBe("ddpg");
    expect(ctx.state.multiSpecs[0].sensor).toBe("adv_lidar");
  });

  test("the form exposes per-vehicle beamng options and snapshots them", () => {
    openWorkflow(ctx, "multi_train");
    const keys = ctx.state.fields.map((f) => f.key);
    expect(keys).toContain("multi_hints");
    expect(keys).toContain("multi_body_orientation");
    expect(keys).toContain("multi_road_info");
    expect(keys).toContain("multi_wheel_info");

    setInput(ctx, "multi_hints", "2");
    focusKey(ctx, "multi_body_orientation");
    cycleChoice(ctx, 1); // false -> true
    ctx.state.fields.find((f) => f.key === "add")!.onAction!();
    expect(ctx.state.multiSpecs[0].trajectory_hints).toBe(2);
    expect(ctx.state.multiSpecs[0].body_orientation).toBe(true);
    expect(ctx.state.multiSpecs[0].road_info).toBe(false);
    expect(ctx.state.multiSpecs[0].wheel_info).toBe(false);
    expect(ctx.state.multiSpecs[0].save_path).toContain("_h2_ori_0.pth");
  });
});

describe("road_info and wheel_info are offered in the beamng menus (the freeze that kept road_info out is fixed)", () => {
  test("the train form offers both choices", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train");
    const keys = ctx.state.fields.map((f) => f.key);
    expect(keys).toContain("body_orientation"); // sanity: beamng options are present
    expect(keys).toContain("road_info");
    expect(keys).toContain("wheel_info");
  });
});

describe("save path encodes the beamng options (checkpoint hints, body orientation)", () => {
  test("toggling body orientation appends _ori to the save path", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train");
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_lidar.pth");
    focusKey(ctx, "body_orientation");
    cycleChoice(ctx, 1); // false -> true
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_lidar_ori.pth");
  });

  test("checkpoint hints append _h<n> to the save path", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train");
    setInput(ctx, "trajectory_hints", "3");
    refreshDerivedPaths(ctx); // INPUT event handler does this at runtime
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_lidar_h3.pth");
  });
});

describe("up / down arrows navigate between fields", () => {
  test("down moves to the next field, up moves back", async () => {
    const { ctx, setup } = await makeCtx();
    installKeymap(ctx);
    openWorkflow(ctx, "train"); // focus starts on field 0 (algo_name)
    expect(ctx.state.focusIndex).toBe(0);
    setup.mockInput.pressArrow("down");
    expect(ctx.state.focusIndex).toBe(1);
    setup.mockInput.pressArrow("up");
    expect(ctx.state.focusIndex).toBe(0);
  });
});
