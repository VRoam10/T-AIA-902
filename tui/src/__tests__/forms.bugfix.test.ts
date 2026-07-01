// Regression coverage for three reported TUI bugs:
//   1. beamng option rows overlapping (long labels / values word-wrapped in a
//      height-1 row, so the 2nd line painted over the next row).
//   2. the train/evaluate save path not following the selected algorithm/env.
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
    { name: "q_learning", default_config: { learning_rate: 0.85, discount_factor: 0.99 }, compatible_envs: ["taxi"] },
    {
      name: "dqn",
      default_config: { lr: 0.001, gamma: 0.99, target_update_freq: 100 },
      compatible_envs: ["taxi", "beamng_lidar", "beamng_continuous_predicted"],
    },
    { name: "ddpg", default_config: { lr: 0.0005, gamma: 0.99 }, compatible_envs: ["beamng_continuous", "beamng_continuous_predicted"] },
  ],
  environments: [
    { name: "taxi", metadata: {} },
    { name: "beamng_lidar", metadata: {} },
    { name: "beamng_continuous", metadata: {} },
    { name: "beamng_continuous_predicted", metadata: {} },
  ],
  compatible_envs: {
    q_learning: ["taxi"],
    dqn: ["taxi", "beamng_lidar", "beamng_continuous_predicted"],
    ddpg: ["beamng_continuous", "beamng_continuous_predicted"],
  },
  benchmarks: ["convergence", "comparison", "gridsearch"],
  beamng_maps: ["gridmap_v2", "italy"],
  beamng_vehicles: [
    { id: "taxi", label: "Burnside (Taxi)" },
    { id: "gavril_t_series", label: "Gavril T-Series" },
  ],
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
    vehicleIds: catalog.beamng_vehicles.map((v) => v.id),
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
    focusKey(ctx, "algo_name");
    cycleChoice(ctx, 1); // q_learning -> dqn (rebuilds; dqn has beamng envs)
    focusKey(ctx, "env_name");
    cycleChoice(ctx, 1); // taxi -> beamng_lidar (rebuilds with the beamng fields)
    expect(fieldVal(ctx, "env_name")).toBe("beamng_lidar");

    await setup.renderOnce();
    const frame = setup.captureCharFrame();

    // When a label word-wraps in a height-1 row, "Body" and "orientation" land
    // on separate lines, so the contiguous string never appears in the frame.
    expect(frame).toContain("Checkpoint hints");
    expect(frame).toContain("Body orientation");
  });

  test("a long environment value renders intact (not split across lines)", async () => {
    const { ctx, setup } = await makeCtx();
    openWorkflow(ctx, "train");
    focusKey(ctx, "algo_name");
    cycleChoice(ctx, 1); // -> dqn
    focusKey(ctx, "env_name");
    cycleChoice(ctx, 2); // taxi -> ... -> beamng_continuous_predicted
    expect(fieldVal(ctx, "env_name")).toBe("beamng_continuous_predicted");

    await setup.renderOnce();
    const frame = setup.captureCharFrame();
    expect(frame).toContain("beamng_continuous_predicted");
  });
});

describe("bug 2: save path follows the selected algorithm/env", () => {
  test("changing the algorithm updates the default save path", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train"); // default q_learning / taxi
    expect(fieldVal(ctx, "save_path")).toBe("outputs/q_learning_taxi.pth");

    focusKey(ctx, "algo_name");
    cycleChoice(ctx, 1); // q_learning -> dqn
    expect(fieldVal(ctx, "algo_name")).toBe("dqn");
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_taxi.pth");
  });

  test("changing the environment updates the default save path", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train");
    focusKey(ctx, "algo_name");
    cycleChoice(ctx, 1); // -> dqn (envs: taxi, beamng_lidar, ...)
    focusKey(ctx, "env_name");
    cycleChoice(ctx, 1); // taxi -> beamng_lidar
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_beamng_lidar.pth");
  });

  test("evaluate model path also follows the algorithm", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "evaluate");
    focusKey(ctx, "algo_name");
    cycleChoice(ctx, 1); // q_learning -> dqn
    expect(fieldVal(ctx, "model_path")).toBe("outputs/dqn_taxi.pth");
  });
});

describe("bug 3: multi-agent training can select a per-vehicle algorithm", () => {
  let ctx: Ctx;
  beforeEach(async () => {
    ctx = (await makeCtx()).ctx;
    ctx.state.multiSpecs = [];
  });

  test("the form exposes an algorithm + environment + vehicle choice", () => {
    openWorkflow(ctx, "multi_train");
    const keys = ctx.state.fields.map((f) => f.key);
    expect(keys).toContain("multi_algo");
    expect(keys).toContain("multi_env");
    expect(keys).toContain("multi_vehicle");
  });

  test("the algorithm choice offers the catalog's multi_algos", () => {
    openWorkflow(ctx, "multi_train");
    const algo = ctx.state.fields.find((f) => f.key === "multi_algo")!;
    expect(algo.options).toEqual(["dqn", "ddpg"]);
  });

  test("changing the algorithm constrains the environment list to its beamng envs", () => {
    openWorkflow(ctx, "multi_train");
    focusKey(ctx, "multi_algo");
    cycleChoice(ctx, 1); // dqn -> ddpg
    const env = ctx.state.fields.find((f) => f.key === "multi_env")!;
    expect(env.options).toEqual(["beamng_continuous", "beamng_continuous_predicted"]);
  });

  test("Add vehicle snapshots the currently selected algorithm into the spec list", () => {
    openWorkflow(ctx, "multi_train");
    focusKey(ctx, "multi_algo");
    cycleChoice(ctx, 1); // dqn -> ddpg
    ctx.state.fields.find((f) => f.key === "add")!.onAction!();
    expect(ctx.state.multiSpecs.length).toBe(1);
    expect(ctx.state.multiSpecs[0].algo).toBe("ddpg");
    expect(ctx.state.multiSpecs[0].env).toBe("beamng_continuous");
  });

  test("the form exposes per-vehicle beamng options and snapshots them", () => {
    openWorkflow(ctx, "multi_train");
    const keys = ctx.state.fields.map((f) => f.key);
    expect(keys).toContain("multi_hints");
    expect(keys).toContain("multi_body_orientation");
    expect(keys).not.toContain("multi_wheel_terrain");

    setInput(ctx, "multi_hints", "2");
    focusKey(ctx, "multi_body_orientation");
    cycleChoice(ctx, 1); // false -> true
    ctx.state.fields.find((f) => f.key === "add")!.onAction!();
    expect(ctx.state.multiSpecs[0].trajectory_hints).toBe(2);
    expect(ctx.state.multiSpecs[0].body_orientation).toBe(true);
    expect(ctx.state.multiSpecs[0].wheel_terrain).toBe(false);
    expect(ctx.state.multiSpecs[0].save_path).toContain("_h2_ori_0.pth");
  });
});

describe("wheel_terrain is removed from the beamng menus (it freezes training)", () => {
  test("the train form offers no wheel_terrain choice", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train");
    focusKey(ctx, "algo_name");
    cycleChoice(ctx, 1); // -> dqn
    focusKey(ctx, "env_name");
    cycleChoice(ctx, 1); // -> beamng_lidar (beamng options present)
    const keys = ctx.state.fields.map((f) => f.key);
    expect(keys).toContain("body_orientation"); // sanity: beamng options are present
    expect(keys).not.toContain("wheel_terrain");
  });
});

describe("save path encodes the beamng options (checkpoint hints, body orientation)", () => {
  test("toggling body orientation appends _ori to the save path", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train");
    focusKey(ctx, "algo_name");
    cycleChoice(ctx, 1); // -> dqn
    focusKey(ctx, "env_name");
    cycleChoice(ctx, 1); // -> beamng_lidar
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_beamng_lidar.pth");
    focusKey(ctx, "body_orientation");
    cycleChoice(ctx, 1); // false -> true
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_beamng_lidar_ori.pth");
  });

  test("checkpoint hints append _h<n> to the save path", async () => {
    const { ctx } = await makeCtx();
    openWorkflow(ctx, "train");
    focusKey(ctx, "algo_name");
    cycleChoice(ctx, 1); // -> dqn
    focusKey(ctx, "env_name");
    cycleChoice(ctx, 1); // -> beamng_lidar
    setInput(ctx, "trajectory_hints", "3");
    refreshDerivedPaths(ctx); // INPUT event handler does this at runtime
    expect(fieldVal(ctx, "save_path")).toBe("outputs/dqn_beamng_lidar_h3.pth");
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
