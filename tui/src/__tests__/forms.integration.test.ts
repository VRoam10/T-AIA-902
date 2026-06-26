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
import { cycleChoice } from "../form.ts";
import { moveMenu, selectedWorkflow } from "../menu.ts";
import { appendLog } from "../status.ts";
import { LOG_PREVIEW_LIMIT } from "../widgets.ts";
import { MAIN_MENU_OPTIONS } from "../workflows.ts";
import { makeWelcomeSyntax } from "../welcome.ts";

const catalog: Catalog = {
  algorithms: [
    { name: "q_learning", default_config: { learning_rate: 0.85, discount_factor: 0.99 }, compatible_envs: ["taxi"] },
    { name: "dqn", default_config: { lr: 0.001, gamma: 0.99, batch_size: 64 }, compatible_envs: ["taxi", "beamng_lidar"] },
  ],
  environments: [
    { name: "taxi", metadata: {} },
    { name: "beamng_lidar", metadata: {} },
  ],
  compatible_envs: { q_learning: ["taxi"], dqn: ["taxi", "beamng_lidar"] },
  benchmarks: ["convergence", "comparison", "gridsearch"],
  beamng_maps: ["gridmap_v2", "italy"],
  beamng_vehicles: [{ id: "taxi", label: "Taxi" }],
  multi_algos: ["dqn"],
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
    vehicleIds: catalog.beamng_vehicles.map((v) => v.id),
    readme: "# RL Pipeline\n",
    welcomeSyntax: makeWelcomeSyntax(),
    onChoiceChanged: (f) => onChoiceChanged(ctx, f),
    rebuildActiveForm: (k) => rebuildActiveForm(ctx, k),
  };
});

describe("train form", () => {
  test("builds q_learning's algo/env/hyperparameters by default", () => {
    openWorkflow(ctx, "train");
    expect(keys()).toContain("algo_name");
    expect(keys()).toContain("env_name");
    expect(keys()).toContain("param:discount_factor"); // q_learning's hyperparam
    expect(keys()).not.toContain("param:gamma"); // dqn's, must be absent
  });

  test("switching algorithm rebuilds with the new algo's params + env list", () => {
    openWorkflow(ctx, "train");
    const algoIdx = ctx.state.fields.findIndex((f) => f.key === "algo_name");
    ctx.state.focusIndex = algoIdx;
    cycleChoice(ctx, 1); // q_learning -> dqn

    const algo = ctx.state.fields.find((f) => f.key === "algo_name")!;
    expect(algo.options![algo.index ?? 0]).toBe("dqn");
    expect(keys()).toContain("param:gamma"); // dqn's hyperparam now present
    expect(keys()).not.toContain("param:discount_factor"); // q_learning's gone
    const env = ctx.state.fields.find((f) => f.key === "env_name")!;
    expect(env.options).toEqual(["taxi", "beamng_lidar"]); // dqn's compatible envs
  });
});

describe("other workflows", () => {
  test("evaluate form has model_path + episodes", () => {
    openWorkflow(ctx, "evaluate");
    expect(keys()).toContain("model_path");
    expect(keys()).toContain("n_episodes");
  });

  test("benchmark form has grid + seeds inputs", () => {
    openWorkflow(ctx, "benchmark");
    expect(keys()).toContain("seeds_text");
    expect(keys()).toContain("param_grid_json");
  });

  test("multi-train gates the start button until a vehicle is added", () => {
    openWorkflow(ctx, "multi_train");
    expect(ctx.state.fieldErrors.has("_vehicles")).toBe(true); // gated: no vehicle yet
    expect(ctx.state.multiSpecs.length).toBe(0);
  });
});

describe("log rendering (prevents opentui.dll segfault / exit 3 on long runs)", () => {
  test("appending many lines never adds a renderable per line", () => {
    backToMenu(ctx); // start clean
    const before = ctx.scene.outputBox.content.getChildrenCount();
    for (let i = 0; i < LOG_PREVIEW_LIMIT * 5; i++) appendLog(ctx, `log line ${i}`);
    // The fix: logs render through ONE in-place text node, so the child count
    // stays constant no matter how many lines stream in (no native buffer churn).
    expect(ctx.scene.outputBox.content.getChildrenCount()).toBe(before);
  });
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
    openWorkflow(ctx, "benchmark");
    expect(selectedWorkflow(ctx)).toBe("benchmark");
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
