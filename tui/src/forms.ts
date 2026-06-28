// Per-workflow form builders. Each appends fields to the active form and sets
// ctx.state.runAction to the handler that builds the payload and starts the run.

import { TextRenderable } from "@opentui/core";

import type { Ctx } from "./context.ts";
import { COLOR, GLYPH } from "./theme.ts";
import { bool, num, vJson, vNonNegNumber, vNumber, vPosInt, vSeeds } from "./validators.ts";
import { addAction, addChoice, addDivider, addInput, readValues } from "./form.ts";
import { runTrajectorySequence, startRun } from "./runner.ts";
import { appendLog, setStatus } from "./status.ts";
import {
  BEAMNG_DEFAULTS,
  BEAMNG_MAPS,
  MULTI_COLORS,
  buildBenchmarkPayload,
  buildEvaluatePayload,
  buildHumanPlayPayload,
  buildMultiTrainPayload,
  buildTrainPayload,
  trainSavePath,
  type BeamNGFields,
  type BenchmarkState,
  type EvaluateState,
  type HumanPlayState,
  type MultiTrainState,
  type TrainState,
  type WorkflowId,
} from "./workflows.ts";

function beamngFieldsFrom(values: Record<string, unknown>, withRandom: boolean): BeamNGFields {
  const f: BeamNGFields = {
    map_name: (values.map_name as string) ?? BEAMNG_DEFAULTS.map_name,
    vehicle_id: (values.vehicle_id as string) ?? BEAMNG_DEFAULTS.vehicle_id,
    trajectory_hints: num(values.trajectory_hints, 0),
    body_orientation: bool(values.body_orientation),
    wheel_terrain: bool(values.wheel_terrain),
  };
  if (withRandom) f.random_path = bool(values.random_path);
  return f;
}

function addBeamngFields(ctx: Ctx, envName: string, withRandom: boolean): void {
  if (!envName.startsWith("beamng")) return;
  addDivider(ctx, "beamng options");
  addChoice(ctx, "map_name", "Map", [...BEAMNG_MAPS]);
  addChoice(ctx, "vehicle_id", "Vehicle", ctx.vehicleIds);
  addInput(ctx, "trajectory_hints", "Checkpoint hints", "0", vNonNegNumber);
  addChoice(ctx, "body_orientation", "Body orientation", ["false", "true"]);
  addChoice(ctx, "wheel_terrain", "Wheel terrain", ["false", "true"]);
  if (withRandom) addChoice(ctx, "random_path", "Randomize path", ["false", "true"]);
}

function addAgentParamInputs(ctx: Ctx, algoName: string): void {
  const algo = ctx.catalog.algorithms.find((a) => a.name === algoName);
  if (!algo) return;
  const numeric = Object.entries(algo.default_config).filter(
    ([key, val]) =>
      typeof val === "number" && key !== "n_states" && key !== "n_actions" && key !== "state_type",
  );
  if (numeric.length === 0) return;
  addDivider(ctx, "hyperparameters");
  for (const [key, val] of numeric) addInput(ctx, `param:${key}`, key, String(val), vNumber);
}

function collectAgentParams(values: Record<string, unknown>): Record<string, number> {
  const params: Record<string, number> = {};
  for (const [k, v] of Object.entries(values)) {
    if (k.startsWith("param:")) params[k.slice("param:".length)] = Number(v);
  }
  return params;
}

// Resolve the algorithm and its environment list from a rebuild preset (so a
// rebuild keeps the user's selection), else fall back to the first of each.
// Shared by the train / evaluate / benchmark forms.
function resolveAlgoEnv(ctx: Ctx): { algo: string; envs: string[]; env: string } {
  const algo = (ctx.state.pendingPreset?.algo_name as string) ?? ctx.algoNames[0] ?? "";
  const envs = ctx.catalog.compatible_envs[algo] ?? [];
  const presetEnv = ctx.state.pendingPreset?.env_name as string | undefined;
  const env = presetEnv && envs.includes(presetEnv) ? presetEnv : envs[0] ?? "";
  return { algo, envs, env };
}

function buildTrainForm(ctx: Ctx): void {
  const { catalog, state } = ctx;
  const { algo, envs, env } = resolveAlgoEnv(ctx);
  ctx.scene.formPanel.title = " Train an agent ";
  addChoice(ctx, "algo_name", "Algorithm", ctx.algoNames, Math.max(0, ctx.algoNames.indexOf(algo)));
  addChoice(ctx, "env_name", "Environment", envs, Math.max(0, envs.indexOf(env)));
  addBeamngFields(ctx, env, true);
  addAgentParamInputs(ctx, algo);
  addDivider(ctx, "run");
  addInput(ctx, "n_episodes", "Episodes", "500", vPosInt);
  addInput(ctx, "save_path", "Save path", trainSavePath(algo, env));
  addChoice(ctx, "checkpoint_policy", "Checkpoint", ["resume", "reset"]);
  addAction(ctx, "Start training", "run", undefined, { primary: true });

  state.runAction = () => {
    const v = readValues(ctx);
    const envName = v.env_name as string;
    const trainState: TrainState = {
      algo_name: v.algo_name as string,
      env_name: envName,
      n_episodes: num(v.n_episodes, 500),
      save_path: (v.save_path as string) || undefined,
      agent_params: collectAgentParams(v),
      checkpoint_policy: (v.checkpoint_policy as "resume" | "reset") ?? "resume",
      beamng: envName.startsWith("beamng") ? beamngFieldsFrom(v, true) : undefined,
    };
    startRun(ctx, "train", buildTrainPayload(catalog, trainState), "Train");
  };
}

function buildEvaluateForm(ctx: Ctx): void {
  const { catalog, state } = ctx;
  const { algo, envs, env } = resolveAlgoEnv(ctx);
  ctx.scene.formPanel.title = " Evaluate an agent ";
  addChoice(ctx, "algo_name", "Algorithm", ctx.algoNames, Math.max(0, ctx.algoNames.indexOf(algo)));
  addChoice(ctx, "env_name", "Environment", envs, Math.max(0, envs.indexOf(env)));
  addInput(ctx, "model_path", "Model path", trainSavePath(algo, env));
  addBeamngFields(ctx, env, false);
  addInput(ctx, "n_episodes", "Episodes", "10", vPosInt);
  addAction(ctx, "Start evaluation", "run", undefined, { primary: true });

  state.runAction = () => {
    const v = readValues(ctx);
    const envName = v.env_name as string;
    const evalState: EvaluateState = {
      algo_name: v.algo_name as string,
      env_name: envName,
      model_path: (v.model_path as string) || undefined,
      n_episodes: num(v.n_episodes, 10),
      beamng: envName.startsWith("beamng") ? beamngFieldsFrom(v, false) : undefined,
    };
    startRun(ctx, "evaluate", buildEvaluatePayload(catalog, evalState), "Evaluate");
  };
}

function gridPreset(algo: string): Record<string, number[]> {
  if (algo === "q_learning") {
    return { learning_rate: [0.1, 0.5, 0.85], discount_factor: [0.9, 0.95, 0.99] };
  }
  if (algo === "dqn" || algo === "dqn_per") return { lr: [0.001, 0.0005], gamma: [0.95, 0.99] };
  return { gamma: [0.95, 0.99] };
}

function parseGrid(raw: string): Record<string, unknown[]> {
  try {
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

function buildBenchmarkForm(ctx: Ctx): void {
  const { catalog, state } = ctx;
  const { algo, envs, env } = resolveAlgoEnv(ctx);
  ctx.scene.formPanel.title = " Run a benchmark ";
  addChoice(ctx, "benchmark_name", "Benchmark", ctx.benchNames);
  addChoice(ctx, "algo_name", "Algorithm", ctx.algoNames, Math.max(0, ctx.algoNames.indexOf(algo)));
  addChoice(ctx, "env_name", "Environment", envs, Math.max(0, envs.indexOf(env)));
  addDivider(ctx, "evaluation");
  addInput(ctx, "seeds_text", "Seeds", "0,1,2,3,4", vSeeds);
  addInput(ctx, "eval_episodes", "Eval episodes", "100", vPosInt);
  addInput(ctx, "success_threshold", "Success threshold", "0.0", vNumber);
  addInput(ctx, "max_episodes", "Max episodes", "2000", vPosInt);
  addInput(ctx, "reward_threshold", "Reward threshold", "7.0", vNumber);
  addDivider(ctx, "comparison / gridsearch");
  addInput(ctx, "algos_text", "Algos (comparison)", ctx.algoNames.slice(0, 2).join(","));
  addInput(
    ctx,
    "param_grid_json",
    "Param grid (JSON)",
    JSON.stringify(gridPreset(ctx.algoNames[0] ?? "")),
    vJson,
  );
  addAction(ctx, "Run benchmark", "run", undefined, { primary: true });

  state.runAction = () => {
    const v = readValues(ctx);
    const seeds = String(v.seeds_text)
      .split(",")
      .map((s) => Number(s.trim()))
      .filter((n) => Number.isFinite(n));
    const benchState: BenchmarkState = {
      benchmark_name: v.benchmark_name as string,
      seeds,
      eval_episodes: num(v.eval_episodes, 100),
      success_threshold: num(v.success_threshold, 0),
      max_episodes: num(v.max_episodes, 2000),
      reward_threshold: num(v.reward_threshold, 7),
      algo_name: v.algo_name as string,
      env_name: v.env_name as string,
      algos: String(v.algos_text)
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
      param_grid: parseGrid(String(v.param_grid_json)),
    };
    startRun(ctx, "benchmark", buildBenchmarkPayload(catalog, benchState), "Benchmark");
  };
}

function buildHumanPlayForm(ctx: Ctx): void {
  const { catalog, state } = ctx;
  ctx.scene.formPanel.title = " Human play (BeamNG) ";
  const sensors = ["None", "LiDAR"];
  if (catalog.environments.some((e) => e.name === "beamng_camera")) sensors.push("Camera");
  addChoice(ctx, "map_name", "Map", [...BEAMNG_MAPS]);
  addChoice(ctx, "vehicle_id", "Vehicle", ctx.vehicleIds);
  addChoice(ctx, "sensor", "Sensor", sensors);
  addAction(ctx, "Launch human play", "run", undefined, { primary: true });

  state.runAction = () => {
    const v = readValues(ctx);
    const playState: HumanPlayState = {
      map_name: v.map_name as string,
      vehicle_id: v.vehicle_id as string,
      sensor: v.sensor as string,
    };
    startRun(ctx, "human-play", buildHumanPlayPayload(playState), "Human play");
  };
}

function buildTrajectoryForm(ctx: Ctx): void {
  ctx.scene.formPanel.title = " Generate trajectories (BeamNG) ";
  addChoice(ctx, "map_name", "Map", [...BEAMNG_MAPS, "all"]);
  addChoice(ctx, "overwrite", "Overwrite", ["false", "true"]);
  addAction(ctx, "Generate", "run", undefined, { primary: true });

  ctx.state.runAction = () => {
    const v = readValues(ctx);
    const overwrite = bool(v.overwrite);
    const choice = v.map_name as string;
    const maps = choice === "all" ? [...BEAMNG_MAPS] : [choice];
    runTrajectorySequence(ctx, maps, overwrite, 0);
  };
}

function addMultiSpec(ctx: Ctx): void {
  const { catalog, state } = ctx;
  const algo = "dqn";
  const env = (catalog.compatible_envs[algo] ?? []).find((e) => e.startsWith("beamng")) ?? "beamng";
  const index = state.multiSpecs.length;
  state.multiSpecs.push({
    algo,
    env,
    vehicle_id: "taxi",
    color: MULTI_COLORS[index % MULTI_COLORS.length],
    save_path: `outputs/multi-agents/${algo}_${env}_${index}.pth`,
    trajectory_hints: 0,
    body_orientation: false,
    wheel_terrain: false,
  });
  appendLog(
    ctx,
    `${GLYPH.dot} Added vehicle ${index}: ${algo} / ${env} (${MULTI_COLORS[index % MULTI_COLORS.length]})`,
  );
}

function buildMultiTrainForm(ctx: Ctx): void {
  const { renderer, state } = ctx;
  ctx.scene.formPanel.title = " Multi-agent training (BeamNG) ";
  // The vehicle-count line is a plain text node, not a focusable field.
  const count = new TextRenderable(renderer, {
    id: "multi-count",
    content: `Vehicles configured: ${state.multiSpecs.length}`,
    fg: state.multiSpecs.length > 0 ? COLOR.ok : COLOR.running,
  });
  ctx.scene.formBody.add(count);
  state.formNodes.push(count);
  addChoice(ctx, "map_name", "Map", [...BEAMNG_MAPS]);
  addChoice(ctx, "random_path", "Randomize path", ["false", "true"]);
  addInput(ctx, "n_episodes", "Episodes", "500", vPosInt);
  addInput(ctx, "time_limit_minutes", "Time limit (min)", "0.0", vNonNegNumber);
  addChoice(ctx, "checkpoint_policy", "Checkpoint", ["resume", "reset"]);
  addDivider(ctx, "vehicles");
  addAction(ctx, "Add vehicle", "add", () => {
    addMultiSpec(ctx);
    ctx.rebuildActiveForm();
  });
  addAction(ctx, "Remove last vehicle", "remove", () => {
    if (state.multiSpecs.length === 0) {
      setStatus(ctx, `${GLYPH.err} No vehicle to remove`, COLOR.err);
      return;
    }
    state.multiSpecs.pop();
    ctx.rebuildActiveForm();
  });
  addAction(ctx, "Start multi-agent training", "run", undefined, { primary: true });
  // Gate the primary CTA on having at least one vehicle; re-seeded every rebuild
  // (clearForm wipes fieldErrors, and add/remove rebuild the whole form).
  if (state.multiSpecs.length === 0) state.fieldErrors.set("_vehicles", "Add at least one vehicle");

  state.runAction = () => {
    if (state.multiSpecs.length === 0) {
      setStatus(ctx, `${GLYPH.err} Add at least one vehicle first`, COLOR.err);
      return;
    }
    const v = readValues(ctx);
    const multiState: MultiTrainState = {
      map_name: v.map_name as string,
      random_path: bool(v.random_path),
      n_episodes: num(v.n_episodes, 500),
      time_limit_minutes: num(v.time_limit_minutes, 0),
      checkpoint_policy: (v.checkpoint_policy as "resume" | "reset") ?? "resume",
      specs: state.multiSpecs,
    };
    startRun(ctx, "multi-train", buildMultiTrainPayload(ctx.catalog, multiState), "Multi-agent training");
  };
}

const BUILDERS: Record<Exclude<WorkflowId, "quit">, (ctx: Ctx) => void> = {
  train: buildTrainForm,
  evaluate: buildEvaluateForm,
  benchmark: buildBenchmarkForm,
  human_play: buildHumanPlayForm,
  trajectory: buildTrajectoryForm,
  multi_train: buildMultiTrainForm,
};

export function buildForm(ctx: Ctx, id: Exclude<WorkflowId, "quit">): void {
  BUILDERS[id](ctx);
}
