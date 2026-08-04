// Per-workflow form builders. Each appends fields to the active form and sets
// ctx.state.runAction to the handler that builds the payload and starts the run.

import { TextRenderable } from "@opentui/core";

import type { Catalog } from "./backend.ts";
import type { Ctx } from "./context.ts";
import { COLOR, GLYPH } from "./theme.ts";
import { bool, num, vNonNegNumber, vNumber, vPosInt } from "./validators.ts";
import { addAction, addChoice, addDivider, addInput, readValues } from "./form.ts";
import { runTrajectorySequence, startRun } from "./runner.ts";
import { appendLog, setStatus } from "./status.ts";
import {
  BEAMNG_DEFAULTS,
  BEAMNG_MAPS,
  BEAMNG_SENSORS,
  MULTI_COLORS,
  RACE_COLORS,
  TRACK_KINDS,
  beamngPathSuffix,
  buildCoursePayload,
  buildHumanPlayPayload,
  buildMultiTrainPayload,
  buildTrainPayload,
  resolveTrack,
  trainSavePath,
  tracksFor,
  type BeamNGFields,
  type CourseOpponent,
  type CourseState,
  type HumanPlayState,
  type MultiTrainState,
  type RacerState,
  type TrackKind,
  type TrainState,
  type WorkflowId,
} from "./workflows.ts";

function beamngFieldsFrom(
  values: Record<string, unknown>,
  withRandom: boolean,
  catalog?: Catalog,
): BeamNGFields {
  const map_name = (values.map_name as string) ?? BEAMNG_DEFAULTS.map_name;
  const f: BeamNGFields = {
    map_name,
    sensor: (values.sensor as string) ?? BEAMNG_DEFAULTS.sensor,
    trajectory_hints: num(values.trajectory_hints, 0),
    body_orientation: bool(values.body_orientation),
    road_info: bool(values.road_info),
    wheel_info: bool(values.wheel_info),
    track: catalog
      ? resolveTrack(catalog, map_name, values.track_kind as string, values.track as string)
      : "",
  };
  if (withRandom) {
    f.random_path = bool(values.random_path);
    f.dense_episodes = num(values.dense_episodes, 0);
  }
  return f;
}

// A form builder runs BEFORE the rebuild preset is applied, so anything that
// decides which fields exist must read the preset rather than the freshly built
// field (which still holds its default). Mirrors resolveAlgo.
function presetChoice(ctx: Ctx, key: string, allowed: readonly string[], fallback: string): string {
  const raw = ctx.state.pendingPreset?.[key];
  return typeof raw === "string" && allowed.includes(raw) ? raw : fallback;
}

// The two fields that pick what gets driven: the generated road-network paths, or
// one of the game's own race tracks narrowed to sprints or laps. Two fields rather
// than one long list because a key plus its kind does not fit the value column,
// and because sprint-vs-lap is the choice that actually matters.
function addTrackFields(ctx: Ctx): void {
  const mapName = presetChoice(ctx, "map_name", BEAMNG_MAPS, BEAMNG_DEFAULTS.map_name);
  const kind = presetChoice(ctx, "track_kind", TRACK_KINDS, "generated") as TrackKind;
  addChoice(ctx, "track_kind", "Track", [...TRACK_KINDS], TRACK_KINDS.indexOf(kind));
  if (kind === "generated") return;
  const keys = tracksFor(ctx.catalog, mapName, kind);
  if (keys.length === 0) {
    // The level ships no track of this kind (or BeamNG could not be read).
    addDivider(ctx, `no ${kind} track on ${mapName}`);
    return;
  }
  addChoice(ctx, "track", "└ name", keys);
}

function addBeamngFields(ctx: Ctx, withRandom: boolean): void {
  addDivider(ctx, "beamng options");
  addChoice(ctx, "map_name", "Map", [...BEAMNG_MAPS]);
  addTrackFields(ctx);
  // The perception axis. The output axis is derived from the algorithm, so it is
  // deliberately not a field — see workflows.ts.
  addChoice(ctx, "sensor", "Sensor", [...BEAMNG_SENSORS]);
  // Checkpoint hints and body orientation change what a checkpoint represents,
  // so editing them refreshes the derived save/model path.
  addInput(ctx, "trajectory_hints", "Checkpoint hints", "0", vNonNegNumber, () => refreshDerivedPaths(ctx));
  addChoice(ctx, "body_orientation", "Body orientation", ["false", "true"]);
  // Road position (edges, road-relative heading, curvature, look-ahead) and wheel
  // performance (slip, slide, ABS, lateral g). Both change the observation width, so
  // both feed the derived save path — see controller.onChoiceChanged.
  addChoice(ctx, "road_info", "Road position", ["false", "true"]);
  addChoice(ctx, "wheel_info", "Wheel performance", ["false", "true"]);
  if (withRandom) {
    addChoice(ctx, "random_path", "Randomize path", ["false", "true"]);
    // Curriculum warm-up: dense checkpoints (8 m) for the first N episodes,
    // then sparse (25 m). Training-only; evaluation always runs sparse.
    // Label kept within LABEL_WIDTH (see context.ts) — "Dense warm-up episodes"
    // overflowed it and rendered clipped, colliding with its own value.
    addInput(ctx, "dense_episodes", "Warm-up episodes", "0", vNonNegNumber);
  }
}

// Recompute the train "save path" from the currently selected algo + sensor +
// beamng options, so the path always reflects the config (the user shouldn't have
// to hand-edit it). Called after a rebuild and whenever a contributing field
// changes; no-op outside the train form.
export function refreshDerivedPaths(ctx: Ctx): void {
  if (ctx.state.activeWorkflow !== "train") return;
  const field = ctx.state.fields.find((f) => f.key === "save_path");
  if (!field || field.kind !== "input" || !field.input) return;
  const v = readValues(ctx);
  const beamng = beamngFieldsFrom(v, false);
  const next = trainSavePath(v.algo_name as string, beamng.sensor, beamng);
  if (field.input.value !== next) {
    field.input.value = next;
    ctx.renderer.requestRender();
  }
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

// Resolve the algorithm from a rebuild preset (so a rebuild keeps the user's
// selection), else fall back to the first registered one.
function resolveAlgo(ctx: Ctx): string {
  return (ctx.state.pendingPreset?.algo_name as string) ?? ctx.algoNames[0] ?? "";
}

function buildTrainForm(ctx: Ctx): void {
  const { catalog, state } = ctx;
  const algo = resolveAlgo(ctx);
  ctx.scene.formPanel.title = " Train an agent ";
  addChoice(ctx, "algo_name", "Algorithm", ctx.algoNames, Math.max(0, ctx.algoNames.indexOf(algo)));
  addBeamngFields(ctx, true);
  addAgentParamInputs(ctx, algo);
  addDivider(ctx, "run");
  addInput(ctx, "n_episodes", "Episodes", "500", vPosInt);
  addInput(ctx, "save_path", "Save path", trainSavePath(algo, BEAMNG_DEFAULTS.sensor));
  addChoice(ctx, "checkpoint_policy", "Checkpoint", ["resume", "reset"]);
  addAction(ctx, "Start training", "run", undefined, { primary: true });

  state.runAction = () => {
    const v = readValues(ctx);
    const trainState: TrainState = {
      algo_name: v.algo_name as string,
      n_episodes: num(v.n_episodes, 500),
      save_path: (v.save_path as string) || undefined,
      agent_params: collectAgentParams(v),
      checkpoint_policy: (v.checkpoint_policy as "resume" | "reset") ?? "resume",
      beamng: beamngFieldsFrom(v, true, catalog),
    };
    startRun(ctx, "train", buildTrainPayload(catalog, trainState), "Train");
  };
}

function buildHumanPlayForm(ctx: Ctx): void {
  const { state } = ctx;
  ctx.scene.formPanel.title = " Human play ";
  addChoice(ctx, "map_name", "Map", [...BEAMNG_MAPS]);
  addTrackFields(ctx);
  // The sensor picks which observation readout is shown while you drive.
  addChoice(ctx, "sensor", "Sensor", [...BEAMNG_SENSORS]);
  addChoice(ctx, "random_path", "Randomize path", ["false", "true"]);
  addChoice(ctx, "road_info", "Road position", ["false", "true"]);
  addChoice(ctx, "wheel_info", "Wheel performance", ["false", "true"]);
  addAction(ctx, "Launch human play", "run", undefined, { primary: true });

  state.runAction = () => {
    const v = readValues(ctx);
    const playState: HumanPlayState = {
      map_name: v.map_name as string,
      sensor: v.sensor as string,
      random_path: bool(v.random_path),
      road_info: bool(v.road_info),
      wheel_info: bool(v.wheel_info),
      track: resolveTrack(ctx.catalog, v.map_name as string, v.track_kind as string, v.track as string),
    };
    startRun(ctx, "human-play", buildHumanPlayPayload(playState), "Human play");
  };
}

// Resolve the "next vehicle" algo from a rebuild preset (so a rebuild keeps the
// user's selection) — mirrors resolveAlgo for the train form.
function resolveMultiAlgo(ctx: Ctx): { algos: string[]; algo: string } {
  const algos = ctx.catalog.multi_algos.length > 0 ? ctx.catalog.multi_algos : ctx.algoNames;
  const presetAlgo = ctx.state.pendingPreset?.multi_algo as string | undefined;
  const algo = presetAlgo && algos.includes(presetAlgo) ? presetAlgo : algos[0] ?? "";
  return { algos, algo };
}

// Snapshot the currently selected algo/sensor + beamng options into a spec.
function addMultiSpec(ctx: Ctx): void {
  const { state } = ctx;
  const v = readValues(ctx);
  const index = state.multiSpecs.length;
  const algo = (v.multi_algo as string) || ctx.catalog.multi_algos[0] || ctx.algoNames[0] || "dqn";
  const sensor = (v.multi_sensor as string) || BEAMNG_DEFAULTS.sensor;
  const trajectory_hints = num(v.multi_hints, 0);
  const body_orientation = bool(v.multi_body_orientation);
  const road_info = bool(v.multi_road_info);
  const wheel_info = bool(v.multi_wheel_info);
  const color = MULTI_COLORS[index % MULTI_COLORS.length];
  const suffix = beamngPathSuffix({ trajectory_hints, body_orientation, road_info, wheel_info });
  state.multiSpecs.push({
    algo,
    sensor,
    color,
    save_path: `outputs/multi-agents/${algo}_${sensor}${suffix}_${index}.pth`,
    trajectory_hints,
    body_orientation,
    road_info,
    wheel_info,
  });
  appendLog(ctx, `${GLYPH.dot} Added vehicle ${index}: ${algo} / ${sensor} (${color})`);
}

// Render the configured vehicles as a read-only list (plain text nodes, not
// focusable fields). Mirrors addDivider's node bookkeeping.
function addMultiSpecList(ctx: Ctx): void {
  const { renderer, scene, state } = ctx;
  const push = (id: string, content: string, fg: string) => {
    const node = new TextRenderable(renderer, { id, content, fg, wrapMode: "none" });
    scene.formBody.add(node);
    state.formNodes.push(node);
  };
  if (state.multiSpecs.length === 0) {
    push("multi-empty", `${GLYPH.dot} No vehicles yet — set the options above, then Add vehicle`, COLOR.running);
    return;
  }
  state.multiSpecs.forEach((s, i) => {
    const opts = [
      s.trajectory_hints > 0 ? `h${s.trajectory_hints}` : "",
      s.body_orientation ? "ori" : "",
      s.road_info ? "road" : "",
      s.wheel_info ? "whl" : "",
    ]
      .filter(Boolean)
      .join(" ");
    const tail = opts ? `  ${opts}` : "";
    push(`multi-spec-${i}`, `  ${i}  ${s.algo} / ${s.sensor}${tail}  (${s.color})`, COLOR.ok);
  });
}

function buildMultiTrainForm(ctx: Ctx): void {
  const { state } = ctx;
  ctx.scene.formPanel.title = " Multi-agent training ";
  addChoice(ctx, "map_name", "Map", [...BEAMNG_MAPS]);
  // On a game track every car trains the same authored line from a starting grid,
  // instead of each getting its own generated path.
  addTrackFields(ctx);
  addChoice(ctx, "random_path", "Randomize path", ["false", "true"]);
  addInput(ctx, "n_episodes", "Episodes", "500", vPosInt);
  addInput(ctx, "time_limit_minutes", "Time limit (min)", "0.0", vNonNegNumber);
  addChoice(ctx, "checkpoint_policy", "Checkpoint", ["resume", "reset"]);

  // Per-vehicle configuration: each car picks its own algorithm + sensor;
  // "Add vehicle" snapshots the selection into the list.
  const { algos, algo } = resolveMultiAlgo(ctx);
  addDivider(ctx, "add a vehicle");
  addChoice(ctx, "multi_algo", "Algorithm", algos, Math.max(0, algos.indexOf(algo)));
  addChoice(ctx, "multi_sensor", "Sensor", [...BEAMNG_SENSORS]);
  addInput(ctx, "multi_hints", "Checkpoint hints", "0", vNonNegNumber);
  addChoice(ctx, "multi_body_orientation", "Body orientation", ["false", "true"]);
  addChoice(ctx, "multi_road_info", "Road position", ["false", "true"]);
  addChoice(ctx, "multi_wheel_info", "Wheel performance", ["false", "true"]);
  addAction(ctx, "Add vehicle", "add", () => {
    addMultiSpec(ctx);
    ctx.rebuildActiveForm("multi_algo");
  });

  addDivider(ctx, "configured vehicles");
  addMultiSpecList(ctx);
  addAction(ctx, "Remove last vehicle", "remove", () => {
    if (state.multiSpecs.length === 0) {
      setStatus(ctx, `${GLYPH.err} No vehicle to remove`, COLOR.err);
      return;
    }
    state.multiSpecs.pop();
    ctx.rebuildActiveForm("multi_algo");
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
      track: resolveTrack(ctx.catalog, v.map_name as string, v.track_kind as string, v.track as string),
    };
    startRun(ctx, "multi-train", buildMultiTrainPayload(ctx.catalog, multiState), "Multi-agent training");
  };
}

// One racer block. A race is always exactly two entrants, so these are fixed
// fields rather than the dynamic add/remove list multi-agent training uses.
function addRacerFields(ctx: Ctx, index: number, algos: string[], defaultAlgo: string): void {
  const n = index + 1;
  addDivider(ctx, `racer ${n}`);
  addChoice(ctx, `r${n}_algo`, "Algorithm", algos, Math.max(0, algos.indexOf(defaultAlgo)));
  addChoice(ctx, `r${n}_sensor`, "Sensor", [...BEAMNG_SENSORS]);
  addInput(ctx, `r${n}_hints`, "Checkpoint hints", "0", vNonNegNumber, () => refreshRacerPaths(ctx));
  addChoice(ctx, `r${n}_body_orientation`, "Body orientation", ["false", "true"]);
  addChoice(ctx, `r${n}_road_info`, "Road position", ["false", "true"]);
  addChoice(ctx, `r${n}_wheel_info`, "Wheel performance", ["false", "true"]);
  addInput(ctx, `r${n}_model_path`, "Checkpoint", trainSavePath(defaultAlgo, BEAMNG_DEFAULTS.sensor));
}

// Keep each racer's checkpoint path in step with its algorithm + sensor, the same
// way the train form keeps its save path current.
export function refreshRacerPaths(ctx: Ctx): void {
  if (ctx.state.activeWorkflow !== "course") return;
  const v = readValues(ctx);
  for (const n of [1, 2]) {
    const field = ctx.state.fields.find((f) => f.key === `r${n}_model_path`);
    if (!field || field.kind !== "input" || !field.input) continue;
    const next = trainSavePath(v[`r${n}_algo`] as string, v[`r${n}_sensor`] as string, {
      trajectory_hints: num(v[`r${n}_hints`], 0),
      body_orientation: bool(v[`r${n}_body_orientation`]),
      road_info: bool(v[`r${n}_road_info`]),
      wheel_info: bool(v[`r${n}_wheel_info`]),
    });
    if (field.input.value !== next) field.input.value = next;
  }
  ctx.renderer.requestRender();
}

function racerFrom(values: Record<string, unknown>, index: number): RacerState {
  const n = index + 1;
  return {
    algo: values[`r${n}_algo`] as string,
    sensor: values[`r${n}_sensor`] as string,
    model_path: (values[`r${n}_model_path`] as string) ?? "",
    color: RACE_COLORS[index % RACE_COLORS.length],
    trajectory_hints: num(values[`r${n}_hints`], 0),
    body_orientation: bool(values[`r${n}_body_orientation`]),
    road_info: bool(values[`r${n}_road_info`]),
    wheel_info: bool(values[`r${n}_wheel_info`]),
  };
}

function buildCourseForm(ctx: Ctx): void {
  const { state } = ctx;
  ctx.scene.formPanel.title = " Course mode (race) ";
  const algos = ctx.algoNames;
  const defaultAlgo = algos[0] ?? "dqn";
  const opponent = (state.pendingPreset?.opponent as CourseOpponent) ?? "algo";

  addChoice(ctx, "map_name", "Map", [...BEAMNG_MAPS]);
  // Race the generated paths, or one of the game's own tracks — a "lap" track is
  // a real closed circuit, which the generated open roads can never be.
  addTrackFields(ctx);
  addChoice(ctx, "opponent", "Opponent", ["algo", "human"], opponent === "human" ? 1 : 0);
  addInput(ctx, "races", "Races", "1", vPosInt);
  // Learning off = frozen policies, so the race shows what the checkpoints
  // actually learned. On = the agents keep updating with the leader reward.
  addChoice(ctx, "learning", "Learning", ["false", "true"]);

  addRacerFields(ctx, 0, algos, defaultAlgo);
  if (opponent === "human") {
    addDivider(ctx, "racer 2");
    const node = new TextRenderable(ctx.renderer, {
      id: "course-human",
      content: `  ${GLYPH.dot} You are racing — drive in-game (realtime)`,
      fg: COLOR.ok,
      wrapMode: "none",
    });
    ctx.scene.formBody.add(node);
    ctx.state.formNodes.push(node);
  } else {
    addRacerFields(ctx, 1, algos, algos[1] ?? defaultAlgo);
  }

  addAction(ctx, "Start race", "run", undefined, { primary: true });

  state.runAction = () => {
    const v = readValues(ctx);
    const isHuman = (v.opponent as string) === "human";
    const racers = isHuman
      ? [racerFrom(v, 0), { ...racerFrom(v, 0), color: RACE_COLORS[1], human: true }]
      : [racerFrom(v, 0), racerFrom(v, 1)];
    const courseState: CourseState = {
      map_name: v.map_name as string,
      opponent: isHuman ? "human" : "algo",
      // laps > 1 needs a closed circuit; the backend rejects anything else.
      laps: 1,
      races: num(v.races, 1),
      learning: bool(v.learning),
      racers,
      track: resolveTrack(ctx.catalog, v.map_name as string, v.track_kind as string, v.track as string),
    };
    startRun(ctx, "course", buildCoursePayload(courseState), "Race");
  };
}

function buildTrajectoryForm(ctx: Ctx): void {
  ctx.scene.formPanel.title = " Generate trajectories ";
  // "all" walks every map in one run — what you want after anything that changes
  // the generated geometry, since every cache goes stale together.
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

const BUILDERS: Record<Exclude<WorkflowId, "quit">, (ctx: Ctx) => void> = {
  train: buildTrainForm,
  multi_train: buildMultiTrainForm,
  human_play: buildHumanPlayForm,
  course: buildCourseForm,
  trajectory: buildTrajectoryForm,
};

export function buildForm(ctx: Ctx, id: Exclude<WorkflowId, "quit">): void {
  BUILDERS[id](ctx);
}
