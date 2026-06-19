// OpenTUI front-end for the RL pipeline — a keyboard-driven "command center".
// All RL work happens in Python via `core.tui_backend`; this file is
// presentation + input only. Pure logic lives in workflows.ts / progress.ts.

import {
  BoxRenderable,
  InputRenderable,
  InputRenderableEvents,
  MarkdownRenderable,
  RGBA,
  ScrollBoxRenderable,
  SelectRenderable,
  SelectRenderableEvents,
  SyntaxStyle,
  TextRenderable,
  createCliRenderer,
  type CliRenderer,
  type KeyEvent,
} from "@opentui/core";

import {
  loadCatalog,
  runBackend,
  type BackendCommand,
  type BackendEvent,
  type Catalog,
} from "./backend.ts";
import { parseProgress, progressBar } from "./progress.ts";
import { BORDER, COLOR, GLYPH, SPINNER } from "./theme.ts";
import { makeButton, makeHelpOverlay, makeLogModal, makePanel, type Button } from "./widgets.ts";
import {
  BEAMNG_DEFAULTS,
  BEAMNG_MAPS,
  MAIN_MENU_OPTIONS,
  MULTI_COLORS,
  buildBenchmarkPayload,
  buildEvaluatePayload,
  buildHumanPlayPayload,
  buildMultiTrainPayload,
  buildTrainPayload,
  buildTrajectoryPayload,
  trainSavePath,
  type BeamNGFields,
  type BenchmarkState,
  type EvaluateState,
  type HumanPlayState,
  type MultiSpecState,
  type MultiTrainState,
  type TrainState,
  type TrajectoryState,
  type WorkflowId,
} from "./workflows.ts";

// --------------------------------------------------------------------------- //
// Field model: one focusable form control plus how to read/validate its value.
// --------------------------------------------------------------------------- //
type FieldKind = "choice" | "input" | "action";

interface Field {
  key: string;
  label: string;
  kind: FieldKind;
  row: BoxRenderable;
  labelText: TextRenderable;
  valueText?: TextRenderable;
  input?: InputRenderable;
  button?: Button;
  options?: string[];
  index?: number;
  primary?: boolean; // an action gated on form validity
  validate?: (raw: string) => string | null;
  read(): unknown;
  onAction?: () => void;
}

const LABEL_WIDTH = 16;
const VALUE_WIDTH = 26;

const wrap = (i: number, n: number): number => ((i % n) + n) % n;
const num = (v: unknown, fallback: number): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};
const bool = (v: unknown): boolean => v === true || v === "true" || v === "yes" || v === "y";

// Validators return an error message, or null when the value is acceptable.
const vNumber = (raw: string): string | null =>
  Number.isFinite(Number(raw)) && raw.trim() !== "" ? null : "must be a number";
const vPosInt = (raw: string): string | null => {
  const n = Number(raw);
  return Number.isInteger(n) && n >= 1 ? null : "must be an integer ≥ 1";
};
const vNonNegNumber = (raw: string): string | null => {
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? null : "must be a number ≥ 0";
};
const vSeeds = (raw: string): string | null => {
  const parts = raw.split(",").map((s) => s.trim()).filter(Boolean);
  if (parts.length === 0) return "enter at least one seed";
  return parts.every((s) => Number.isFinite(Number(s))) ? null : "seeds must be numbers";
};
const vJson = (raw: string): string | null => {
  try {
    JSON.parse(raw);
    return null;
  } catch {
    return "invalid JSON";
  }
};

// --------------------------------------------------------------------------- //
// Renderer + catalog bootstrap
// --------------------------------------------------------------------------- //
// autoFocus:false — a mouse click must NOT steal keyboard focus from the menu
// (that left the welcome screen stuck, unable to navigate back to the workflows).
// backgroundColor:"transparent" — let the terminal's own background show through.
const renderer: CliRenderer = await createCliRenderer({
  exitOnCtrlC: false,
  autoFocus: false,
  backgroundColor: "transparent",
});
renderer.setTerminalTitle("RL Pipeline");
const catalog: Catalog = await loadCatalog();

// The Welcome screen renders the project README (as real markdown) so the
// landing view is useful.
const README_TEXT = await (async () => {
  const root = process.env.T_AIA_ROOT ?? "..";
  try {
    return await Bun.file(`${root}/README.md`).text();
  } catch {
    return "# RL Pipeline\n\nREADME.md introuvable.";
  }
})();

// Tokyo Night theme for the rendered README markdown.
const welcomeSyntax = SyntaxStyle.fromStyles({
  default: { fg: RGBA.fromHex(COLOR.value) },
  "markup.heading": { fg: RGBA.fromHex(COLOR.accent), bold: true },
  "markup.heading.1": { fg: RGBA.fromHex(COLOR.accent), bold: true },
  "markup.heading.2": { fg: RGBA.fromHex(COLOR.action), bold: true },
  "markup.heading.3": { fg: RGBA.fromHex(COLOR.cyan), bold: true },
  "markup.bold": { fg: RGBA.fromHex(COLOR.fg), bold: true },
  "markup.italic": { fg: RGBA.fromHex(COLOR.fg), italic: true },
  "markup.list": { fg: RGBA.fromHex(COLOR.action) },
  "markup.raw": { fg: RGBA.fromHex(COLOR.cyan) },
  "markup.link": { fg: RGBA.fromHex(COLOR.accent), underline: true },
  "markup.quote": { fg: RGBA.fromHex(COLOR.muted), italic: true },
});

// --------------------------------------------------------------------------- //
// Static layout
// --------------------------------------------------------------------------- //
const screen = new BoxRenderable(renderer, {
  id: "screen",
  width: "100%",
  height: "100%",
  flexDirection: "column",
});
renderer.root.add(screen);

// Header: wordmark + breadcrumb + live status badge.
const header = new BoxRenderable(renderer, {
  id: "header",
  height: 3,
  flexShrink: 0,
  borderStyle: BORDER.header,
  borderColor: COLOR.accent,
  flexDirection: "row",
  alignItems: "center",
  paddingLeft: 1,
  paddingRight: 1,
});
const wordmark = new TextRenderable(renderer, {
  id: "wordmark",
  content: `${GLYPH.logo} RL PIPELINE`,
  fg: COLOR.accent,
});
const breadcrumb = new TextRenderable(renderer, {
  id: "breadcrumb",
  content: "",
  fg: COLOR.muted,
  flexGrow: 1,
  marginLeft: 3,
});
const headerBadge = new TextRenderable(renderer, { id: "badge", content: "ready", fg: COLOR.muted });
header.add(wordmark);
header.add(breadcrumb);
header.add(headerBadge);
screen.add(header);

// Body: sidebar (workflows + status) | main (form).
const body = new BoxRenderable(renderer, {
  id: "body",
  flexGrow: 1,
  flexShrink: 1,
  minHeight: 0,
  flexDirection: "row",
});
screen.add(body);

const sidebar = new BoxRenderable(renderer, {
  id: "sidebar",
  width: 32,
  flexShrink: 0,
  flexDirection: "column",
});
body.add(sidebar);

// Declarative responsive sizing (no resize handler): the panel grows to fill
// the sidebar and shrinks down to a floor that still shows the whole menu.
const workflowsPanel = makePanel(renderer, {
  id: "workflows",
  title: "Workflows",
  flexGrow: 1,
  flexShrink: 1,
  minHeight: MAIN_MENU_OPTIONS.length + 2,
});
sidebar.add(workflowsPanel);

const menu = new SelectRenderable(renderer, {
  id: "menu",
  width: "100%",
  height: MAIN_MENU_OPTIONS.length,
  options: menuOptions(null),
  wrapSelection: true,
  selectedBackgroundColor: COLOR.borderMuted,
  selectedTextColor: COLOR.accent,
  focusedBackgroundColor: COLOR.surfaceFocus,
  focusedTextColor: COLOR.fg,
  textColor: COLOR.label,
  showDescription: false,
});
workflowsPanel.add(menu);

const statusPanel = makePanel(renderer, {
  id: "statuspanel",
  title: "Status",
  height: 6,
  flexShrink: 1,
  minHeight: 4,
});
sidebar.add(statusPanel);
const statusLine = new TextRenderable(renderer, {
  id: "status-line",
  content: "Pick a workflow, press ⏎",
  fg: COLOR.label,
});
const statusBar = new TextRenderable(renderer, { id: "status-bar", content: "", fg: COLOR.running });
const statusPostfix = new TextRenderable(renderer, {
  id: "status-postfix",
  content: `${catalog.algorithms.length} algos ${GLYPH.dot} ${catalog.environments.length} envs`,
  fg: COLOR.muted,
});
statusPanel.add(statusLine);
statusPanel.add(statusBar);
statusPanel.add(statusPostfix);

// Main column: scrollable form.
const main = new BoxRenderable(renderer, { id: "main", flexGrow: 1, flexDirection: "column" });
body.add(main);

const formPanel = new ScrollBoxRenderable(renderer, {
  id: "form",
  flexGrow: 1,
  borderStyle: BORDER.panel,
  borderColor: COLOR.borderFocus,
  title: " Welcome ",
  titleAlignment: "left",
  paddingLeft: 2,
  paddingRight: 2,
  paddingTop: 1,
  scrollY: true,
});
main.add(formPanel);
const formBody = formPanel.content;

// Output: docked scrollback log, sticky to the bottom.
const outputBox = new ScrollBoxRenderable(renderer, {
  id: "output",
  height: 7,
  flexShrink: 1,
  minHeight: 4,
  borderStyle: BORDER.panel,
  borderColor: COLOR.border,
  title: " Output · l for full logs ",
  titleAlignment: "left",
  paddingLeft: 1,
  paddingRight: 1,
  stickyScroll: true,
  stickyStart: "bottom",
});
screen.add(outputBox);

const footer = new TextRenderable(renderer, {
  id: "footer",
  height: 1,
  flexShrink: 0,
  content: `? help · l logs · ⇥ field · ⏎ run · ← → choice · esc back · ^C quit`,
  fg: COLOR.muted,
});
screen.add(footer);

const help = makeHelpOverlay(renderer);
renderer.root.add(help.box);
const logs = makeLogModal(renderer);
renderer.root.add(logs.box);

// --------------------------------------------------------------------------- //
// Mutable state
// --------------------------------------------------------------------------- //
type RunState = "idle" | "running" | "done" | "error";
let activeWorkflow: WorkflowId | null = null;
let fields: Field[] = [];
let formNodes: { id: string }[] = [];
let focusIndex = 0;
let focusedInput: InputRenderable | null = null;
let backendHandle: { kill(): void } | null = null;
let logLineId = 0;
let runState: RunState = "idle";
const fieldErrors = new Map<string, string>();
// Captured field values handed to a form builder during a rebuild so it can
// derive algo/env from what the user had selected (fields are cleared first).
let pendingPreset: Record<string, unknown> | null = null;
// Set when the user cancels a run, so a multi-map trajectory sequence stops
// instead of advancing to the next map on the killed process's exit event.
let trajectoryCancelled = false;

function menuOptions(activeId: WorkflowId | null) {
  return MAIN_MENU_OPTIONS.map((o) => ({
    name: `${o.id === activeId ? GLYPH.active : GLYPH.idle}  ${o.label}`,
    description: "",
    value: o.id,
  }));
}

// --------------------------------------------------------------------------- //
// Status / badge / progress
// --------------------------------------------------------------------------- //
function setBadge(text: string, fg: string): void {
  headerBadge.content = text;
  headerBadge.fg = fg;
}

function setStatus(line: string, fg: string): void {
  statusLine.content = line;
  statusLine.fg = fg;
}

function setProgress(bar: string, postfix: string, fg: string): void {
  statusBar.content = bar;
  statusBar.fg = fg;
  statusPostfix.content = postfix;
  statusPostfix.fg = postfix ? COLOR.value : COLOR.muted;
}

function appendLog(text: string): void {
  for (const raw of text.split("\n")) {
    outputBox.add(new TextRenderable(renderer, { id: `log-${logLineId++}`, content: raw, fg: COLOR.value }));
  }
  logs.append(text); // mirror into the full-logs modal
}

// --------------------------------------------------------------------------- //
// Form construction
// --------------------------------------------------------------------------- //
function makeRow(key: string): { row: BoxRenderable; labelText: TextRenderable } {
  const row = new BoxRenderable(renderer, {
    id: `row-${key}`,
    flexDirection: "row",
    alignItems: "center",
    height: 1,
    width: "100%",
  });
  const labelText = new TextRenderable(renderer, {
    id: `lbl-${key}`,
    content: "",
    fg: COLOR.label,
    width: LABEL_WIDTH,
  });
  row.add(labelText);
  formBody.add(row);
  formNodes.push(row);
  return { row, labelText };
}

function addChoice(key: string, label: string, options: string[], selected = 0): void {
  const { row, labelText } = makeRow(key);
  const valueText = new TextRenderable(renderer, {
    id: `val-${key}`,
    content: "",
    fg: COLOR.value,
    width: VALUE_WIDTH,
  });
  row.add(valueText);
  fields.push({
    key,
    label,
    kind: "choice",
    row,
    labelText,
    valueText,
    options,
    index: Math.min(Math.max(selected, 0), Math.max(options.length - 1, 0)),
    read: () => {
      const f = fields.find((x) => x.key === key)!;
      return f.options![f.index ?? 0];
    },
  });
}

function addInput(
  key: string,
  label: string,
  value: string,
  validate?: (raw: string) => string | null,
): void {
  const { row, labelText } = makeRow(key);
  const input = new InputRenderable(renderer, {
    id: `f-${key}`,
    width: VALUE_WIDTH,
    value,
    textColor: COLOR.value,
    cursorColor: COLOR.borderFocus,
    backgroundColor: "transparent",
    focusedBackgroundColor: COLOR.surfaceFocus,
  });
  row.add(input);
  const field: Field = { key, label, kind: "input", row, labelText, input, validate, read: () => input.value };
  fields.push(field);
  if (validate) {
    input.on(InputRenderableEvents.INPUT, () => {
      validateField(field);
      paintFocus();
      syncValidation();
      renderer.requestRender();
    });
    validateField(field);
  }
}

function addAction(
  label: string,
  key: string,
  onAction?: () => void,
  opts: { primary?: boolean } = {},
): void {
  const button = makeButton(renderer, key, label);
  formBody.add(button.box);
  formNodes.push(button.box);
  fields.push({
    key,
    label,
    kind: "action",
    row: button.box,
    labelText: button.text,
    valueText: button.text,
    button,
    primary: opts.primary,
    read: () => "run",
    onAction: onAction ?? (() => runAction()),
  });
}

function addDivider(text: string): void {
  const divider = new TextRenderable(renderer, {
    id: `div-${formNodes.length}`,
    content: `── ${text} ${"─".repeat(Math.max(0, 34 - text.length))}`,
    fg: COLOR.muted,
    marginTop: 1,
  });
  formBody.add(divider);
  formNodes.push(divider);
}

function addFormHint(text: string): void {
  const hint = new TextRenderable(renderer, {
    id: `hint-${formNodes.length}`,
    content: text,
    fg: COLOR.muted,
    marginTop: 1,
  });
  formBody.add(hint);
  formNodes.push(hint);
}

function validateField(f: Field): void {
  if (!f.validate || !f.input) return;
  const err = f.validate(f.input.value);
  if (err) fieldErrors.set(f.key, `${f.label}: ${err}`);
  else fieldErrors.delete(f.key);
}

function formValid(): boolean {
  return fieldErrors.size === 0;
}

function firstError(): string | null {
  const it = fieldErrors.values().next();
  return it.done ? null : it.value;
}

// Reflect validation in the status panel + CTA when idle (don't clobber runs).
function syncValidation(): void {
  if (runState === "running") return;
  const err = firstError();
  if (err) {
    setStatus(`${GLYPH.err} ${err}`, COLOR.err);
    setBadge("invalid", COLOR.err);
  } else if (runState === "idle") {
    setStatus(`${GLYPH.marker} ${labelFor(activeWorkflow)} ready`, COLOR.label);
    setBadge("ready", COLOR.muted);
  }
}

function labelFor(id: WorkflowId | null): string {
  return MAIN_MENU_OPTIONS.find((o) => o.id === id)?.label ?? "Workflow";
}

// --------------------------------------------------------------------------- //
// Field focus / rendering
// --------------------------------------------------------------------------- //
function renderField(f: Field, focused: boolean): void {
  if (f.kind === "action") {
    const disabled = !!f.primary && !formValid();
    const fg = disabled ? COLOR.muted : focused ? COLOR.onAccent : COLOR.action;
    const mk = focused ? `${GLYPH.marker} ` : "  "; // non-colour focus cue
    f.button!.box.borderColor = disabled ? COLOR.border : focused ? COLOR.accent : COLOR.action;
    f.valueText!.content = `${mk}${GLYPH.run} ${f.label}`;
    f.valueText!.fg = fg;
    f.valueText!.bg = focused && !disabled ? COLOR.action : undefined;
    return;
  }
  const marker = focused ? `${GLYPH.marker} ` : "  ";
  f.labelText.content = `${marker}${f.label}`.padEnd(LABEL_WIDTH);
  f.labelText.fg = focused ? COLOR.labelFocus : COLOR.label;
  if (f.kind === "choice") {
    const value = f.options![f.index ?? 0] ?? "";
    f.valueText!.content = focused ? `${GLYPH.left} ${value} ${GLYPH.right}` : `  ${value}`;
    f.valueText!.fg = focused ? COLOR.accent : COLOR.value;
  } else if (f.kind === "input" && f.input) {
    f.input.textColor = fieldErrors.has(f.key) ? COLOR.err : COLOR.value;
  }
}

function paintFocus(): void {
  fields.forEach((f, i) => renderField(f, i === focusIndex));
}

function focusField(index: number): void {
  if (fields.length === 0) return;
  focusIndex = wrap(index, fields.length);
  const f = fields[focusIndex];
  if (focusedInput && (f.kind !== "input" || f.input !== focusedInput)) {
    focusedInput.blur();
    focusedInput = null;
  }
  if (f.kind === "input" && f.input) {
    f.input.focus();
    focusedInput = f.input;
  }
  paintFocus();
  formPanel.scrollChildIntoView(f.row.id);
  renderer.requestRender();
}

function cycleChoice(delta: number): void {
  const f = fields[focusIndex];
  if (f?.kind !== "choice" || !f.options || f.options.length === 0) return;
  f.index = wrap((f.index ?? 0) + delta, f.options.length);
  renderField(f, true);
  onChoiceChanged(f);
  renderer.requestRender();
}

function readValues(): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of fields) out[f.key] = f.read();
  return out;
}

// --------------------------------------------------------------------------- //
// Run lifecycle
// --------------------------------------------------------------------------- //
let spinnerTimer: ReturnType<typeof setInterval> | null = null;
let spinnerFrame = 0;
let runLabel = "";
let lastPercent = -1;

function stopSpinner(): void {
  if (spinnerTimer) {
    clearInterval(spinnerTimer);
    spinnerTimer = null;
  }
}

function tickSpinner(): void {
  spinnerFrame = (spinnerFrame + 1) % SPINNER.length;
  setStatus(`${SPINNER[spinnerFrame]} ${runLabel}`, COLOR.running);
  setBadge(`${SPINNER[spinnerFrame]} ${lastPercent >= 0 ? `${lastPercent}%` : "running"}`, COLOR.running);
  renderer.requestRender();
}

function beginRun(label: string): void {
  trajectoryCancelled = false;
  runState = "running";
  runLabel = label;
  lastPercent = -1;
  formPanel.borderColor = COLOR.running;
  appendLog(`\n── ${label} ──`);
  setProgress("", "", COLOR.running);
  spinnerFrame = 0;
  tickSpinner();
  stopSpinner();
  spinnerTimer = setInterval(tickSpinner, 120);
}

function endRun(code: number | null, label: string): void {
  backendHandle = null;
  stopSpinner();
  if (code === 0) {
    runState = "done";
    setStatus(`${GLYPH.ok} Done: ${label}`, COLOR.ok);
    setBadge("done", COLOR.ok);
    setProgress(progressBar(100, VALUE_WIDTH - 8), "complete", COLOR.ok);
    formPanel.borderColor = COLOR.ok;
  } else {
    runState = "error";
    setStatus(`${GLYPH.err} Failed (exit ${code}) — see Output`, COLOR.err);
    setBadge("failed", COLOR.err);
    setProgress("", "", COLOR.err);
    formPanel.borderColor = COLOR.err;
  }
  if (fields.length > 0) focusField(focusIndex);
}

function startRun(command: BackendCommand, payload: unknown, label: string): void {
  if (backendHandle) return;
  beginRun(label);
  backendHandle = runBackend(command, payload, (ev: BackendEvent) => onBackendEvent(ev, label));
}

function onBackendEvent(ev: BackendEvent, label: string): void {
  switch (ev.type) {
    case "progress": {
      const info = ev.text ? parseProgress(ev.text) : null;
      if (info) {
        if (info.percent >= 0) lastPercent = info.percent;
        const pct = info.percent >= 0 ? info.percent : 0;
        const counts = info.total > 0 ? `${info.current}/${info.total}` : "";
        setProgress(
          `${progressBar(pct, VALUE_WIDTH - 8)} ${info.percent >= 0 ? `${info.percent}%` : ""}`.trim(),
          [counts, info.postfix].filter(Boolean).join("  "),
          COLOR.running,
        );
        renderer.requestRender();
      }
      break;
    }
    case "stdout":
    case "stderr":
      if (ev.text) appendLog(ev.text);
      break;
    case "result":
      appendLog(`${GLYPH.dot} result ${JSON.stringify(ev.result)}`);
      break;
    case "error":
      if (ev.text) appendLog(`${GLYPH.err} ${ev.text}`);
      break;
    case "exit":
      endRun(ev.code ?? null, label);
      break;
  }
  renderer.requestRender();
}

// --------------------------------------------------------------------------- //
// Breadcrumb
// --------------------------------------------------------------------------- //
function updateBreadcrumb(): void {
  if (!activeWorkflow) {
    breadcrumb.content = "";
    return;
  }
  const v = readValues();
  const parts = [labelFor(activeWorkflow)];
  for (const key of ["algo_name", "benchmark_name", "env_name", "map_name"]) {
    if (typeof v[key] === "string" && v[key]) parts.push(v[key] as string);
  }
  breadcrumb.content = parts.join(` ${GLYPH.sep} `);
}

// --------------------------------------------------------------------------- //
// Per-workflow form builders
// --------------------------------------------------------------------------- //
type ActionFn = () => void;
let runAction: ActionFn = () => {};

const algoNames = catalog.algorithms.map((a) => a.name);
const benchNames = catalog.benchmarks;
const vehicleIds = catalog.beamng_vehicles.map((v) => v.id);

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

function addBeamngFields(envName: string, withRandom: boolean): void {
  if (!envName.startsWith("beamng")) return;
  addDivider("beamng options");
  addChoice("map_name", "Map", [...BEAMNG_MAPS]);
  addChoice("vehicle_id", "Vehicle", vehicleIds);
  addInput("trajectory_hints", "Checkpoint hints", "0", vNonNegNumber);
  addChoice("body_orientation", "Body orientation", ["false", "true"]);
  addChoice("wheel_terrain", "Wheel terrain", ["false", "true"]);
  if (withRandom) addChoice("random_path", "Randomize path", ["false", "true"]);
}

function addAgentParamInputs(algoName: string): void {
  const algo = catalog.algorithms.find((a) => a.name === algoName);
  if (!algo) return;
  const numeric = Object.entries(algo.default_config).filter(
    ([key, val]) =>
      typeof val === "number" && key !== "n_states" && key !== "n_actions" && key !== "state_type",
  );
  if (numeric.length === 0) return;
  addDivider("hyperparameters");
  for (const [key, val] of numeric) addInput(`param:${key}`, key, String(val), vNumber);
}

function collectAgentParams(values: Record<string, unknown>): Record<string, number> {
  const params: Record<string, number> = {};
  for (const [k, v] of Object.entries(values)) {
    if (k.startsWith("param:")) params[k.slice("param:".length)] = Number(v);
  }
  return params;
}

function buildTrainForm(): void {
  const algo = (pendingPreset?.algo_name as string) ?? algoNames[0] ?? "";
  const envs = catalog.compatible_envs[algo] ?? [];
  const presetEnv = pendingPreset?.env_name as string | undefined;
  const env = presetEnv && envs.includes(presetEnv) ? presetEnv : envs[0] ?? "";
  formPanel.title = " Train an agent ";
  addChoice("algo_name", "Algorithm", algoNames, Math.max(0, algoNames.indexOf(algo)));
  addChoice("env_name", "Environment", envs, Math.max(0, envs.indexOf(env)));
  addBeamngFields(env, true);
  addAgentParamInputs(algo);
  addDivider("run");
  addInput("n_episodes", "Episodes", "500", vPosInt);
  addInput("save_path", "Save path", trainSavePath(algo, env));
  addChoice("checkpoint_policy", "Checkpoint", ["resume", "reset"]);
  addAction("Start training", "run", undefined, { primary: true });

  runAction = () => {
    const v = readValues();
    const envName = v.env_name as string;
    const state: TrainState = {
      algo_name: v.algo_name as string,
      env_name: envName,
      n_episodes: num(v.n_episodes, 500),
      save_path: (v.save_path as string) || undefined,
      agent_params: collectAgentParams(v),
      checkpoint_policy: (v.checkpoint_policy as "resume" | "reset") ?? "resume",
      beamng: envName.startsWith("beamng") ? beamngFieldsFrom(v, true) : undefined,
    };
    startRun("train", buildTrainPayload(catalog, state), "Train");
  };
}

function buildEvaluateForm(): void {
  const algo = (pendingPreset?.algo_name as string) ?? algoNames[0] ?? "";
  const envs = catalog.compatible_envs[algo] ?? [];
  const presetEnv = pendingPreset?.env_name as string | undefined;
  const env = presetEnv && envs.includes(presetEnv) ? presetEnv : envs[0] ?? "";
  formPanel.title = " Evaluate an agent ";
  addChoice("algo_name", "Algorithm", algoNames, Math.max(0, algoNames.indexOf(algo)));
  addChoice("env_name", "Environment", envs, Math.max(0, envs.indexOf(env)));
  addInput("model_path", "Model path", trainSavePath(algo, env));
  addBeamngFields(env, false);
  addInput("n_episodes", "Episodes", "10", vPosInt);
  addAction("Start evaluation", "run", undefined, { primary: true });

  runAction = () => {
    const v = readValues();
    const envName = v.env_name as string;
    const state: EvaluateState = {
      algo_name: v.algo_name as string,
      env_name: envName,
      model_path: (v.model_path as string) || undefined,
      n_episodes: num(v.n_episodes, 10),
      beamng: envName.startsWith("beamng") ? beamngFieldsFrom(v, false) : undefined,
    };
    startRun("evaluate", buildEvaluatePayload(catalog, state), "Evaluate");
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

function buildBenchmarkForm(): void {
  const algo = (pendingPreset?.algo_name as string) ?? algoNames[0] ?? "";
  formPanel.title = " Run a benchmark ";
  addChoice("benchmark_name", "Benchmark", benchNames);
  addChoice("algo_name", "Algorithm", algoNames, Math.max(0, algoNames.indexOf(algo)));
  const envs = catalog.compatible_envs[algo] ?? [];
  const presetEnv = pendingPreset?.env_name as string | undefined;
  const env = presetEnv && envs.includes(presetEnv) ? presetEnv : envs[0] ?? "";
  addChoice("env_name", "Environment", envs, Math.max(0, envs.indexOf(env)));
  addDivider("evaluation");
  addInput("seeds_text", "Seeds", "0,1,2,3,4", vSeeds);
  addInput("eval_episodes", "Eval episodes", "100", vPosInt);
  addInput("success_threshold", "Success threshold", "0.0", vNumber);
  addInput("max_episodes", "Max episodes", "2000", vPosInt);
  addInput("reward_threshold", "Reward threshold", "7.0", vNumber);
  addDivider("comparison / gridsearch");
  addInput("algos_text", "Algos (comparison)", algoNames.slice(0, 2).join(","));
  addInput("param_grid_json", "Param grid (JSON)", JSON.stringify(gridPreset(algoNames[0] ?? "")), vJson);
  addAction("Run benchmark", "run", undefined, { primary: true });

  runAction = () => {
    const v = readValues();
    const seeds = String(v.seeds_text)
      .split(",")
      .map((s) => Number(s.trim()))
      .filter((n) => Number.isFinite(n));
    const state: BenchmarkState = {
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
    startRun("benchmark", buildBenchmarkPayload(catalog, state), "Benchmark");
  };
}

function buildHumanPlayForm(): void {
  formPanel.title = " Human play (BeamNG) ";
  const sensors = ["None", "LiDAR"];
  if (catalog.environments.some((e) => e.name === "beamng_camera")) sensors.push("Camera");
  addChoice("map_name", "Map", [...BEAMNG_MAPS]);
  addChoice("vehicle_id", "Vehicle", vehicleIds);
  addChoice("sensor", "Sensor", sensors);
  addAction("Launch human play", "run", undefined, { primary: true });

  runAction = () => {
    const v = readValues();
    const state: HumanPlayState = {
      map_name: v.map_name as string,
      vehicle_id: v.vehicle_id as string,
      sensor: v.sensor as string,
    };
    startRun("human-play", buildHumanPlayPayload(state), "Human play");
  };
}

function buildTrajectoryForm(): void {
  formPanel.title = " Generate trajectories (BeamNG) ";
  addChoice("map_name", "Map", [...BEAMNG_MAPS, "all"]);
  addChoice("overwrite", "Overwrite", ["false", "true"]);
  addAction("Generate", "run", undefined, { primary: true });

  runAction = () => {
    const v = readValues();
    const overwrite = bool(v.overwrite);
    const choice = v.map_name as string;
    const maps = choice === "all" ? [...BEAMNG_MAPS] : [choice];
    runTrajectorySequence(maps, overwrite, 0);
  };
}

function runTrajectorySequence(maps: string[], overwrite: boolean, index: number): void {
  if (index === 0 && backendHandle) return; // double-start guard, mirrors startRun
  if (index >= maps.length) {
    runState = "done";
    stopSpinner();
    setStatus(`${GLYPH.ok} Done: ${maps.length} map(s)`, COLOR.ok);
    setBadge("done", COLOR.ok);
    setProgress(progressBar(100, VALUE_WIDTH - 8), "complete", COLOR.ok);
    formPanel.borderColor = COLOR.ok;
    if (fields.length > 0) focusField(focusIndex);
    renderer.requestRender();
    return;
  }
  const map = maps[index];
  const state: TrajectoryState = { map_name: map, overwrite };
  beginRun(`Trajectory ${map} (${index + 1}/${maps.length})`);
  backendHandle = runBackend("trajectory", buildTrajectoryPayload(state), (ev) => {
    if (ev.type === "progress") {
      onBackendEvent(ev, runLabel);
    } else if (ev.type === "stdout" || ev.type === "stderr") {
      if (ev.text) appendLog(ev.text);
    } else if (ev.type === "result") {
      appendLog(`${GLYPH.dot} result ${JSON.stringify(ev.result)}`);
    } else if (ev.type === "error") {
      if (ev.text) appendLog(`${GLYPH.err} ${ev.text}`);
    } else if (ev.type === "exit") {
      backendHandle = null;
      if (trajectoryCancelled) {
        trajectoryCancelled = false;
        return;
      }
      runTrajectorySequence(maps, overwrite, index + 1);
    }
    renderer.requestRender();
  });
}

// Multi-agent: a dynamic list of vehicle specs the user grows before running.
const multiSpecs: MultiSpecState[] = [];

function buildMultiTrainForm(): void {
  formPanel.title = " Multi-agent training (BeamNG) ";
  const count = new TextRenderable(renderer, {
    id: "multi-count",
    content: `Vehicles configured: ${multiSpecs.length}`,
    fg: multiSpecs.length > 0 ? COLOR.ok : COLOR.running,
  });
  formBody.add(count);
  formNodes.push(count);
  addChoice("map_name", "Map", [...BEAMNG_MAPS]);
  addChoice("random_path", "Randomize path", ["false", "true"]);
  addInput("n_episodes", "Episodes", "500", vPosInt);
  addInput("time_limit_minutes", "Time limit (min)", "0.0", vNonNegNumber);
  addChoice("checkpoint_policy", "Checkpoint", ["resume", "reset"]);
  addDivider("vehicles");
  addAction("Add vehicle", "add", () => {
    addMultiSpec();
    rebuildActiveForm();
  });
  addAction("Remove last vehicle", "remove", () => {
    if (multiSpecs.length === 0) {
      setStatus(`${GLYPH.err} No vehicle to remove`, COLOR.err);
      return;
    }
    multiSpecs.pop();
    rebuildActiveForm();
  });
  addAction("Start multi-agent training", "run", undefined, { primary: true });
  // Gate the primary CTA on having at least one vehicle; re-seeded every rebuild
  // (clearForm wipes fieldErrors, and add/remove rebuild the whole form).
  if (multiSpecs.length === 0) fieldErrors.set("_vehicles", "Add at least one vehicle");

  runAction = () => {
    if (multiSpecs.length === 0) {
      setStatus(`${GLYPH.err} Add at least one vehicle first`, COLOR.err);
      return;
    }
    const v = readValues();
    const state: MultiTrainState = {
      map_name: v.map_name as string,
      random_path: bool(v.random_path),
      n_episodes: num(v.n_episodes, 500),
      time_limit_minutes: num(v.time_limit_minutes, 0),
      checkpoint_policy: (v.checkpoint_policy as "resume" | "reset") ?? "resume",
      specs: multiSpecs,
    };
    startRun("multi-train", buildMultiTrainPayload(catalog, state), "Multi-agent training");
  };
}

function addMultiSpec(): void {
  const algo = "dqn";
  const env = (catalog.compatible_envs[algo] ?? []).find((e) => e.startsWith("beamng")) ?? "beamng";
  const index = multiSpecs.length;
  multiSpecs.push({
    algo,
    env,
    vehicle_id: "taxi",
    color: MULTI_COLORS[index % MULTI_COLORS.length],
    save_path: `outputs/multi-agents/${algo}_${env}_${index}.pth`,
    trajectory_hints: 0,
    body_orientation: false,
    wheel_terrain: false,
  });
  appendLog(`${GLYPH.dot} Added vehicle ${index}: ${algo} / ${env} (${MULTI_COLORS[index % MULTI_COLORS.length]})`);
}

const FORM_BUILDERS: Record<Exclude<WorkflowId, "quit">, () => void> = {
  train: buildTrainForm,
  evaluate: buildEvaluateForm,
  benchmark: buildBenchmarkForm,
  human_play: buildHumanPlayForm,
  trajectory: buildTrajectoryForm,
  multi_train: buildMultiTrainForm,
};

// Rebuild the active form, preserving field values that still apply.
function onChoiceChanged(f: Field): void {
  if (
    (activeWorkflow === "train" || activeWorkflow === "evaluate") &&
    (f.key === "algo_name" || f.key === "env_name")
  ) {
    rebuildActiveForm(f.key);
  }
  updateBreadcrumb();
}

function clearForm(): void {
  if (focusedInput) {
    focusedInput.blur();
    focusedInput = null;
  }
  for (const node of formNodes) formBody.remove(node.id);
  formNodes = [];
  fields = [];
  fieldErrors.clear();
  focusIndex = 0;
  formPanel.scrollTo(0);
}

function openWorkflow(id: WorkflowId, focusKey?: string): void {
  if (id === "quit") {
    shutdown();
    return;
  }
  const preset = activeWorkflow === id ? readValues() : undefined;
  activeWorkflow = id;
  menu.options = menuOptions(id);
  menu.setSelectedIndex(MAIN_MENU_OPTIONS.findIndex((o) => o.id === id));
  menu.blur(); // a workflow owns the keyboard now — stop the menu double-handling keys
  clearForm();
  runState = "idle";
  formPanel.borderColor = COLOR.borderFocus;
  pendingPreset = preset ?? null; // builders read this to keep algo/env on rebuild
  FORM_BUILDERS[id]();
  pendingPreset = null;
  if (preset) applyPreset(preset);
  addFormHint(`⇥ next field   ⏎ run the focused button   esc back`);
  for (const f of fields) if (f.kind === "input" && f.validate) validateField(f);
  setProgress("", "", COLOR.muted);
  setStatus(`${GLYPH.marker} ${labelFor(id)} ready`, COLOR.label);
  setBadge("ready", COLOR.muted);
  updateBreadcrumb();
  const target = focusKey ? Math.max(0, fields.findIndex((f) => f.key === focusKey)) : 0;
  focusField(target);
  syncValidation();
  renderer.requestRender();
}

// Re-apply previously entered values to matching fields after a rebuild.
function applyPreset(preset: Record<string, unknown>): void {
  for (const f of fields) {
    if (!(f.key in preset)) continue;
    const val = preset[f.key];
    if (f.kind === "input" && f.input && typeof val === "string") {
      f.input.value = val;
    } else if (f.kind === "choice" && f.options) {
      const idx = f.options.indexOf(String(val));
      if (idx >= 0) f.index = idx;
    }
  }
}

function rebuildActiveForm(focusKey?: string): void {
  if (activeWorkflow && activeWorkflow !== "quit") openWorkflow(activeWorkflow, focusKey);
}

// Render the README into the Welcome panel as real, scrollable markdown.
function buildWelcome(): void {
  formPanel.title = " Welcome · README ";
  formPanel.borderColor = COLOR.border;
  formPanel.scrollTop = 0;
  const md = new MarkdownRenderable(renderer, {
    id: "welcome-md",
    width: "100%",
    content: README_TEXT,
    syntaxStyle: welcomeSyntax,
    conceal: true,
    bg: "transparent",
  });
  formBody.add(md);
  formNodes.push(md);
}

function backToMenu(): void {
  activeWorkflow = null;
  clearForm();
  menu.options = menuOptions(null);
  buildWelcome();
  runState = "idle";
  setProgress("", `${catalog.algorithms.length} algos ${GLYPH.dot} ${catalog.environments.length} envs`, COLOR.muted);
  setStatus("Pick a workflow, press ⏎", COLOR.label);
  setBadge("ready", COLOR.muted);
  updateBreadcrumb();
  menu.focus();
  renderer.requestRender();
}

function shutdown(): void {
  if (backendHandle) backendHandle.kill();
  renderer.destroy();
  process.exit(0);
}

function openLogs(): void {
  logs.show();
  logs.box.focus(); // so ↑↓ / PgUp / PgDn scroll the modal
  renderer.requestRender();
}

function closeLogs(): void {
  logs.hide();
  if (activeWorkflow && fields.length > 0) focusField(focusIndex);
  else menu.focus();
  renderer.requestRender();
}

// --------------------------------------------------------------------------- //
// Keyboard routing
// --------------------------------------------------------------------------- //
menu.on(SelectRenderableEvents.ITEM_SELECTED, (_index: number, option: { value?: unknown }) => {
  openWorkflow(option.value as WorkflowId);
});

renderer.keyInput.on("keypress", (key: KeyEvent) => {
  // Help overlay swallows input while open.
  if (help.visible) {
    key.preventDefault(); // swallow everything else while the overlay is open
    if (key.name === "escape" || key.name === "?" || key.sequence === "?") help.hide();
    if (key.ctrl && key.name === "c") shutdown();
    return;
  }

  if (logs.visible) {
    if (key.name === "escape" || key.name === "l") {
      closeLogs();
      return;
    }
    if (key.ctrl && key.name === "c") {
      shutdown();
      return;
    }
    // Let scroll keys reach the focused modal; swallow the rest.
    const scrollKeys = ["up", "down", "pageup", "pagedown", "home", "end"];
    if (!scrollKeys.includes(key.name)) key.preventDefault();
    return;
  }

  if (key.ctrl && key.name === "c") {
    if (backendHandle) {
      trajectoryCancelled = true;
      backendHandle.kill();
      backendHandle = null;
      stopSpinner();
      runState = "idle";
      setStatus(`${GLYPH.err} Cancelled`, COLOR.err);
      setBadge("cancelled", COLOR.err);
      setProgress("", "", COLOR.muted);
      formPanel.borderColor = COLOR.border;
      if (fields.length > 0) focusField(focusIndex);
      renderer.requestRender();
    } else {
      shutdown();
    }
    return;
  }

  // `?` opens help unless the user is typing into a text field.
  if ((key.name === "?" || key.sequence === "?") && !focusedInput) {
    help.toggle();
    return;
  }

  if (key.name === "l" && !focusedInput) {
    openLogs();
    return;
  }

  if (key.name === "escape") {
    if (activeWorkflow !== null && !backendHandle) backToMenu();
    else if (activeWorkflow === null) menu.focus(); // recover focus (e.g. after a mouse click)
    renderer.requestRender();
    return;
  }

  if (activeWorkflow === null) {
    // Welcome screen: the menu owns ↑↓ / ⏎; PgUp / PgDn scroll the README.
    if (key.name === "pageup") {
      formPanel.scrollTop = Math.max(0, formPanel.scrollTop - 5);
      renderer.requestRender();
    } else if (key.name === "pagedown") {
      formPanel.scrollTop += 5;
      renderer.requestRender();
    }
    return; // menu handles its own arrows / enter
  }

  if (backendHandle) return; // controls locked while a run is in flight

  if (key.name === "tab") {
    focusField(focusIndex + (key.shift ? -1 : 1));
    return;
  }

  if (key.name === "left") {
    cycleChoice(-1);
    return;
  }
  if (key.name === "right") {
    cycleChoice(1);
    return;
  }

  if (key.name === "return" || key.name === "enter") {
    const f = fields[focusIndex];
    if (!f) return;
    if (f.kind === "action") {
      if (f.primary && !formValid()) {
        setStatus(`${GLYPH.err} ${firstError() ?? "fix the fields above"}`, COLOR.err);
        return;
      }
      f.onAction?.();
    } else {
      focusField(focusIndex + 1);
    }
  }
});

// --------------------------------------------------------------------------- //
// Boot
// --------------------------------------------------------------------------- //
// No imperative resize handler: the layout is fully declarative (flex + minHeight)
// so Yoga reflows on resize. OpenTUI's processResize already requests a render,
// and mutating layout / calling requestRender from a "resize" listener is
// discouraged (and a crash risk in the native resize path).

buildWelcome();
menu.focus();
setStatus("Pick a workflow, press ⏎", COLOR.label);
renderer.requestRender();

// Debug/verification affordances:
//   T_AIA_TUI_OPEN=<workflow id> opens a form on boot
//   T_AIA_TUI_LOGS=1            opens the logs viewer on boot
const autoOpen = process.env.T_AIA_TUI_OPEN as WorkflowId | undefined;
if (autoOpen) openWorkflow(autoOpen);
if (process.env.T_AIA_TUI_LOGS) {
  appendLog("Example log line — full logs render here.");
  openLogs();
}
