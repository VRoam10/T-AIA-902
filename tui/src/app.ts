// OpenTUI front-end for the RL pipeline — a keyboard-driven "command center".
// All RL work happens in Python via `core.tui_backend`; this file is the
// composition root: it builds the renderer, the scene, and the shared Ctx, then
// wires the feature modules together and boots. Behaviour lives in the modules:
//   scene.ts      static layout            form.ts      field engine
//   status.ts     status/log/breadcrumb    runner.ts    backend run lifecycle
//   forms.ts      per-workflow forms        welcome.ts   README markdown
//   controller.ts open/back/rebuild glue    keymap.ts    keyboard routing

import { createCliRenderer, type CliRenderer } from "@opentui/core";

import { loadCatalog, type Catalog } from "./backend.ts";
import { createState, type Ctx } from "./context.ts";
import type { WorkflowId } from "./workflows.ts";
import { buildScene } from "./scene.ts";
import { paintMenu } from "./menu.ts";
import { appendLog } from "./status.ts";
import { onChoiceChanged, openLogs, openWorkflow, rebuildActiveForm } from "./controller.ts";
import { installKeymap } from "./keymap.ts";
import { buildWelcome, loadReadme, makeWelcomeSyntax } from "./welcome.ts";

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
const readme = await loadReadme();
const scene = buildScene(renderer, catalog);

const ctx: Ctx = {
  renderer,
  catalog,
  scene,
  state: createState(),
  algoNames: catalog.algorithms.map((a) => a.name),
  benchNames: catalog.benchmarks,
  sensors: catalog.beamng_sensors,
  readme,
  welcomeSyntax: makeWelcomeSyntax(),
  // Wire the upward hooks the lower modules call (breaks the form↔controller cycle).
  onChoiceChanged: (f) => onChoiceChanged(ctx, f),
  rebuildActiveForm: (focusKey) => rebuildActiveForm(ctx, focusKey),
};

installKeymap(ctx);

// Boot. No imperative resize handler: the layout is fully declarative
// (flex + minHeight) so Yoga reflows on resize, and mutating layout / calling
// requestRender from a "resize" listener is discouraged (native-path crash risk).
buildWelcome(ctx);
paintMenu(ctx);
renderer.requestRender();

// Debug/verification affordances:
//   T_AIA_TUI_OPEN=<workflow id> opens a form on boot
//   T_AIA_TUI_LOGS=1            opens the logs viewer on boot
const autoOpen = process.env.T_AIA_TUI_OPEN as WorkflowId | undefined;
if (autoOpen) openWorkflow(ctx, autoOpen);
if (process.env.T_AIA_TUI_LOGS) {
  appendLog(ctx, "Example log line — full logs render here.");
  openLogs(ctx);
}
