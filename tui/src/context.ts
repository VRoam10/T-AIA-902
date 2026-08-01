// The shared application context: renderer + scene + catalog-derived data +
// mutable state. Every feature module operates on a `Ctx`. Dependencies flow
// one direction (controller → forms → runner → form → status); the single
// upward call (a focused field changed) is the `onChoiceChanged` hook, wired by
// the controller in app.ts.

import {
  type BoxRenderable,
  type CliRenderer,
  type InputRenderable,
  type SyntaxStyle,
  type TextRenderable,
} from "@opentui/core";

import type { BackendHandle, Catalog } from "./backend.ts";
import type { Scene } from "./scene.ts";
import type { Button } from "./widgets.ts";
import type { MultiSpecState, WorkflowId } from "./workflows.ts";

// Form field layout widths (columns). LABEL_WIDTH fits the longest label plus
// the 2-char focus marker ("▸ ") and a 2-col gutter — e.g. "target_update_freq"
// (18) + 2 + 2 = 22. Rows are one line tall, so a label that overran its column
// would word-wrap and overflow onto the next row; labels render wrapMode "none".
export const LABEL_WIDTH = 22;
export const VALUE_WIDTH = 26;

export type FieldKind = "choice" | "input" | "action";

/** One focusable form control plus how to read/validate its value. */
export interface Field {
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

export type RunState = "idle" | "running" | "done" | "error";

/** All mutable runtime state, kept in one place rather than module globals. */
export interface AppState {
  activeWorkflow: WorkflowId | null;
  menuIndex: number; // cursor position in the workflows menu
  fields: Field[];
  formNodes: { id: string }[];
  focusIndex: number;
  focusedInput: InputRenderable | null;
  backendHandle: BackendHandle | null;
  runState: RunState;
  fieldErrors: Map<string, string>;
  // Captured field values handed to a form builder during a rebuild so it can
  // derive algo/env from what the user had selected (fields are cleared first).
  pendingPreset: Record<string, unknown> | null;
  // Set when the user cancels a run, so a multi-map trajectory sequence stops
  // instead of advancing to the next map on the killed process's exit event.
  trajectoryCancelled: boolean;
  // Run lifecycle.
  spinnerTimer: ReturnType<typeof setInterval> | null;
  spinnerFrame: number;
  runLabel: string;
  lastPercent: number;
  // Cooperative stop bookkeeping: `stopRequested` marks a stop in flight so the
  // exit renders as "stopped" and a second request force-kills; `stopTimer` is
  // the force-kill fallback used if the backend never exits.
  stopRequested: boolean;
  stopTimer: Timer | null;
  // The active form's "run" handler, set by the form builder.
  runAction: () => void;
  // Multi-agent: a dynamic list of vehicle specs the user grows before running.
  multiSpecs: MultiSpecState[];
}

export function createState(): AppState {
  return {
    activeWorkflow: null,
    menuIndex: 0,
    fields: [],
    formNodes: [],
    focusIndex: 0,
    focusedInput: null,
    backendHandle: null,
    runState: "idle",
    fieldErrors: new Map<string, string>(),
    pendingPreset: null,
    trajectoryCancelled: false,
    spinnerTimer: null,
    spinnerFrame: 0,
    runLabel: "",
    lastPercent: -1,
    stopRequested: false,
    stopTimer: null,
    runAction: () => {},
    multiSpecs: [],
  };
}

export interface Ctx {
  renderer: CliRenderer;
  catalog: Catalog;
  scene: Scene;
  state: AppState;
  // Catalog-derived, computed once.
  algoNames: string[];
  benchNames: string[];
  sensors: string[];
  // Welcome screen.
  readme: string;
  welcomeSyntax: SyntaxStyle;
  // Upward hooks wired by the controller in app.ts (break the form↔controller cycle).
  onChoiceChanged: (f: Field) => void;
  rebuildActiveForm: (focusKey?: string) => void;
}
