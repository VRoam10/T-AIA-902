// The form field engine: build focusable rows (choice / input / action),
// validate them, render focus state, and move focus. Operates on the shared Ctx.

import { BoxRenderable, InputRenderable, InputRenderableEvents, TextRenderable } from "@opentui/core";

import { LABEL_WIDTH, VALUE_WIDTH, type Ctx, type Field } from "./context.ts";
import { COLOR, GLYPH } from "./theme.ts";
import { makeButton } from "./widgets.ts";
import { labelFor, setBadge, setStatus } from "./status.ts";
import { wrap } from "./validators.ts";

function makeRow(ctx: Ctx, key: string): { row: BoxRenderable; labelText: TextRenderable } {
  const { renderer, scene, state } = ctx;
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
    flexShrink: 0,
    // Never wrap: a wrapped label measures 2 lines tall and overflows the
    // height-1 row onto the next field (the "stacked options" bug).
    wrapMode: "none",
  });
  row.add(labelText);
  scene.formBody.add(row);
  state.formNodes.push(row);
  return { row, labelText };
}

export function addChoice(ctx: Ctx, key: string, label: string, options: string[], selected = 0): void {
  const { renderer, state } = ctx;
  const { row, labelText } = makeRow(ctx, key);
  const valueText = new TextRenderable(renderer, {
    id: `val-${key}`,
    content: "",
    fg: COLOR.value,
    // Fill the rest of the row so long choice values (e.g.
    // "beamng_continuous_predicted") stay readable; never wrap (see makeRow).
    flexGrow: 1,
    flexShrink: 1,
    wrapMode: "none",
  });
  row.add(valueText);
  const field: Field = {
    key,
    label,
    kind: "choice",
    row,
    labelText,
    valueText,
    options,
    index: Math.min(Math.max(selected, 0), Math.max(options.length - 1, 0)),
    read: () => options[field.index ?? 0],
  };
  state.fields.push(field);
}

export function addInput(
  ctx: Ctx,
  key: string,
  label: string,
  value: string,
  validate?: (raw: string) => string | null,
  onInput?: () => void,
): void {
  const { renderer, state } = ctx;
  const { row, labelText } = makeRow(ctx, key);
  const input = new InputRenderable(renderer, {
    id: `f-${key}`,
    // Grow to fill the row (basis VALUE_WIDTH) so long values like a derived
    // save path stay readable instead of scrolling out of a narrow box.
    width: VALUE_WIDTH,
    flexGrow: 1,
    flexShrink: 1,
    value,
    textColor: COLOR.value,
    cursorColor: COLOR.borderFocus,
    backgroundColor: "transparent",
    focusedBackgroundColor: COLOR.surfaceFocus,
  });
  row.add(input);
  const field: Field = { key, label, kind: "input", row, labelText, input, validate, read: () => input.value };
  state.fields.push(field);
  if (validate || onInput) {
    input.on(InputRenderableEvents.INPUT, () => {
      if (validate) {
        validateField(ctx, field);
        paintFocus(ctx);
        syncValidation(ctx);
      }
      onInput?.();
      renderer.requestRender();
    });
    if (validate) validateField(ctx, field);
  }
}

export function addAction(
  ctx: Ctx,
  label: string,
  key: string,
  onAction?: () => void,
  opts: { primary?: boolean } = {},
): void {
  const { renderer, state } = ctx;
  const button = makeButton(renderer, key, label);
  ctx.scene.formBody.add(button.box);
  state.formNodes.push(button.box);
  state.fields.push({
    key,
    label,
    kind: "action",
    row: button.box,
    labelText: button.text,
    valueText: button.text,
    button,
    primary: opts.primary,
    read: () => "run",
    onAction: onAction ?? (() => ctx.state.runAction()),
  });
}

export function addDivider(ctx: Ctx, text: string): void {
  const { renderer, state } = ctx;
  const divider = new TextRenderable(renderer, {
    id: `div-${state.formNodes.length}`,
    content: `── ${text} ${"─".repeat(Math.max(0, 34 - text.length))}`,
    fg: COLOR.muted,
    marginTop: 1,
    wrapMode: "none",
  });
  ctx.scene.formBody.add(divider);
  state.formNodes.push(divider);
}

export function addFormHint(ctx: Ctx, text: string): void {
  const { renderer, state } = ctx;
  const hint = new TextRenderable(renderer, {
    id: `hint-${state.formNodes.length}`,
    content: text,
    fg: COLOR.muted,
    marginTop: 1,
    wrapMode: "none",
  });
  ctx.scene.formBody.add(hint);
  state.formNodes.push(hint);
}

export function validateField(ctx: Ctx, f: Field): void {
  if (!f.validate || !f.input) return;
  const err = f.validate(f.input.value);
  if (err) ctx.state.fieldErrors.set(f.key, `${f.label}: ${err}`);
  else ctx.state.fieldErrors.delete(f.key);
}

export function formValid(ctx: Ctx): boolean {
  return ctx.state.fieldErrors.size === 0;
}

export function firstError(ctx: Ctx): string | null {
  const it = ctx.state.fieldErrors.values().next();
  return it.done ? null : it.value;
}

// Reflect validation in the status panel + CTA when idle (don't clobber runs).
export function syncValidation(ctx: Ctx): void {
  if (ctx.state.runState === "running") return;
  const err = firstError(ctx);
  if (err) {
    setStatus(ctx, `${GLYPH.err} ${err}`, COLOR.err);
    setBadge(ctx, "invalid", COLOR.err);
  } else if (ctx.state.runState === "idle") {
    setStatus(ctx, `${GLYPH.marker} ${labelFor(ctx.state.activeWorkflow)} ready`, COLOR.label);
    setBadge(ctx, "ready", COLOR.muted);
  }
}

export function renderField(ctx: Ctx, f: Field, focused: boolean): void {
  if (f.kind === "action") {
    const disabled = !!f.primary && !formValid(ctx);
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
    f.input.textColor = ctx.state.fieldErrors.has(f.key) ? COLOR.err : COLOR.value;
  }
}

export function paintFocus(ctx: Ctx): void {
  ctx.state.fields.forEach((f, i) => renderField(ctx, f, i === ctx.state.focusIndex));
}

export function focusField(ctx: Ctx, index: number): void {
  const { state, scene, renderer } = ctx;
  if (state.fields.length === 0) return;
  state.focusIndex = wrap(index, state.fields.length);
  const f = state.fields[state.focusIndex];
  if (state.focusedInput && (f.kind !== "input" || f.input !== state.focusedInput)) {
    state.focusedInput.blur();
    state.focusedInput = null;
  }
  if (f.kind === "input" && f.input) {
    f.input.focus();
    state.focusedInput = f.input;
  }
  paintFocus(ctx);
  scene.formPanel.scrollChildIntoView(f.row.id);
  renderer.requestRender();
}

export function cycleChoice(ctx: Ctx, delta: number): void {
  const { state, renderer } = ctx;
  const f = state.fields[state.focusIndex];
  if (f?.kind !== "choice" || !f.options || f.options.length === 0) return;
  f.index = wrap((f.index ?? 0) + delta, f.options.length);
  renderField(ctx, f, true);
  ctx.onChoiceChanged(f);
  renderer.requestRender();
}

export function readValues(ctx: Ctx): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of ctx.state.fields) out[f.key] = f.read();
  return out;
}

export function clearForm(ctx: Ctx): void {
  const { state, scene } = ctx;
  if (state.focusedInput) {
    state.focusedInput.blur();
    state.focusedInput = null;
  }
  for (const node of state.formNodes) scene.formBody.remove(node.id);
  state.formNodes = [];
  state.fields = [];
  state.fieldErrors.clear();
  state.focusIndex = 0;
  scene.formPanel.scrollTo(0);
}

// Fields whose value is derived from the chosen algorithm + environment. On a
// rebuild (triggered by changing algo/env) the freshly-built default must win,
// so we DON'T restore the old value — otherwise the path keeps the previous
// model's name forever (the "save path doesn't follow the model" bug).
const DERIVED_FROM_ALGO_ENV = new Set(["save_path", "model_path"]);

// Re-apply previously entered values to matching fields after a rebuild.
export function applyPreset(ctx: Ctx, preset: Record<string, unknown>): void {
  for (const f of ctx.state.fields) {
    if (!(f.key in preset)) continue;
    if (DERIVED_FROM_ALGO_ENV.has(f.key)) continue;
    const val = preset[f.key];
    if (f.kind === "input" && f.input && typeof val === "string") {
      f.input.value = val;
    } else if (f.kind === "choice" && f.options) {
      const idx = f.options.indexOf(String(val));
      if (idx >= 0) f.index = idx;
    }
  }
}
