// Orchestration glue: opening/closing workflows and overlays, rebuilding the
// active form, and shutdown. This is the only module that knows about all the
// others; everything below it flows one direction.

import type { Ctx, Field } from "./context.ts";
import { COLOR, GLYPH } from "./theme.ts";
import { paintMenu } from "./menu.ts";
import { MAIN_MENU_OPTIONS, type WorkflowId } from "./workflows.ts";
import {
  addFormHint,
  applyPreset,
  clearForm,
  focusField,
  readValues,
  syncValidation,
  validateField,
} from "./form.ts";
import { buildForm, refreshDerivedPaths } from "./forms.ts";
import { BREADCRUMB_KEYS, labelFor, setBadge, setProgress, setStatus, updateBreadcrumb } from "./status.ts";
import { buildWelcome } from "./welcome.ts";

export function openWorkflow(ctx: Ctx, id: WorkflowId, focusKey?: string): void {
  if (id === "quit") {
    void shutdown(ctx);
    return;
  }
  const { state } = ctx;
  const preset = state.activeWorkflow === id ? readValues(ctx) : undefined;
  state.activeWorkflow = id;
  const menuIdx = MAIN_MENU_OPTIONS.findIndex((o) => o.id === id);
  if (menuIdx >= 0) state.menuIndex = menuIdx;
  paintMenu(ctx);
  clearForm(ctx);
  state.runState = "idle";
  ctx.scene.formPanel.borderColor = COLOR.borderFocus;
  state.pendingPreset = preset ?? null; // builders read this to keep algo/env on rebuild
  buildForm(ctx, id);
  state.pendingPreset = null;
  if (preset) applyPreset(ctx, preset);
  refreshDerivedPaths(ctx); // make the save/model path reflect the resolved algo+env+beamng options
  addFormHint(ctx, `⇥ / ↑↓ field   ⏎ run the focused button   esc back`);
  for (const f of state.fields) if (f.kind === "input" && f.validate) validateField(ctx, f);
  setProgress(ctx, "", "", COLOR.muted);
  setStatus(ctx, `${GLYPH.marker} ${labelFor(id)} ready`, COLOR.label);
  setBadge(ctx, "ready", COLOR.muted);
  updateBreadcrumb(ctx);
  const target = focusKey ? Math.max(0, state.fields.findIndex((f) => f.key === focusKey)) : 0;
  focusField(ctx, target);
  syncValidation(ctx);
  ctx.renderer.requestRender();
}

export function rebuildActiveForm(ctx: Ctx, focusKey?: string): void {
  if (ctx.state.activeWorkflow && ctx.state.activeWorkflow !== "quit") {
    openWorkflow(ctx, ctx.state.activeWorkflow, focusKey);
  }
}

// A focused choice changed: rebuild train/evaluate/benchmark when algo/env
// changes (their dependent fields differ), and always refresh the breadcrumb.
export function onChoiceChanged(ctx: Ctx, f: Field): void {
  const wf = ctx.state.activeWorkflow;
  if (
    (wf === "train" || wf === "evaluate" || wf === "benchmark") &&
    (f.key === "algo_name" || f.key === "env_name")
  ) {
    rebuildActiveForm(ctx, f.key); // openWorkflow refreshes the breadcrumb itself
    return;
  }
  // Multi-agent: changing the per-vehicle algorithm refreshes its compatible env list.
  if (wf === "multi_train" && f.key === "multi_algo") {
    rebuildActiveForm(ctx, f.key);
    return;
  }
  // body orientation feeds the derived save/model path but doesn't change the
  // field set, so update the path in place rather than rebuilding.
  if ((wf === "train" || wf === "evaluate") && f.key === "body_orientation") {
    refreshDerivedPaths(ctx);
    return;
  }
  if ((BREADCRUMB_KEYS as readonly string[]).includes(f.key)) updateBreadcrumb(ctx);
}

export function backToMenu(ctx: Ctx): void {
  const { state, catalog } = ctx;
  state.activeWorkflow = null;
  clearForm(ctx);
  paintMenu(ctx);
  buildWelcome(ctx);
  state.runState = "idle";
  setProgress(
    ctx,
    "",
    `${catalog.algorithms.length} algos ${GLYPH.dot} ${catalog.environments.length} envs`,
    COLOR.muted,
  );
  setStatus(ctx, "Pick a workflow, press ⏎", COLOR.label);
  setBadge(ctx, "ready", COLOR.muted);
  updateBreadcrumb(ctx);
  ctx.renderer.requestRender();
}

export async function shutdown(ctx: Ctx): Promise<void> {
  const handle = ctx.state.backendHandle;
  if (handle) {
    // Cooperative quit: the backend saves checkpoints and closes BeamNG in its
    // finally, then exits. Wait (bounded) so the sim actually closes before we
    // tear the UI down; force-kill if it overruns.
    setStatus(ctx, `${GLYPH.dot} Closing simulator…`, COLOR.running);
    ctx.renderer.requestRender();
    handle.stop();
    await Promise.race([handle.exited, Bun.sleep(25000)]);
    handle.kill();
  }
  ctx.renderer.destroy();
  process.exit(0);
}

export function openLogs(ctx: Ctx): void {
  ctx.scene.logs.show();
  ctx.scene.logs.box.focus(); // so ↑↓ / PgUp / PgDn scroll the modal
  ctx.renderer.requestRender();
}

export function closeLogs(ctx: Ctx): void {
  ctx.scene.logs.hide();
  if (ctx.state.activeWorkflow && ctx.state.fields.length > 0) focusField(ctx, ctx.state.focusIndex);
  ctx.renderer.requestRender();
}
