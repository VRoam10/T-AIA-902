// View helpers that write into the header badge, the Status panel, the docked
// Output log, and the breadcrumb. All take the shared Ctx.

import type { Ctx } from "./context.ts";
import { COLOR, GLYPH } from "./theme.ts";
import { MAIN_MENU_OPTIONS, type WorkflowId } from "./workflows.ts";
import { readValues } from "./form.ts";

// Choice keys that appear in the header breadcrumb (changing any of them
// refreshes it; changing other choices does not).
export const BREADCRUMB_KEYS = ["algo_name", "benchmark_name", "env_name", "map_name"] as const;

export function labelFor(id: WorkflowId | null): string {
  return MAIN_MENU_OPTIONS.find((o) => o.id === id)?.label ?? "Workflow";
}

export function setBadge(ctx: Ctx, text: string, fg: string): void {
  ctx.scene.headerBadge.content = text;
  ctx.scene.headerBadge.fg = fg;
}

export function setStatus(ctx: Ctx, line: string, fg: string): void {
  ctx.scene.statusLine.content = line;
  ctx.scene.statusLine.fg = fg;
}

export function setProgress(ctx: Ctx, bar: string, postfix: string, fg: string): void {
  ctx.scene.statusBar.content = bar;
  ctx.scene.statusBar.fg = fg;
  ctx.scene.statusPostfix.content = postfix;
  ctx.scene.statusPostfix.fg = postfix ? COLOR.value : COLOR.muted;
}

export function appendLog(ctx: Ctx, text: string): void {
  // Both sinks update a single in-place text node — never one renderable per
  // line, which would crash opentui.dll on long runs (segfault → Bun exit 3).
  ctx.scene.outputSink.append(text); // docked preview (last LOG_PREVIEW_LIMIT lines)
  ctx.scene.logs.append(text); // full-logs modal
}

export function updateBreadcrumb(ctx: Ctx): void {
  if (!ctx.state.activeWorkflow) {
    ctx.scene.breadcrumb.content = "";
    return;
  }
  const v = readValues(ctx);
  const parts = [labelFor(ctx.state.activeWorkflow)];
  for (const key of BREADCRUMB_KEYS) {
    if (typeof v[key] === "string" && v[key]) parts.push(v[key] as string);
  }
  ctx.scene.breadcrumb.content = parts.join(` ${GLYPH.sep} `);
}
