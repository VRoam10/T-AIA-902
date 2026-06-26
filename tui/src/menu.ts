// The workflows menu: paint the selection cursor and move it. Selecting an item
// is handled by the keymap (which calls the controller) to avoid a cycle.

import type { Ctx } from "./context.ts";
import { COLOR, GLYPH } from "./theme.ts";
import { wrap } from "./validators.ts";
import { MAIN_MENU_OPTIONS } from "./workflows.ts";

/** Repaint every menu row to reflect the current cursor (state.menuIndex). */
export function paintMenu(ctx: Ctx): void {
  ctx.scene.menuItems.forEach((item, i) => {
    const selected = i === ctx.state.menuIndex;
    item.box.backgroundColor = selected ? COLOR.borderMuted : undefined;
    item.text.content = `${selected ? GLYPH.active : GLYPH.idle}  ${MAIN_MENU_OPTIONS[i].label}`;
    item.text.fg = selected ? COLOR.accent : COLOR.label;
  });
  ctx.renderer.requestRender();
}

/** Move the cursor by `delta`, wrapping around the list. */
export function moveMenu(ctx: Ctx, delta: number): void {
  ctx.state.menuIndex = wrap(ctx.state.menuIndex + delta, MAIN_MENU_OPTIONS.length);
  paintMenu(ctx);
}

/** The workflow id currently under the cursor. */
export function selectedWorkflow(ctx: Ctx) {
  return MAIN_MENU_OPTIONS[ctx.state.menuIndex].id;
}
