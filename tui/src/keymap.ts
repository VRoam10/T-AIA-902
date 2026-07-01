// Keyboard routing: the single global keypress handler plus the menu selection
// wiring. All decisions read/write the shared Ctx and delegate to the modules.

import { type KeyEvent } from "@opentui/core";

import type { Ctx } from "./context.ts";
import { COLOR, GLYPH } from "./theme.ts";
import { cycleChoice, firstError, focusField, formValid } from "./form.ts";
import { moveMenu, selectedWorkflow } from "./menu.ts";
import { stopSpinner } from "./runner.ts";
import { setBadge, setProgress, setStatus } from "./status.ts";
import { backToMenu, closeLogs, openLogs, openWorkflow, shutdown } from "./controller.ts";

const SCROLL_KEYS = ["up", "down", "pageup", "pagedown", "home", "end"];

export function installKeymap(ctx: Ctx): void {
  const { renderer, scene, state } = ctx;

  renderer.keyInput.on("keypress", (key: KeyEvent) => {
    // Help overlay swallows input while open.
    if (scene.help.visible) {
      key.preventDefault();
      if (key.name === "escape" || key.name === "?" || key.sequence === "?") scene.help.hide();
      if (key.ctrl && key.name === "c") shutdown(ctx);
      return;
    }

    if (scene.logs.visible) {
      if (key.name === "escape" || key.name === "l") {
        closeLogs(ctx);
        return;
      }
      if (key.ctrl && key.name === "c") {
        shutdown(ctx);
        return;
      }
      // Let scroll keys reach the focused modal; swallow the rest.
      if (!SCROLL_KEYS.includes(key.name)) key.preventDefault();
      return;
    }

    if (key.ctrl && key.name === "c") {
      if (state.backendHandle) {
        state.trajectoryCancelled = true;
        state.backendHandle.kill();
        state.backendHandle = null;
        stopSpinner(ctx);
        state.runState = "idle";
        setStatus(ctx, `${GLYPH.err} Cancelled`, COLOR.err);
        setBadge(ctx, "cancelled", COLOR.err);
        setProgress(ctx, "", "", COLOR.muted);
        scene.formPanel.borderColor = COLOR.border;
        if (state.fields.length > 0) focusField(ctx, state.focusIndex);
        renderer.requestRender();
      } else {
        shutdown(ctx);
      }
      return;
    }

    // `?` opens help unless the user is typing into a text field.
    if ((key.name === "?" || key.sequence === "?") && !state.focusedInput) {
      scene.help.toggle();
      return;
    }

    if (key.name === "l" && !state.focusedInput) {
      openLogs(ctx);
      return;
    }

    if (key.name === "escape") {
      if (state.activeWorkflow !== null && !state.backendHandle) backToMenu(ctx);
      renderer.requestRender();
      return;
    }

    if (state.activeWorkflow === null) {
      // Welcome screen: ↑↓ move the menu cursor, ⏎ opens it, PgUp/PgDn scroll README.
      if (key.name === "up") moveMenu(ctx, -1);
      else if (key.name === "down") moveMenu(ctx, 1);
      else if (key.name === "return" || key.name === "enter") openWorkflow(ctx, selectedWorkflow(ctx));
      else if (key.name === "pageup") {
        scene.formPanel.scrollTop = Math.max(0, scene.formPanel.scrollTop - 5);
        renderer.requestRender();
      } else if (key.name === "pagedown") {
        scene.formPanel.scrollTop += 5;
        renderer.requestRender();
      }
      return;
    }

    if (state.backendHandle) return; // controls locked while a run is in flight

    if (key.name === "tab") {
      focusField(ctx, state.focusIndex + (key.shift ? -1 : 1));
      return;
    }

    // ↑ / ↓ move between fields too (faster than reaching for ⇥).
    if (key.name === "up") {
      focusField(ctx, state.focusIndex - 1);
      return;
    }
    if (key.name === "down") {
      focusField(ctx, state.focusIndex + 1);
      return;
    }

    if (key.name === "left") {
      cycleChoice(ctx, -1);
      return;
    }
    if (key.name === "right") {
      cycleChoice(ctx, 1);
      return;
    }

    if (key.name === "return" || key.name === "enter") {
      const f = state.fields[state.focusIndex];
      if (!f) return;
      if (f.kind === "action") {
        if (f.primary && !formValid(ctx)) {
          setStatus(ctx, `${GLYPH.err} ${firstError(ctx) ?? "fix the fields above"}`, COLOR.err);
          return;
        }
        f.onAction?.();
      } else {
        focusField(ctx, state.focusIndex + 1);
      }
    }
  });
}
