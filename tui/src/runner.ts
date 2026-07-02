// Backend run lifecycle: spinner, progress, and the stdout/stderr/exit handling
// for a running Python job. Also drives the multi-map trajectory sequence.

import { runBackend, type BackendCommand, type BackendEvent } from "./backend.ts";
import { VALUE_WIDTH, type Ctx } from "./context.ts";
import { parseProgress, progressBar } from "./progress.ts";
import { COLOR, GLYPH, SPINNER } from "./theme.ts";
import { buildTrajectoryPayload, type TrajectoryState } from "./workflows.ts";
import { focusField } from "./form.ts";
import { appendLog, setBadge, setProgress, setStatus } from "./status.ts";

export function stopSpinner(ctx: Ctx): void {
  if (ctx.state.spinnerTimer) {
    clearInterval(ctx.state.spinnerTimer);
    ctx.state.spinnerTimer = null;
  }
}

function tickSpinner(ctx: Ctx): void {
  const { state } = ctx;
  state.spinnerFrame = (state.spinnerFrame + 1) % SPINNER.length;
  setStatus(ctx, `${SPINNER[state.spinnerFrame]} ${state.runLabel}`, COLOR.running);
  setBadge(
    ctx,
    `${SPINNER[state.spinnerFrame]} ${state.lastPercent >= 0 ? `${state.lastPercent}%` : "running"}`,
    COLOR.running,
  );
  ctx.renderer.requestRender();
}

export function beginRun(ctx: Ctx, label: string): void {
  const { state } = ctx;
  state.trajectoryCancelled = false;
  state.runState = "running";
  state.runLabel = label;
  state.lastPercent = -1;
  ctx.scene.formPanel.borderColor = COLOR.running;
  appendLog(ctx, `\n── ${label} ──`);
  appendLog(ctx, `${GLYPH.dot} press Esc, s, or Ctrl+C to stop this run`);
  setProgress(ctx, "", "", COLOR.running);
  state.spinnerFrame = 0;
  tickSpinner(ctx);
  stopSpinner(ctx);
  state.spinnerTimer = setInterval(() => tickSpinner(ctx), 120);
}

// Shared success-end: status/badge/progress/border for a completed run.
function completeRun(ctx: Ctx, message: string): void {
  ctx.state.runState = "done";
  setStatus(ctx, `${GLYPH.ok} ${message}`, COLOR.ok);
  setBadge(ctx, "done", COLOR.ok);
  setProgress(ctx, progressBar(100, VALUE_WIDTH - 8), "complete", COLOR.ok);
  ctx.scene.formPanel.borderColor = COLOR.ok;
}

// Reset the UI to a neutral idle after a cooperative stop (the backend has
// already saved checkpoints and, for most commands, closed the simulator).
function finalizeStopped(ctx: Ctx): void {
  const { state } = ctx;
  if (state.stopTimer) {
    clearTimeout(state.stopTimer);
    state.stopTimer = null;
  }
  state.stopRequested = false;
  state.runState = "idle";
  stopSpinner(ctx);
  setStatus(ctx, `${GLYPH.ok} Stopped`, COLOR.ok);
  setBadge(ctx, "stopped", COLOR.muted);
  setProgress(ctx, "", "", COLOR.muted);
  ctx.scene.formPanel.borderColor = COLOR.border;
  if (state.fields.length > 0) focusField(ctx, state.focusIndex);
}

function endRun(ctx: Ctx, code: number | null, label: string): void {
  const { state } = ctx;
  state.backendHandle = null;
  if (state.stopRequested) {
    finalizeStopped(ctx);
    return;
  }
  stopSpinner(ctx);
  if (code === 0) {
    completeRun(ctx, `Done: ${label}`);
  } else {
    state.runState = "error";
    setStatus(ctx, `${GLYPH.err} Failed (exit ${code}) — see Output`, COLOR.err);
    setBadge(ctx, "failed", COLOR.err);
    setProgress(ctx, "", "", COLOR.err);
    ctx.scene.formPanel.borderColor = COLOR.err;
  }
  if (state.fields.length > 0) focusField(ctx, state.focusIndex);
}

export function startRun(ctx: Ctx, command: BackendCommand, payload: unknown, label: string): void {
  if (ctx.state.backendHandle) return;
  beginRun(ctx, label);
  ctx.state.backendHandle = runBackend(command, payload, (ev: BackendEvent) =>
    onBackendEvent(ctx, ev, label),
  );
}

// Cooperatively stop the active run: the backend saves checkpoints and closes
// BeamNG in its `finally`, then exits (finalized on its exit event). A second
// request force-kills, in case the graceful stop hangs.
export function requestStop(ctx: Ctx): void {
  const { state } = ctx;
  const handle = state.backendHandle;
  if (!handle) return;
  if (state.stopRequested) {
    handle.kill();
    return;
  }
  state.stopRequested = true;
  state.trajectoryCancelled = true; // also halts a multi-map trajectory sequence
  setStatus(ctx, `${GLYPH.dot} Stopping…`, COLOR.running);
  setBadge(ctx, "stopping", COLOR.running);
  handle.stop();
  state.stopTimer = setTimeout(() => {
    state.backendHandle?.kill();
  }, 25000);
  ctx.renderer.requestRender();
}

export function onBackendEvent(ctx: Ctx, ev: BackendEvent, label: string): void {
  switch (ev.type) {
    case "progress": {
      const info = ev.text ? parseProgress(ev.text) : null;
      if (info) {
        if (info.percent >= 0) ctx.state.lastPercent = info.percent;
        const pct = info.percent >= 0 ? info.percent : 0;
        const counts = info.total > 0 ? `${info.current}/${info.total}` : "";
        setProgress(
          ctx,
          `${progressBar(pct, VALUE_WIDTH - 8)} ${info.percent >= 0 ? `${info.percent}%` : ""}`.trim(),
          [counts, info.postfix].filter(Boolean).join("  "),
          COLOR.running,
        );
        ctx.renderer.requestRender();
      }
      break;
    }
    case "stdout":
    case "stderr":
      if (ev.text) appendLog(ctx, ev.text);
      break;
    case "result":
      appendLog(ctx, `${GLYPH.dot} result ${JSON.stringify(ev.result)}`);
      break;
    case "error":
      if (ev.text) appendLog(ctx, `${GLYPH.err} ${ev.text}`);
      break;
    case "exit":
      endRun(ctx, ev.code ?? null, label);
      break;
  }
  ctx.renderer.requestRender();
}

export function runTrajectorySequence(
  ctx: Ctx,
  maps: string[],
  overwrite: boolean,
  index: number,
): void {
  const { state, renderer } = ctx;
  if (index === 0 && state.backendHandle) return; // double-start guard, mirrors startRun
  if (index >= maps.length) {
    stopSpinner(ctx);
    completeRun(ctx, `Done: ${maps.length} map(s)`);
    if (state.fields.length > 0) focusField(ctx, state.focusIndex);
    renderer.requestRender();
    return;
  }
  const map = maps[index];
  const trajState: TrajectoryState = { map_name: map, overwrite };
  beginRun(ctx, `Trajectory ${map} (${index + 1}/${maps.length})`);
  // Non-exit events share the generic handler; only the exit advances the
  // sequence (instead of ending the run) so all maps run back-to-back.
  state.backendHandle = runBackend("trajectory", buildTrajectoryPayload(trajState), (ev) => {
    if (ev.type !== "exit") {
      onBackendEvent(ctx, ev, state.runLabel);
      return;
    }
    state.backendHandle = null;
    if (state.trajectoryCancelled) {
      state.trajectoryCancelled = false;
      finalizeStopped(ctx);
      renderer.requestRender();
      return;
    }
    runTrajectorySequence(ctx, maps, overwrite, index + 1);
  });
}
