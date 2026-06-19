// Bridge to the Python action layer (`python -m core.tui_backend`).
// Pure line-parsing logic lives in `parseBackendLine` so tests cover the
// exact parser the app uses at runtime.

const RESULT_PREFIX = "[TUI_RESULT] ";
const ERROR_PREFIX = "[TUI_ERROR] ";

export type BackendEvent = {
  // "progress" carries a transient line repainted in place (tqdm \r updates);
  // it should drive the live status bar, not the scrolling log.
  type: "stdout" | "stderr" | "exit" | "result" | "error" | "progress";
  text?: string;
  code?: number | null;
  result?: unknown;
};

export type BackendCommand =
  | "train"
  | "evaluate"
  | "benchmark"
  | "human-play"
  | "trajectory"
  | "multi-train";

export interface CatalogAlgorithm {
  name: string;
  default_config: Record<string, unknown>;
  compatible_envs: string[] | null;
}

export interface CatalogEnvironment {
  name: string;
  metadata: Record<string, unknown>;
}

export interface Catalog {
  algorithms: CatalogAlgorithm[];
  environments: CatalogEnvironment[];
  compatible_envs: Record<string, string[]>;
  benchmarks: string[];
  beamng_maps: string[];
  beamng_vehicles: { id: string; label: string }[];
  multi_algos: string[];
}

const PYTHON = process.env.T_AIA_PYTHON ?? "python";
const REPO_ROOT = process.env.T_AIA_ROOT ?? "..";

/** Classify one line of backend stdout. */
export function parseBackendLine(line: string): BackendEvent {
  if (line.startsWith(RESULT_PREFIX)) {
    return { type: "result", result: JSON.parse(line.slice(RESULT_PREFIX.length)) };
  }
  if (line.startsWith(ERROR_PREFIX)) {
    return { type: "error", text: line.slice(ERROR_PREFIX.length) };
  }
  return { type: "stdout", text: line };
}

export async function loadCatalog(): Promise<Catalog> {
  const proc = Bun.spawn([PYTHON, "-m", "core.tui_backend", "catalog"], {
    cwd: REPO_ROOT,
    stdout: "pipe",
    stderr: "pipe",
  });
  const stdout = await new Response(proc.stdout).text();
  const stderr = await new Response(proc.stderr).text();
  const code = await proc.exited;
  if (code !== 0) {
    throw new Error(`catalog failed (exit ${code}): ${stderr}`);
  }
  return JSON.parse(stdout) as Catalog;
}

export function runBackend(
  command: BackendCommand,
  payload: unknown,
  onEvent: (event: BackendEvent) => void,
): { kill(): void } {
  const proc = Bun.spawn(
    [PYTHON, "-m", "core.tui_backend", command, "--config-json", JSON.stringify(payload)],
    { cwd: REPO_ROOT, stdout: "pipe", stderr: "pipe" },
  );

  const pump = async (stream: ReadableStream<Uint8Array>, isErr: boolean) => {
    const decoder = new TextDecoder();
    let buffer = "";
    const reader = stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = drain(buffer, isErr);
    }
    // Flush a trailing partial line (e.g. a final tqdm bar with no newline).
    if (buffer.length > 0) emitLine(buffer.replace(/\r$/, ""), isErr, false);
  };

  // Emit every complete line in `buffer`, returning the unconsumed remainder.
  // Lines end at \n, \r\n, or a lone \r — the last being tqdm repainting one
  // line in place, which we surface as a transient "progress" update.
  const drain = (buffer: string, isErr: boolean): string => {
    for (;;) {
      const rIdx = buffer.indexOf("\r");
      const nIdx = buffer.indexOf("\n");
      if (rIdx === -1 && nIdx === -1) break;

      let cut: number;
      let next: number;
      let transient = false;

      if (nIdx === -1 || (rIdx !== -1 && rIdx < nIdx)) {
        if (rIdx === buffer.length - 1) break; // maybe a split \r\n — wait for more
        if (buffer[rIdx + 1] === "\n") {
          cut = rIdx;
          next = rIdx + 2; // \r\n → permanent line
        } else {
          cut = rIdx;
          next = rIdx + 1;
          transient = true; // lone \r → transient repaint
        }
      } else {
        cut = nIdx;
        next = nIdx + 1; // \n → permanent line
      }

      emitLine(buffer.slice(0, cut), isErr, transient);
      buffer = buffer.slice(next);
    }
    return buffer;
  };

  const emitLine = (line: string, isErr: boolean, transient: boolean) => {
    const ev = parseBackendLine(line);
    // RESULT_PREFIX / ERROR_PREFIX lines win regardless of stream or transience.
    if (ev.type === "result" || ev.type === "error") {
      onEvent(ev);
      return;
    }
    if (transient) {
      onEvent({ type: "progress", text: line });
      return;
    }
    onEvent(isErr ? { type: "stderr", text: line } : { type: "stdout", text: line });
  };

  void (async () => {
    await Promise.all([pump(proc.stdout, false), pump(proc.stderr, true)]);
    const code = await proc.exited;
    onEvent({ type: "exit", code });
  })();

  return {
    kill() {
      proc.kill();
    },
  };
}
