// Pure helpers for turning tqdm progress lines into structured data and for
// rendering a text progress bar. No OpenTUI renderer is touched here, so both
// functions are covered by plain unit tests.
//
// A tqdm line (written to stderr, repainted in place with \r) looks like:
//   "Training:  42%|████▏     | 210/500 [00:12<00:18, 15.3ep/s, reward=6.4, avg=5.9]"

import { GLYPH } from "./theme.ts";

export interface ProgressInfo {
  /** Leading description, e.g. "Training" / "Evaluating" ("" if absent). */
  label: string;
  /** 0–100, or -1 when tqdm reports no percentage yet. */
  percent: number;
  current: number;
  total: number;
  /** Formatted key=val postfix pairs, e.g. "reward 6.4  avg 5.9" ("" if none). */
  postfix: string;
}

const ANSI = /\x1b\[[0-9;]*m/g;

/**
 * Parse one tqdm progress line. Returns null when the line carries neither a
 * percentage nor an `n/total` count (i.e. it is not a progress line).
 */
export function parseProgress(line: string): ProgressInfo | null {
  const clean = line.replace(ANSI, "").trim();
  if (!clean) return null;

  const pctMatch = clean.match(/(\d+)%/);
  const countMatch = clean.match(/(\d+)\s*\/\s*(\d+)/);
  if (!pctMatch && !countMatch) return null;

  const labelMatch = clean.match(/^([A-Za-z][\w ]*?):/);
  const label = labelMatch ? labelMatch[1].trim() : "";

  const percent = pctMatch ? Math.min(100, Math.max(0, Number(pctMatch[1]))) : -1;
  const current = countMatch ? Number(countMatch[1]) : 0;
  const total = countMatch ? Number(countMatch[2]) : 0;

  // tqdm postfix pairs look like `reward=6.4, avg=5.9`. The rate token
  // (`15.3ep/s`) has no `=`, so a key=val match skips it cleanly.
  const pairs: string[] = [];
  for (const m of clean.matchAll(/([A-Za-z_]\w*)=([^,\]\s]+)/g)) {
    pairs.push(`${m[1]} ${m[2]}`);
  }

  return { label, percent, current, total, postfix: pairs.join("  ") };
}

/** Render a fixed-width progress bar string. `percent` < 0 renders empty. */
export function progressBar(percent: number, width: number): string {
  const w = Math.max(0, Math.floor(width));
  const p = Number.isFinite(percent) ? Math.min(100, Math.max(0, percent)) : 0;
  const filled = Math.round((p / 100) * w);
  return GLYPH.barFull.repeat(filled) + GLYPH.barEmpty.repeat(w - filled);
}
