// Pure value helpers and field validators. No OpenTUI renderer is touched here,
// so these are trivially testable and shared across the form modules.

/** Wrap an index into [0, n). */
export const wrap = (i: number, n: number): number => ((i % n) + n) % n;

/** Coerce to a finite number, or fall back. */
export const num = (v: unknown, fallback: number): number => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};

/** Coerce a choice/string value to a boolean. */
export const bool = (v: unknown): boolean =>
  v === true || v === "true" || v === "yes" || v === "y";

// Validators return an error message, or null when the value is acceptable.

export const vNumber = (raw: string): string | null =>
  Number.isFinite(Number(raw)) && raw.trim() !== "" ? null : "must be a number";

export const vPosInt = (raw: string): string | null => {
  const n = Number(raw);
  return Number.isInteger(n) && n >= 1 ? null : "must be an integer ≥ 1";
};

export const vNonNegNumber = (raw: string): string | null => {
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? null : "must be a number ≥ 0";
};

export const vSeeds = (raw: string): string | null => {
  const parts = raw.split(",").map((s) => s.trim()).filter(Boolean);
  if (parts.length === 0) return "enter at least one seed";
  return parts.every((s) => Number.isFinite(Number(s))) ? null : "seeds must be numbers";
};

export const vJson = (raw: string): string | null => {
  try {
    JSON.parse(raw);
    return null;
  } catch {
    return "invalid JSON";
  }
};
