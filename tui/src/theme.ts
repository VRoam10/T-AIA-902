// Tokyo Night design tokens for the TUI. Pure data — no OpenTUI renderer is
// touched here, so the palette can be imported anywhere (incl. tests).

/** Semantic colour tokens. Component code references these, never raw hex. */
export const COLOR = {
  // surfaces
  bg: "#1a1b26",
  bgDark: "#16161e",
  surface: "#20222e",
  surfaceFocus: "#2a2e3e",
  // borders
  border: "#3a3f4b",
  borderMuted: "#292e42",
  borderFocus: "#7aa2f7",
  // text
  fg: "#c0caf5",
  value: "#a9b1d6",
  label: "#9aa5b1",
  labelFocus: "#c0caf5",
  muted: "#888fb0",
  onAccent: "#1a1b26",
  // accents / state
  accent: "#7aa2f7",
  cyan: "#7dcfff",
  ok: "#9ece6a",
  err: "#f7768e",
  running: "#e0af68",
  action: "#bb9af7",
} as const;

/** Single-codepoint glyphs used across the UI (consistent icon family). */
export const GLYPH = {
  active: "●",
  idle: "○",
  marker: "▸",
  run: "▶",
  ok: "✓",
  err: "✗",
  dot: "·",
  sep: "›",
  left: "‹",
  right: "›",
  barFull: "▓",
  barEmpty: "░",
  logo: "◢◤",
} as const;

/** Braille spinner frames (advanced by a timer while a run is in flight). */
export const SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"] as const;

/** Border styles by role. */
export const BORDER = {
  panel: "rounded",
  header: "double",
  button: "rounded",
} as const;
