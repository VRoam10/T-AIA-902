// The Welcome screen renders the project README as real, scrollable markdown.

import { MarkdownRenderable, RGBA, SyntaxStyle } from "@opentui/core";

import type { Ctx } from "./context.ts";
import { COLOR } from "./theme.ts";

/** Load README.md from the repo root (T_AIA_ROOT), with a fallback. */
export async function loadReadme(): Promise<string> {
  const root = process.env.T_AIA_ROOT ?? "..";
  try {
    return await Bun.file(`${root}/README.md`).text();
  } catch {
    return "# RL Pipeline\n\nREADME.md introuvable.";
  }
}

/** Tokyo Night theme for the rendered README markdown. */
export function makeWelcomeSyntax(): SyntaxStyle {
  return SyntaxStyle.fromStyles({
    default: { fg: RGBA.fromHex(COLOR.value) },
    "markup.heading": { fg: RGBA.fromHex(COLOR.accent), bold: true },
    "markup.heading.1": { fg: RGBA.fromHex(COLOR.accent), bold: true },
    "markup.heading.2": { fg: RGBA.fromHex(COLOR.action), bold: true },
    "markup.heading.3": { fg: RGBA.fromHex(COLOR.cyan), bold: true },
    "markup.bold": { fg: RGBA.fromHex(COLOR.fg), bold: true },
    "markup.italic": { fg: RGBA.fromHex(COLOR.fg), italic: true },
    "markup.list": { fg: RGBA.fromHex(COLOR.action) },
    "markup.raw": { fg: RGBA.fromHex(COLOR.cyan) },
    "markup.link": { fg: RGBA.fromHex(COLOR.accent), underline: true },
    "markup.quote": { fg: RGBA.fromHex(COLOR.muted), italic: true },
  });
}

export function buildWelcome(ctx: Ctx): void {
  const { formPanel, formBody } = ctx.scene;
  formPanel.title = " Welcome · README ";
  formPanel.borderColor = COLOR.border;
  formPanel.scrollTop = 0;
  const md = new MarkdownRenderable(ctx.renderer, {
    id: "welcome-md",
    width: "100%",
    content: ctx.readme,
    syntaxStyle: ctx.welcomeSyntax,
    conceal: true,
    bg: "transparent",
  });
  formBody.add(md);
  ctx.state.formNodes.push(md);
}
