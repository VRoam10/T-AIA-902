// Renderer-coupled UI factories shared by app.ts. Each returns plain OpenTUI
// renderables; stateful styling (focus, enabled, run states) is driven by the
// caller so all interaction logic stays in one place.

import {
  BoxRenderable,
  ScrollBoxRenderable,
  TextRenderable,
  type CliRenderer,
} from "@opentui/core";

import { BORDER, COLOR, GLYPH } from "./theme.ts";

type Dimension = number | "auto" | `${number}%`;

export interface PanelOptions {
  id: string;
  title?: string;
  width?: Dimension;
  height?: Dimension;
  minHeight?: Dimension;
  flexGrow?: number;
  flexShrink?: number;
  marginTop?: number;
}

/** A titled, rounded panel used for the sidebar, form, status and output. */
export function makePanel(renderer: CliRenderer, opts: PanelOptions): BoxRenderable {
  return new BoxRenderable(renderer, {
    id: opts.id,
    title: opts.title ? ` ${opts.title} ` : undefined,
    titleAlignment: "left",
    borderStyle: BORDER.panel,
    borderColor: COLOR.border,
    focusedBorderColor: COLOR.borderFocus,
    width: opts.width,
    height: opts.height,
    minHeight: opts.minHeight,
    flexGrow: opts.flexGrow,
    flexShrink: opts.flexShrink,
    marginTop: opts.marginTop,
    paddingLeft: 1,
    paddingRight: 1,
    flexDirection: "column",
  });
}

export interface Button {
  box: BoxRenderable;
  text: TextRenderable;
  label: string;
}

/** A boxed call-to-action button. Focus/disabled styling is applied by caller. */
export function makeButton(renderer: CliRenderer, id: string, label: string): Button {
  const box = new BoxRenderable(renderer, {
    id: `btnbox-${id}`,
    borderStyle: BORDER.button,
    borderColor: COLOR.action,
    alignSelf: "flex-start",
    paddingLeft: 2,
    paddingRight: 2,
    marginTop: 1,
    flexDirection: "row",
  });
  const text = new TextRenderable(renderer, {
    id: `btn-${id}`,
    content: `${GLYPH.run} ${label}`,
    fg: COLOR.action,
  });
  box.add(text);
  return { box, text, label };
}

const HELP_LINES = [
  `${GLYPH.marker} ↑ / ↓     move in the workflow menu`,
  `  ⏎          open a workflow · run the focused button`,
  `  ⇥ / ⇧ ⇥    next / previous field`,
  `  ← / →      change the focused choice`,
  `  l          open the full logs viewer`,
  `  esc        back to the menu · close this overlay`,
  `  ^C         cancel a running job · quit`,
  `  ?          toggle this help`,
];

export interface HelpOverlay {
  box: BoxRenderable;
  toggle(): boolean;
  hide(): void;
  readonly visible: boolean;
}

/** A centered, dismissible keyboard cheat-sheet overlaid on the whole UI. */
export function makeHelpOverlay(renderer: CliRenderer): HelpOverlay {
  const box = new BoxRenderable(renderer, {
    id: "help-overlay",
    position: "absolute",
    top: "25%",
    left: "22%",
    width: "56%",
    borderStyle: BORDER.panel,
    borderColor: COLOR.accent,
    backgroundColor: COLOR.bgDark,
    title: " Keyboard ",
    titleAlignment: "center",
    padding: 1,
    flexDirection: "column",
    zIndex: 1000,
    visible: false,
  });
  for (const [i, line] of HELP_LINES.entries()) {
    box.add(new TextRenderable(renderer, { id: `help-${i}`, content: line, fg: COLOR.fg }));
  }
  box.add(
    new TextRenderable(renderer, {
      id: "help-foot",
      content: `\n${GLYPH.dot} press esc or ? to close`,
      fg: COLOR.muted,
    }),
  );

  return {
    box,
    toggle() {
      box.visible = !box.visible;
      renderer.requestRender();
      return box.visible;
    },
    hide() {
      box.visible = false;
      renderer.requestRender();
    },
    get visible() {
      return box.visible;
    },
  };
}

export interface LogModal {
  box: ScrollBoxRenderable;
  append(text: string): void;
  show(): void;
  hide(): void;
  readonly visible: boolean;
}

/**
 * A large, scrollable, dismissible overlay showing the full run log. The docked
 * Output panel stays a small live preview; this is the "see everything" view.
 * Focus the box when shown so ↑↓ / PgUp / PgDn scroll it.
 */
export function makeLogModal(renderer: CliRenderer): LogModal {
  const box = new ScrollBoxRenderable(renderer, {
    id: "log-modal",
    position: "absolute",
    top: "8%",
    left: "6%",
    width: "88%",
    height: "84%",
    borderStyle: BORDER.panel,
    borderColor: COLOR.accent,
    backgroundColor: COLOR.bgDark,
    title: " Logs — ↑↓ PgUp/PgDn scroll · esc or l to close ",
    titleAlignment: "center",
    paddingLeft: 1,
    paddingRight: 1,
    zIndex: 1000,
    visible: false,
    scrollY: true,
    stickyScroll: true,
    stickyStart: "bottom",
  });
  let id = 0;
  return {
    box,
    append(text: string) {
      for (const raw of text.split("\n")) {
        box.add(new TextRenderable(renderer, { id: `lm-${id++}`, content: raw, fg: COLOR.value }));
      }
    },
    show() {
      box.visible = true;
      renderer.requestRender();
    },
    hide() {
      box.visible = false;
      renderer.requestRender();
    },
    get visible() {
      return box.visible;
    },
  };
}
