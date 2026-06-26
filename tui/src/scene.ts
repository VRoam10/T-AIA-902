// Static scene graph: builds every renderable once and returns references the
// rest of the app mutates. No behaviour/state lives here — construction only.

import {
  BoxRenderable,
  ScrollBoxRenderable,
  TextRenderable,
  type CliRenderer,
} from "@opentui/core";

import type { Catalog } from "./backend.ts";
import { BORDER, COLOR, GLYPH } from "./theme.ts";
import {
  LOG_PREVIEW_LIMIT,
  createLogSink,
  makeHelpOverlay,
  makeLogModal,
  makePanel,
  type HelpOverlay,
  type LogModal,
  type LogSink,
} from "./widgets.ts";
import { MAIN_MENU_OPTIONS } from "./workflows.ts";

/** One workflows-menu row: a full-width box (for the selection highlight) whose
 * text wraps to a second line so long labels are never truncated. */
export interface MenuItem {
  box: BoxRenderable;
  text: TextRenderable;
}

export interface Scene {
  screen: BoxRenderable;
  breadcrumb: TextRenderable;
  headerBadge: TextRenderable;
  menuItems: MenuItem[];
  statusLine: TextRenderable;
  statusBar: TextRenderable;
  statusPostfix: TextRenderable;
  formPanel: ScrollBoxRenderable;
  formBody: ScrollBoxRenderable["content"];
  outputBox: ScrollBoxRenderable;
  outputSink: LogSink;
  help: HelpOverlay;
  logs: LogModal;
}

export function buildScene(renderer: CliRenderer, catalog: Catalog): Scene {
  const screen = new BoxRenderable(renderer, {
    id: "screen",
    width: "100%",
    height: "100%",
    flexDirection: "column",
  });
  renderer.root.add(screen);

  // Header: wordmark + breadcrumb + live status badge.
  const header = new BoxRenderable(renderer, {
    id: "header",
    height: 3,
    flexShrink: 0,
    borderStyle: BORDER.header,
    borderColor: COLOR.accent,
    flexDirection: "row",
    alignItems: "center",
    paddingLeft: 1,
    paddingRight: 1,
  });
  const wordmark = new TextRenderable(renderer, {
    id: "wordmark",
    content: `${GLYPH.logo} RL PIPELINE`,
    fg: COLOR.accent,
  });
  const breadcrumb = new TextRenderable(renderer, {
    id: "breadcrumb",
    content: "",
    fg: COLOR.muted,
    flexGrow: 1,
    marginLeft: 3,
  });
  const headerBadge = new TextRenderable(renderer, {
    id: "badge",
    content: "ready",
    fg: COLOR.muted,
  });
  header.add(wordmark);
  header.add(breadcrumb);
  header.add(headerBadge);
  screen.add(header);

  // Body: sidebar (workflows + status) | main (form).
  const body = new BoxRenderable(renderer, {
    id: "body",
    flexGrow: 1,
    flexShrink: 1,
    minHeight: 0,
    flexDirection: "row",
  });
  screen.add(body);

  const sidebar = new BoxRenderable(renderer, {
    id: "sidebar",
    width: 32,
    flexShrink: 0,
    flexDirection: "column",
  });
  body.add(sidebar);

  // Declarative responsive sizing (no resize handler): the panel grows to fill
  // the sidebar and shrinks down to a floor that still shows the whole menu.
  const workflowsPanel = makePanel(renderer, {
    id: "workflows",
    title: "Workflows",
    flexGrow: 1,
    flexShrink: 1,
    minHeight: MAIN_MENU_OPTIONS.length + 2,
  });
  sidebar.add(workflowsPanel);

  // Text-based menu: each item wraps to a 2nd line instead of truncating, so
  // long labels (e.g. "Generate trajectories (BeamNG)") stay fully readable.
  const menuItems: MenuItem[] = MAIN_MENU_OPTIONS.map((o) => {
    const box = new BoxRenderable(renderer, {
      id: `menu-${o.id}`,
      width: "100%",
      flexDirection: "row",
      flexShrink: 0,
    });
    const text = new TextRenderable(renderer, {
      id: `menu-text-${o.id}`,
      content: `${GLYPH.idle}  ${o.label}`,
      fg: COLOR.label,
      width: "100%",
      wrapMode: "word",
    });
    box.add(text);
    workflowsPanel.add(box);
    return { box, text };
  });

  const statusPanel = makePanel(renderer, {
    id: "statuspanel",
    title: "Status",
    height: 6,
    flexShrink: 1,
    minHeight: 4,
  });
  sidebar.add(statusPanel);
  const statusLine = new TextRenderable(renderer, {
    id: "status-line",
    content: "Pick a workflow, press ⏎",
    fg: COLOR.label,
  });
  const statusBar = new TextRenderable(renderer, {
    id: "status-bar",
    content: "",
    fg: COLOR.running,
  });
  const statusPostfix = new TextRenderable(renderer, {
    id: "status-postfix",
    content: `${catalog.algorithms.length} algos ${GLYPH.dot} ${catalog.environments.length} envs`,
    fg: COLOR.muted,
  });
  statusPanel.add(statusLine);
  statusPanel.add(statusBar);
  statusPanel.add(statusPostfix);

  // Main column: scrollable form.
  const main = new BoxRenderable(renderer, { id: "main", flexGrow: 1, flexDirection: "column" });
  body.add(main);

  const formPanel = new ScrollBoxRenderable(renderer, {
    id: "form",
    flexGrow: 1,
    borderStyle: BORDER.panel,
    borderColor: COLOR.borderFocus,
    title: " Welcome ",
    titleAlignment: "left",
    paddingLeft: 2,
    paddingRight: 2,
    paddingTop: 1,
    scrollY: true,
  });
  main.add(formPanel);

  // Output: docked scrollback log, sticky to the bottom.
  const outputBox = new ScrollBoxRenderable(renderer, {
    id: "output",
    height: 7,
    flexShrink: 1,
    minHeight: 4,
    borderStyle: BORDER.panel,
    borderColor: COLOR.border,
    title: " Output · l for full logs ",
    titleAlignment: "left",
    paddingLeft: 1,
    paddingRight: 1,
    stickyScroll: true,
    stickyStart: "bottom",
  });
  screen.add(outputBox);
  // Render logs through a single in-place-updated text node (see createLogSink):
  // adding one renderable per line crashes opentui.dll on long runs.
  const outputSink = createLogSink(renderer, outputBox, LOG_PREVIEW_LIMIT);

  const footer = new TextRenderable(renderer, {
    id: "footer",
    height: 1,
    flexShrink: 0,
    content: `? help · l logs · ⇥ field · ⏎ run · ← → choice · esc back · ^C quit`,
    fg: COLOR.muted,
  });
  screen.add(footer);

  const help = makeHelpOverlay(renderer);
  renderer.root.add(help.box);
  const logs = makeLogModal(renderer);
  renderer.root.add(logs.box);

  return {
    screen,
    breadcrumb,
    headerBadge,
    menuItems,
    statusLine,
    statusBar,
    statusPostfix,
    formPanel,
    formBody: formPanel.content,
    outputBox,
    outputSink,
    help,
    logs,
  };
}
