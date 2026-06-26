import { describe, expect, test } from "bun:test";

import { parseBackendLine, splitStreamLines } from "../backend.ts";

describe("parseBackendLine", () => {
  test("parses a result line into structured JSON", () => {
    const ev = parseBackendLine('[TUI_RESULT] {"status":"ok"}');
    expect(ev.type).toBe("result");
    expect(ev.result).toEqual({ status: "ok" });
  });

  test("parses an error line", () => {
    const ev = parseBackendLine("[TUI_ERROR] FileNotFoundError: outputs/x.pth");
    expect(ev.type).toBe("error");
    expect(ev.text).toBe("FileNotFoundError: outputs/x.pth");
  });

  test("forwards ordinary stdout unchanged", () => {
    const ev = parseBackendLine("Training: 42%|####");
    expect(ev.type).toBe("stdout");
    expect(ev.text).toBe("Training: 42%|####");
  });
});

describe("splitStreamLines", () => {
  test("splits \\n-terminated lines, keeps the unterminated tail in rest", () => {
    const { lines, rest } = splitStreamLines("a\nb\nc");
    expect(lines).toEqual([
      { text: "a", transient: false },
      { text: "b", transient: false },
    ]);
    expect(rest).toBe("c");
  });

  test("treats \\r\\n as one permanent line break (not a transient)", () => {
    const { lines, rest } = splitStreamLines("done\r\nnext\r\n");
    expect(lines).toEqual([
      { text: "done", transient: false },
      { text: "next", transient: false },
    ]);
    expect(rest).toBe("");
  });

  test("marks a lone \\r as a transient repaint (tqdm)", () => {
    const { lines, rest } = splitStreamLines("Training: 10%\rTraining: 20%\r");
    // The trailing lone \r is held back in case the next chunk makes it \r\n.
    expect(lines).toEqual([{ text: "Training: 10%", transient: true }]);
    expect(rest).toBe("Training: 20%\r");
  });

  test("defers a trailing lone \\r so a split \\r\\n is not mis-classified", () => {
    // First chunk ends mid-CRLF.
    const first = splitStreamLines("line\r");
    expect(first.lines).toEqual([]);
    expect(first.rest).toBe("line\r");
    // Next chunk arrives with the \n — now it is one permanent line.
    const second = splitStreamLines(first.rest + "\nmore");
    expect(second.lines).toEqual([{ text: "line", transient: false }]);
    expect(second.rest).toBe("more");
  });

  test("mixes a transient repaint then a permanent newline", () => {
    const { lines, rest } = splitStreamLines("p10\rp20\rDone\n");
    expect(lines).toEqual([
      { text: "p10", transient: true },
      { text: "p20", transient: true },
      { text: "Done", transient: false },
    ]);
    expect(rest).toBe("");
  });

  test("returns no lines when the buffer has no terminator yet", () => {
    const { lines, rest } = splitStreamLines("partial");
    expect(lines).toEqual([]);
    expect(rest).toBe("partial");
  });
});
