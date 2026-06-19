import { describe, expect, test } from "bun:test";

import { parseBackendLine } from "../backend.ts";

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
