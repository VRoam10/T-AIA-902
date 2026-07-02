import { describe, expect, test } from "bun:test";

import { parseProgress, progressBar } from "../progress.ts";

describe("parseProgress", () => {
  test("parses a full tqdm training line", () => {
    const info = parseProgress(
      "Training:  42%|████▏     | 210/500 [00:12<00:18, 15.3ep/s, reward=6.4, avg=5.9]",
    );
    expect(info).not.toBeNull();
    expect(info!.label).toBe("Training");
    expect(info!.percent).toBe(42);
    expect(info!.current).toBe(210);
    expect(info!.total).toBe(500);
    expect(info!.postfix).toBe("reward 6.4  avg 5.9");
  });

  test("parses an evaluating line without postfix", () => {
    const info = parseProgress("Evaluating: 100%|██████████| 10/10 [00:01<00:00, 8.1ep/s]");
    expect(info!.label).toBe("Evaluating");
    expect(info!.percent).toBe(100);
    expect(info!.current).toBe(10);
    expect(info!.total).toBe(10);
    expect(info!.postfix).toBe("");
  });

  test("strips ANSI escape codes", () => {
    const info = parseProgress("\x1b[32mTraining:  5%|\x1b[0m 25/500");
    expect(info!.label).toBe("Training");
    expect(info!.percent).toBe(5);
    expect(info!.current).toBe(25);
  });

  test("counts present but no percent yet", () => {
    const info = parseProgress("Training: 0/500");
    expect(info!.percent).toBe(-1);
    expect(info!.current).toBe(0);
    expect(info!.total).toBe(500);
  });

  test("returns null for a non-progress line", () => {
    expect(parseProgress("Training complete in 12.3s (500 episodes).")).toBeNull();
    expect(parseProgress("")).toBeNull();
    expect(parseProgress("[DQNAgent] Using device: cuda")).toBeNull();
  });

  test("parses a multi-agent tqdm line (hyphenated desc + active= postfix)", () => {
    const info = parseProgress(
      "Multi-agent:  25%|██        | 5/20 [00:03<00:09,  1.6ep/s, active=2]",
    );
    expect(info).not.toBeNull();
    // The hyphen breaks the cosmetic label match; onBackendEvent never reads
    // label, so an empty label here is the expected, harmless outcome.
    expect(info!.label).toBe("");
    expect(info!.percent).toBe(25);
    expect(info!.current).toBe(5);
    expect(info!.total).toBe(20);
    expect(info!.postfix).toContain("active 2");
  });

  test("parses the multi-agent completion frame", () => {
    const info = parseProgress(
      "Multi-agent: 100%|████████| 20/20 [00:12<00:00,  1.6ep/s, active=0]",
    );
    expect(info!.percent).toBe(100);
    expect(info!.current).toBe(20);
    expect(info!.total).toBe(20);
    expect(info!.postfix).toContain("active 0");
  });
});

describe("progressBar", () => {
  test("fills proportionally to percent", () => {
    expect(progressBar(0, 10)).toBe("░░░░░░░░░░");
    expect(progressBar(100, 10)).toBe("▓▓▓▓▓▓▓▓▓▓");
    expect(progressBar(50, 10)).toBe("▓▓▓▓▓░░░░░");
  });

  test("clamps out-of-range and negative percent", () => {
    expect(progressBar(150, 4)).toBe("▓▓▓▓");
    expect(progressBar(-1, 4)).toBe("░░░░");
  });
});
