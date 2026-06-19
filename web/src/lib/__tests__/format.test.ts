import { describe, expect, it } from "vitest";
import { downsample, formatNumber, formatStat, metricLabel } from "../format";

describe("formatNumber", () => {
	it("formats numbers with fixed decimals", () => {
		expect(formatNumber(1.2345)).toBe("1.23");
		expect(formatNumber(7, 0)).toBe("7");
	});

	it("returns a dash for null/undefined/NaN", () => {
		expect(formatNumber(null)).toBe("—");
		expect(formatNumber(undefined)).toBe("—");
		expect(formatNumber(Number.NaN)).toBe("—");
	});
});

describe("formatStat", () => {
	it("formats mean ± std", () => {
		expect(
			formatStat({ mean: 8.1, std: 0.2, ci95: 0.1, min: 7, max: 9, n: 3 }),
		).toBe("8.10 ± 0.20");
	});

	it("returns a dash when missing", () => {
		expect(formatStat(undefined)).toBe("—");
	});
});

describe("metricLabel", () => {
	it("maps known keys and falls back to the raw key", () => {
		expect(metricLabel("eval_mean_reward")).toBe("Eval reward");
		expect(metricLabel("unknown_metric")).toBe("unknown_metric");
	});
});

describe("downsample", () => {
	it("keeps short series intact", () => {
		expect(downsample([1, 2, 3], 10)).toEqual([1, 2, 3]);
	});

	it("reduces long series to at most maxPoints", () => {
		const series = Array.from({ length: 1000 }, (_, i) => i);
		expect(downsample(series, 100)).toHaveLength(100);
	});
});
