import type { StatSummary } from "./types";

/**
 * Format a number with a fixed number of decimals, or "—" when null/undefined.
 */
export function formatNumber(
	value: number | null | undefined,
	decimals = 2,
): string {
	if (value === null || value === undefined || Number.isNaN(value)) return "—";
	return value.toFixed(decimals);
}

/**
 * Format an aggregated stat as a `mean ± std` string, or "—" when missing.
 */
export function formatStat(
	stat: StatSummary | undefined,
	decimals = 2,
): string {
	if (!stat) return "—";
	return `${stat.mean.toFixed(decimals)} ± ${stat.std.toFixed(decimals)}`;
}

const METRIC_LABELS: Record<string, string> = {
	eval_mean_reward: "Eval reward",
	eval_success_rate: "Success rate",
	eval_mean_steps: "Eval steps",
	convergence_episode: "Convergence ep.",
	training_time_s: "Training time (s)",
	final_avg_reward: "Final avg reward",
	mean_reward: "Mean reward",
	best_reward: "Best reward",
};

/**
 * Human-readable label for a raw metric key.
 */
export function metricLabel(key: string): string {
	return METRIC_LABELS[key] ?? key;
}

/**
 * Downsample a numeric series to at most `maxPoints` evenly spaced samples.
 */
export function downsample(values: number[], maxPoints = 400): number[] {
	if (values.length <= maxPoints) return values;
	const step = values.length / maxPoints;
	const out: number[] = [];
	for (let i = 0; i < maxPoints; i++) {
		const value = values[Math.floor(i * step)];
		if (value !== undefined) out.push(value);
	}
	return out;
}
