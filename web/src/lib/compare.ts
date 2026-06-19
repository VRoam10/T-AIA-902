import { getIndex, getRunDetail } from "./data";
import { downsample } from "./format";
import type { RunResult } from "./types";

export type CompareRun = {
	id: string;
	label: string;
	benchmark: string | null;
	algo: string | null;
	env: string | null;
	evalReward: number | null;
	curve: number[];
};

/**
 * Extract a single representative reward curve from any result shape.
 *
 * Multi-seed runs use the per-episode mean across seeds; single runs use their
 * reward curve; comparison runs fall back to the best variant's rolling curve.
 */
function extractCurve(result: RunResult): number[] {
	if (result.per_seed && result.per_seed.length > 0) {
		const curves = result.per_seed
			.map((seed) => seed.rewards ?? [])
			.filter((c) => c.length > 0);
		if (curves.length === 0) return [];
		const length = Math.min(...curves.map((c) => c.length));
		const mean: number[] = [];
		for (let i = 0; i < length; i++) {
			const column = curves.map((c) => c[i] ?? 0);
			mean.push(column.reduce((a, b) => a + b, 0) / column.length);
		}
		return mean;
	}

	if (result.rewards && result.rewards.length > 0) {
		return result.rewards;
	}

	if (result.variants) {
		const variants = Object.values(result.variants);
		let best = variants[0];
		for (const variant of variants) {
			const reward =
				variant.aggregate.eval_mean_reward?.mean ?? Number.NEGATIVE_INFINITY;
			const bestReward =
				best?.aggregate.eval_mean_reward?.mean ?? Number.NEGATIVE_INFINITY;
			if (reward > bestReward) best = variant;
		}
		return best?.mean_rolling ?? [];
	}

	return [];
}

/**
 * Build the dataset for the compare page: one entry per run with a downsampled
 * representative reward curve and its headline eval reward.
 */
export async function getCompareRuns(): Promise<CompareRun[]> {
	const { runs } = await getIndex();
	const out: CompareRun[] = [];

	for (const entry of runs) {
		const detail = await getRunDetail(entry.id);
		const curve = detail ? downsample(extractCurve(detail.result), 300) : [];
		out.push({
			id: entry.id,
			label: `${entry.benchmark} · ${entry.algo}`,
			benchmark: entry.benchmark,
			algo: entry.algo,
			env: entry.env,
			evalReward: entry.headline_eval_reward,
			curve,
		});
	}

	return out;
}
