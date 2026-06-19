/**
 * Shared types for benchmark data consumed by the dashboard.
 *
 * These mirror the JSON emitted by the Python pipeline (core/benchmark_index.py,
 * core/base_benchmark.py). Heterogeneous result shapes are modelled with
 * optional fields so a single RunResult type covers every benchmark.
 */

export type IndexEntry = {
	id: string;
	benchmark: string | null;
	algo: string | null;
	env: string | null;
	seeds: number[] | null;
	n_seeds: number | null;
	git_commit: string | null;
	device: string | null;
	multiseed: boolean;
	headline_eval_reward: number | null;
	path: string;
};

export type BenchmarkIndex = {
	runs: IndexEntry[];
};

export type StatSummary = {
	mean: number;
	std: number;
	ci95: number;
	min: number;
	max: number;
	n: number;
};

export type Aggregate = Record<string, StatSummary>;

export type RunMetadata = {
	git_commit?: string;
	python?: string;
	platform?: string;
	numpy?: string;
	torch?: string | null;
	device?: string;
	benchmark?: string;
	algo?: string;
	env?: string;
	seeds?: number[];
	n_seeds?: number;
	seed?: number;
};

export type SeedRun = {
	rewards?: number[];
	rolling_avgs?: number[];
	[key: string]: unknown;
};

export type VariantResult = {
	aggregate: Aggregate;
	converged_rate: number;
	n_seeds: number;
	mean_rolling: number[];
	std_rolling: number[];
};

export type GridEntry = {
	params: Record<string, number | string>;
	aggregate: Aggregate;
	eval_mean_reward: number;
};

/**
 * Union of every benchmark result shape. The presence of `aggregate`,
 * `variants` or `entries` discriminates the kind at runtime.
 */
export type RunResult = {
	seeds?: number[];
	n_seeds?: number;
	window?: number;
	threshold?: number;
	max_episodes?: number;
	aggregate?: Aggregate;
	per_seed?: SeedRun[];
	rewards?: number[];
	rolling_avgs?: number[];
	variants?: Record<string, VariantResult>;
	entries?: GridEntry[];
	param_names?: string[];
	best?: GridEntry | null;
	eval_mean_reward?: number;
	[key: string]: unknown;
};

export type RunDetail = {
	entry: IndexEntry;
	metadata: RunMetadata | null;
	result: RunResult;
};
