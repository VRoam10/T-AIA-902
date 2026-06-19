import { ComparisonCurvesChart } from "@/components/charts/ComparisonCurvesChart";
import {
	type BarDatum,
	MetricBarChart,
} from "@/components/charts/MetricBarChart";
import { RewardBandChart } from "@/components/charts/RewardBandChart";
import { Card, MetricCard, SectionTitle } from "@/components/ui";
import { formatNumber, formatStat, metricLabel } from "@/lib/format";
import type { Aggregate, RunResult, SeedRun, VariantResult } from "@/lib/types";

const CARD_METRICS = [
	"eval_mean_reward",
	"eval_success_rate",
	"convergence_episode",
	"training_time_s",
];

function aggregateCards(aggregate: Aggregate) {
	return CARD_METRICS.filter((key) => aggregate[key]).map((key) => {
		const stat = aggregate[key];
		return (
			<MetricCard
				key={key}
				label={metricLabel(key)}
				value={stat ? stat.mean.toFixed(2) : "—"}
				hint={stat ? `± ${stat.std.toFixed(2)} (n=${stat.n})` : undefined}
			/>
		);
	});
}

function MultiSeedView({ result }: { result: RunResult }) {
	const aggregate = result.aggregate ?? {};
	const perSeed: SeedRun[] = result.per_seed ?? [];
	const curves = perSeed
		.map((seed) => seed.rewards ?? [])
		.filter((c) => c.length > 0);

	return (
		<div className="space-y-8">
			<div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
				{aggregateCards(aggregate)}
			</div>
			<div>
				<SectionTitle
					title="Reward across seeds"
					subtitle="Mean per episode with ±std band"
				/>
				<Card>
					<RewardBandChart curves={curves} />
				</Card>
			</div>
		</div>
	);
}

function SingleView({ result }: { result: RunResult }) {
	const scalarCards = CARD_METRICS.filter(
		(key) => typeof result[key] === "number",
	).map((key) => (
		<MetricCard
			key={key}
			label={metricLabel(key)}
			value={formatNumber(result[key] as number)}
		/>
	));
	const curve = result.rewards ?? [];

	return (
		<div className="space-y-8">
			<div className="grid grid-cols-2 gap-3 lg:grid-cols-4">{scalarCards}</div>
			<div>
				<SectionTitle title="Reward curve" />
				<Card>
					<RewardBandChart curves={curve.length > 0 ? [curve] : []} />
				</Card>
			</div>
		</div>
	);
}

function statValue(aggregate: Aggregate, key: string): number {
	return aggregate[key]?.mean ?? 0;
}

function ComparisonView({ result }: { result: RunResult }) {
	const variants: Record<string, VariantResult> = result.variants ?? {};
	const names = Object.keys(variants);
	const window = result.window ?? 100;

	const bars: BarDatum[] = names.map((name) => {
		const agg = variants[name]?.aggregate ?? {};
		return {
			label: name,
			value: statValue(agg, "eval_mean_reward"),
			error: agg.eval_mean_reward?.std ?? 0,
		};
	});

	return (
		<div className="space-y-8">
			<div>
				<SectionTitle
					title="Reward curves"
					subtitle="Mean rolling reward ±std per variant"
				/>
				<Card>
					<ComparisonCurvesChart variants={variants} window={window} />
				</Card>
			</div>
			<div>
				<SectionTitle title="Greedy eval reward" />
				<Card>
					<MetricBarChart data={bars} valueLabel="Eval reward" />
				</Card>
			</div>
			<div>
				<SectionTitle
					title="Per-variant metrics"
					subtitle="Mean ± std across seeds"
				/>
				<Card className="overflow-x-auto p-0">
					<table className="w-full text-sm">
						<thead className="border-b border-[var(--border-subtle)] text-left text-[var(--text-muted)]">
							<tr>
								<th className="px-4 py-2 font-medium">Variant</th>
								<th className="px-4 py-2 font-medium">Converged</th>
								<th className="px-4 py-2 font-medium">Eval reward</th>
								<th className="px-4 py-2 font-medium">Success</th>
								<th className="px-4 py-2 font-medium">Conv. ep.</th>
							</tr>
						</thead>
						<tbody>
							{names.map((name) => {
								const variant = variants[name];
								const agg = variant?.aggregate ?? {};
								return (
									<tr
										key={name}
										className="border-b border-[var(--border-subtle)] last:border-0"
									>
										<td className="px-4 py-2 font-medium">{name}</td>
										<td className="px-4 py-2 tabular-nums">
											{((variant?.converged_rate ?? 0) * 100).toFixed(0)}%
										</td>
										<td className="px-4 py-2 tabular-nums">
											{formatStat(agg.eval_mean_reward)}
										</td>
										<td className="px-4 py-2 tabular-nums">
											{formatStat(agg.eval_success_rate)}
										</td>
										<td className="px-4 py-2 tabular-nums">
											{formatStat(agg.convergence_episode)}
										</td>
									</tr>
								);
							})}
						</tbody>
					</table>
				</Card>
			</div>
		</div>
	);
}

function GridSearchView({ result }: { result: RunResult }) {
	const entries = result.entries ?? [];
	const paramNames = result.param_names ?? [];
	const top = entries.slice(0, 10);

	const bars: BarDatum[] = top.map((entry) => ({
		label: Object.values(entry.params).join(", "),
		value: entry.aggregate.eval_mean_reward?.mean ?? entry.eval_mean_reward,
		error: entry.aggregate.eval_mean_reward?.std ?? 0,
	}));

	return (
		<div className="space-y-8">
			<div>
				<SectionTitle
					title="Top configurations"
					subtitle="Greedy eval reward (mean ±std)"
				/>
				<Card>
					<MetricBarChart data={bars} valueLabel="Eval reward" />
				</Card>
			</div>
			<div>
				<SectionTitle title="Leaderboard" />
				<Card className="overflow-x-auto p-0">
					<table className="w-full text-sm">
						<thead className="border-b border-[var(--border-subtle)] text-left text-[var(--text-muted)]">
							<tr>
								<th className="px-4 py-2 font-medium">#</th>
								{paramNames.map((name) => (
									<th key={name} className="px-4 py-2 font-medium">
										{name}
									</th>
								))}
								<th className="px-4 py-2 font-medium">Eval reward</th>
								<th className="px-4 py-2 font-medium">Success</th>
							</tr>
						</thead>
						<tbody>
							{entries.map((entry, idx) => (
								<tr
									key={Object.values(entry.params).join("-")}
									className="border-b border-[var(--border-subtle)] last:border-0"
								>
									<td className="px-4 py-2 tabular-nums">{idx + 1}</td>
									{paramNames.map((name) => (
										<td key={name} className="px-4 py-2 tabular-nums">
											{String(entry.params[name])}
										</td>
									))}
									<td className="px-4 py-2 tabular-nums">
										{formatStat(entry.aggregate.eval_mean_reward)}
									</td>
									<td className="px-4 py-2 tabular-nums">
										{formatStat(entry.aggregate.eval_success_rate)}
									</td>
								</tr>
							))}
						</tbody>
					</table>
				</Card>
			</div>
		</div>
	);
}

/**
 * Render the body of a run, dispatching on the detected result shape
 * (comparison variants, grid-search entries, multi-seed aggregate, or single).
 */
export function RunBody({ result }: { result: RunResult }) {
	if (result.variants) return <ComparisonView result={result} />;
	if (result.entries) return <GridSearchView result={result} />;
	if (result.per_seed && result.aggregate)
		return <MultiSeedView result={result} />;
	return <SingleView result={result} />;
}
