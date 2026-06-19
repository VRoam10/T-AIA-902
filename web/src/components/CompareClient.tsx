"use client";

import { useMemo, useState } from "react";
import {
	CartesianGrid,
	Line,
	LineChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import {
	type BarDatum,
	MetricBarChart,
} from "@/components/charts/MetricBarChart";
import { Card, SectionTitle } from "@/components/ui";
import type { CompareRun } from "@/lib/compare";

const PALETTE = [
	"#3b82f6",
	"#f97316",
	"#10b981",
	"#a855f7",
	"#ef4444",
	"#14b8a6",
];

type Row = Record<string, number>;

function buildRows(selected: CompareRun[]): Row[] {
	const maxLength = Math.max(0, ...selected.map((run) => run.curve.length));
	const rows: Row[] = [];
	for (let i = 0; i < maxLength; i++) {
		const row: Row = { step: i };
		for (const run of selected) {
			const value = run.curve[i];
			if (value !== undefined) row[run.id] = value;
		}
		rows.push(row);
	}
	return rows;
}

/**
 * Interactive multi-run comparison: toggle runs to overlay their reward curves
 * and compare headline eval rewards.
 *
 * @param runs - Every available run with a representative curve and eval reward.
 */
export function CompareClient({ runs }: { runs: CompareRun[] }) {
	const [selected, setSelected] = useState<string[]>(() =>
		runs.slice(0, 4).map((r) => r.id),
	);

	const selectedRuns = useMemo(
		() => runs.filter((run) => selected.includes(run.id)),
		[runs, selected],
	);
	const rows = useMemo(() => buildRows(selectedRuns), [selectedRuns]);
	const bars: BarDatum[] = selectedRuns.map((run) => ({
		label: run.algo ?? run.id,
		value: run.evalReward ?? 0,
	}));

	const colorOf = (id: string) =>
		PALETTE[selected.indexOf(id) % PALETTE.length];

	const toggle = (id: string) =>
		setSelected((current) =>
			current.includes(id) ? current.filter((x) => x !== id) : [...current, id],
		);

	if (runs.length === 0) {
		return (
			<Card>
				<p className="text-[var(--text-secondary)]">No runs to compare yet.</p>
			</Card>
		);
	}

	return (
		<div className="space-y-8">
			<div className="flex flex-wrap gap-2">
				{runs.map((run) => {
					const active = selected.includes(run.id);
					return (
						<button
							key={run.id}
							type="button"
							onClick={() => toggle(run.id)}
							className={`rounded-full border px-3 py-1 text-xs transition-all ${
								active
									? "border-transparent text-white"
									: "border-[var(--border)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
							}`}
							style={active ? { background: colorOf(run.id) } : undefined}
						>
							{run.label}
						</button>
					);
				})}
			</div>

			<div>
				<SectionTitle
					title="Reward curves"
					subtitle="Representative reward per episode"
				/>
				<Card>
					{rows.length === 0 ? (
						<p className="text-sm text-[var(--text-muted)]">
							Select at least one run.
						</p>
					) : (
						<ResponsiveContainer width="100%" height={360}>
							<LineChart
								data={rows}
								margin={{ top: 10, right: 12, left: 0, bottom: 4 }}
							>
								<CartesianGrid
									strokeDasharray="3 3"
									stroke="var(--grid-line)"
									vertical={false}
								/>
								<XAxis
									dataKey="step"
									tickLine={false}
									axisLine={false}
									tick={{ fill: "var(--text-muted)", fontSize: 11 }}
								/>
								<YAxis
									tickLine={false}
									axisLine={false}
									width={48}
									tick={{ fill: "var(--text-muted)", fontSize: 11 }}
								/>
								<Tooltip
									contentStyle={{
										background: "var(--bg-base)",
										border: "1px solid var(--border)",
										borderRadius: 8,
										fontSize: 12,
									}}
								/>
								{selectedRuns.map((run) => (
									<Line
										key={run.id}
										dataKey={run.id}
										name={run.label}
										stroke={colorOf(run.id)}
										strokeWidth={2}
										dot={false}
										connectNulls
										isAnimationActive={false}
									/>
								))}
							</LineChart>
						</ResponsiveContainer>
					)}
				</Card>
			</div>

			<div>
				<SectionTitle
					title="Greedy eval reward"
					subtitle="Headline reward per selected run"
				/>
				<Card>
					<MetricBarChart data={bars} valueLabel="Eval reward" />
				</Card>
			</div>
		</div>
	);
}
