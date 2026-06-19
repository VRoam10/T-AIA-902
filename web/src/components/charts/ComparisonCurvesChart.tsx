"use client";

import {
	Area,
	CartesianGrid,
	ComposedChart,
	Legend,
	Line,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import { downsample } from "@/lib/format";
import type { VariantResult } from "@/lib/types";

const PALETTE = [
	"#3b82f6",
	"#f97316",
	"#10b981",
	"#a855f7",
	"#ef4444",
	"#14b8a6",
];

type Row = Record<string, number | [number, number]>;

/**
 * Build chart rows merging every variant's mean rolling curve and ±std band,
 * aligned on the shortest curve and downsampled for rendering.
 */
function buildRows(
	variants: Record<string, VariantResult>,
	window: number,
): Row[] {
	const names = Object.keys(variants);
	const means = names.map((name) =>
		downsample(variants[name]?.mean_rolling ?? []),
	);
	const stds = names.map((name) =>
		downsample(variants[name]?.std_rolling ?? []),
	);
	const length = Math.min(...means.map((m) => m.length));
	if (!Number.isFinite(length) || length === 0) return [];

	const rows: Row[] = [];
	for (let i = 0; i < length; i++) {
		const row: Row = { episode: window + i };
		names.forEach((name, idx) => {
			const mean = means[idx]?.[i] ?? 0;
			const std = stds[idx]?.[i] ?? 0;
			row[name] = mean;
			row[`${name}__band`] = [mean - std, mean + std];
		});
		rows.push(row);
	}
	return rows;
}

/**
 * Overlay of each variant's mean rolling-average reward with a ±std band.
 *
 * @param variants - Per-variant aggregated curves.
 * @param window - Rolling-average window, used to offset the x-axis.
 */
export function ComparisonCurvesChart({
	variants,
	window,
}: {
	variants: Record<string, VariantResult>;
	window: number;
}) {
	const names = Object.keys(variants);
	const rows = buildRows(variants, window);

	if (rows.length === 0) {
		return (
			<p className="text-sm text-[var(--text-muted)]">No curves available.</p>
		);
	}

	return (
		<ResponsiveContainer width="100%" height={360}>
			<ComposedChart
				data={rows}
				margin={{ top: 10, right: 12, left: 0, bottom: 4 }}
			>
				<CartesianGrid
					strokeDasharray="3 3"
					stroke="var(--grid-line)"
					vertical={false}
				/>
				<XAxis
					dataKey="episode"
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
					labelFormatter={(label) => `Episode ${label}`}
				/>
				<Legend wrapperStyle={{ fontSize: 12 }} />
				{names.map((name, idx) => {
					const color = PALETTE[idx % PALETTE.length];
					return (
						<Area
							key={`${name}__band`}
							dataKey={`${name}__band`}
							stroke="none"
							fill={color}
							fillOpacity={0.12}
							legendType="none"
							isAnimationActive={false}
						/>
					);
				})}
				{names.map((name, idx) => (
					<Line
						key={name}
						dataKey={name}
						stroke={PALETTE[idx % PALETTE.length]}
						strokeWidth={2}
						dot={false}
						isAnimationActive={false}
					/>
				))}
			</ComposedChart>
		</ResponsiveContainer>
	);
}
