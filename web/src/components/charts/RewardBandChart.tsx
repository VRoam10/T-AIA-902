"use client";

import {
	Area,
	CartesianGrid,
	ComposedChart,
	Line,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import { downsample } from "@/lib/format";

type BandPoint = {
	episode: number;
	mean: number;
	band: [number, number];
};

/**
 * Compute the per-episode mean and ±std band across a set of reward curves.
 */
function buildBand(curves: number[][]): BandPoint[] {
	const usable = curves.filter((c) => c.length > 0);
	if (usable.length === 0) return [];
	const length = Math.min(...usable.map((c) => c.length));
	const points: BandPoint[] = [];
	for (let i = 0; i < length; i++) {
		const column = usable.map((c) => c[i] ?? 0);
		const mean = column.reduce((a, b) => a + b, 0) / column.length;
		const variance =
			column.reduce((a, b) => a + (b - mean) ** 2, 0) / column.length;
		const std = Math.sqrt(variance);
		points.push({ episode: i + 1, mean, band: [mean - std, mean + std] });
	}
	return points;
}

/**
 * Line chart of the mean reward per episode with a shaded ±std band across seeds.
 *
 * @param curves - One reward curve (per episode) per seed.
 */
export function RewardBandChart({ curves }: { curves: number[][] }) {
	const reduced = curves.map((c) => downsample(c));
	const data = buildBand(reduced);

	if (data.length === 0) {
		return (
			<p className="text-sm text-[var(--text-muted)]">
				No reward curve available.
			</p>
		);
	}

	return (
		<ResponsiveContainer width="100%" height={320}>
			<ComposedChart
				data={data}
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
					formatter={(value, name) => [
						typeof value === "number" ? value.toFixed(2) : String(value),
						name === "mean" ? "Mean" : name,
					]}
				/>
				<Area
					dataKey="band"
					stroke="none"
					fill="var(--accent)"
					fillOpacity={0.15}
					isAnimationActive={false}
				/>
				<Line
					dataKey="mean"
					stroke="var(--accent)"
					strokeWidth={2}
					dot={false}
					isAnimationActive={false}
				/>
			</ComposedChart>
		</ResponsiveContainer>
	);
}
