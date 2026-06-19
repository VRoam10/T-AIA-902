"use client";

import {
	Bar,
	BarChart,
	CartesianGrid,
	Cell,
	ErrorBar,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";

const PALETTE = [
	"#3b82f6",
	"#f97316",
	"#10b981",
	"#a855f7",
	"#ef4444",
	"#14b8a6",
];

export type BarDatum = {
	label: string;
	value: number;
	error?: number;
};

/**
 * Vertical bar chart with optional ±error bars, one colour per bar.
 *
 * @param data - Bars to render (label, value, optional error).
 * @param valueLabel - Axis/tooltip label for the value dimension.
 */
export function MetricBarChart({
	data,
	valueLabel,
}: {
	data: BarDatum[];
	valueLabel: string;
}) {
	if (data.length === 0) {
		return <p className="text-sm text-[var(--text-muted)]">No data.</p>;
	}

	return (
		<ResponsiveContainer width="100%" height={300}>
			<BarChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 4 }}>
				<CartesianGrid
					strokeDasharray="3 3"
					stroke="var(--grid-line)"
					vertical={false}
				/>
				<XAxis
					dataKey="label"
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
					cursor={{ fill: "var(--bg-muted)" }}
					contentStyle={{
						background: "var(--bg-base)",
						border: "1px solid var(--border)",
						borderRadius: 8,
						fontSize: 12,
					}}
					formatter={(value) => [
						typeof value === "number" ? value.toFixed(2) : String(value),
						valueLabel,
					]}
				/>
				<Bar dataKey="value" radius={[4, 4, 0, 0]}>
					{data.map((entry, idx) => (
						<Cell key={entry.label} fill={PALETTE[idx % PALETTE.length]} />
					))}
					<ErrorBar
						dataKey="error"
						width={4}
						strokeWidth={1.5}
						stroke="var(--text-muted)"
					/>
				</Bar>
			</BarChart>
		</ResponsiveContainer>
	);
}
