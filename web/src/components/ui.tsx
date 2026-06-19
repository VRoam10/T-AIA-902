import type { ReactNode } from "react";

/**
 * Bordered surface used to group related content.
 */
export function Card({
	children,
	className = "",
}: {
	children: ReactNode;
	className?: string;
}) {
	return (
		<div
			className={`rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-card)] p-5 ${className}`}
		>
			{children}
		</div>
	);
}

/**
 * Compact metric tile showing a label and a primary value with optional hint.
 */
export function MetricCard({
	label,
	value,
	hint,
}: {
	label: string;
	value: string;
	hint?: string;
}) {
	return (
		<Card>
			<p className="text-xs uppercase tracking-wider text-[var(--text-muted)]">
				{label}
			</p>
			<p className="mt-1 text-xl font-semibold tabular-nums">{value}</p>
			{hint ? (
				<p className="mt-0.5 text-xs text-[var(--text-muted)]">{hint}</p>
			) : null}
		</Card>
	);
}

/**
 * Small rounded label used for tags such as the benchmark kind or env.
 */
export function Badge({ children }: { children: ReactNode }) {
	return (
		<span className="rounded-full border border-[var(--border)] bg-[var(--bg-muted)] px-2 py-0.5 text-xs text-[var(--text-secondary)]">
			{children}
		</span>
	);
}

/**
 * Section heading with an optional subtitle.
 */
export function SectionTitle({
	title,
	subtitle,
}: {
	title: string;
	subtitle?: string;
}) {
	return (
		<div className="mb-3">
			<h2 className="text-lg font-semibold">{title}</h2>
			{subtitle ? (
				<p className="text-sm text-[var(--text-muted)]">{subtitle}</p>
			) : null}
		</div>
	);
}
