import Link from "next/link";
import { notFound } from "next/navigation";
import { RunBody } from "@/components/RunBody";
import { Badge } from "@/components/ui";
import { getIndex, getRunDetail } from "@/lib/data";

/**
 * Pre-render one page per indexed run for static export.
 */
export async function generateStaticParams() {
	const { runs } = await getIndex();
	return runs.map((run) => ({ id: run.id }));
}

export default async function RunPage({
	params,
}: {
	params: Promise<{ id: string }>;
}) {
	const { id } = await params;
	const detail = await getRunDetail(id);
	if (!detail) notFound();

	const { entry, metadata } = detail;
	const seeds = metadata?.seeds ?? entry.seeds ?? [];

	return (
		<div>
			<Link
				href="/"
				className="text-sm text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
			>
				← All runs
			</Link>

			<div className="mt-3 mb-8">
				<div className="flex flex-wrap items-center gap-2">
					<h1 className="text-2xl font-semibold">{entry.benchmark}</h1>
					<Badge>{entry.multiseed ? "multi-seed" : "single"}</Badge>
				</div>
				<p className="mt-1 text-[var(--text-secondary)]">
					{entry.algo} · {entry.env}
				</p>

				<div className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-sm text-[var(--text-muted)]">
					{seeds.length > 0 ? <span>seeds: {seeds.join(", ")}</span> : null}
					{metadata?.git_commit ? (
						<span>commit: {metadata.git_commit}</span>
					) : null}
					{metadata?.device ? <span>device: {metadata.device}</span> : null}
					{metadata?.torch ? <span>torch: {metadata.torch}</span> : null}
				</div>
			</div>

			<RunBody result={detail.result} />
		</div>
	);
}
