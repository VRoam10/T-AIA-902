import Link from "next/link";
import { Badge, Card } from "@/components/ui";
import { getIndex } from "@/lib/data";
import { formatNumber } from "@/lib/format";

export default async function HomePage() {
	const { runs } = await getIndex();

	return (
		<div>
			<div className="mb-6">
				<h1 className="text-2xl font-semibold">Benchmark runs</h1>
				<p className="mt-1 text-[var(--text-secondary)]">
					{runs.length} run{runs.length === 1 ? "" : "s"} indexed. Click a run
					for details.
				</p>
			</div>

			{runs.length === 0 ? (
				<Card>
					<p className="text-[var(--text-secondary)]">
						No data yet. Run a benchmark, then sync results with{" "}
						<code className="rounded bg-[var(--bg-muted)] px-1.5 py-0.5 text-sm">
							npm run sync
						</code>
						.
					</p>
				</Card>
			) : (
				<div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
					{runs.map((run) => (
						<Link key={run.id} href={`/run/${run.id}`}>
							<Card className="h-full transition-colors hover:border-[var(--border)]">
								<div className="flex items-center justify-between gap-2">
									<span className="font-medium">
										{run.benchmark ?? "benchmark"}
									</span>
									<Badge>{run.multiseed ? "multi-seed" : "single"}</Badge>
								</div>
								<p className="mt-1 text-sm text-[var(--text-secondary)]">
									{run.algo ?? "?"} · {run.env ?? "?"}
								</p>
								<div className="mt-4 flex items-end justify-between">
									<div>
										<p className="text-xs uppercase tracking-wider text-[var(--text-muted)]">
											Eval reward
										</p>
										<p className="text-lg font-semibold tabular-nums">
											{formatNumber(run.headline_eval_reward)}
										</p>
									</div>
									<p className="text-xs text-[var(--text-muted)]">
										{run.n_seeds ?? run.seeds?.length ?? 1} seed
										{(run.n_seeds ?? run.seeds?.length ?? 1) === 1 ? "" : "s"}
									</p>
								</div>
							</Card>
						</Link>
					))}
				</div>
			)}
		</div>
	);
}
