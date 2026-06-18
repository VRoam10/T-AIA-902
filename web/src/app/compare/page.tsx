import { CompareClient } from "@/components/CompareClient";
import { getCompareRuns } from "@/lib/compare";

export default async function ComparePage() {
	const runs = await getCompareRuns();

	return (
		<div>
			<div className="mb-6">
				<h1 className="text-2xl font-semibold">Compare runs</h1>
				<p className="mt-1 text-[var(--text-secondary)]">
					Toggle runs to overlay their reward curves and eval rewards.
				</p>
			</div>
			<CompareClient runs={runs} />
		</div>
	);
}
