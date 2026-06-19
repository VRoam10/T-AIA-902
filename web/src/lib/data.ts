import { promises as fs } from "node:fs";
import path from "node:path";
import type {
	BenchmarkIndex,
	IndexEntry,
	RunDetail,
	RunMetadata,
	RunResult,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

async function readJson<T>(relativePath: string): Promise<T | null> {
	try {
		const raw = await fs.readFile(path.join(DATA_DIR, relativePath), "utf-8");
		return JSON.parse(raw) as T;
	} catch {
		return null;
	}
}

/**
 * Read the benchmark manifest. Returns an empty index when no data is synced.
 */
export async function getIndex(): Promise<BenchmarkIndex> {
	const index = await readJson<BenchmarkIndex>("index.json");
	return index ?? { runs: [] };
}

/**
 * Return the index entry for a given run id, or null when unknown.
 */
export async function getEntry(id: string): Promise<IndexEntry | null> {
	const { runs } = await getIndex();
	return runs.find((run) => run.id === id) ?? null;
}

/**
 * Load the full detail for a run: its index entry, metadata and the richest
 * available result JSON (results_full → summary → results).
 *
 * @param id - The run id (its directory name under public/data).
 */
export async function getRunDetail(id: string): Promise<RunDetail | null> {
	const entry = await getEntry(id);
	if (!entry) return null;

	const metadata = await readJson<RunMetadata>(`${id}/metadata.json`);
	const result =
		(await readJson<RunResult>(`${id}/results_full.json`)) ??
		(await readJson<RunResult>(`${id}/summary.json`)) ??
		(await readJson<RunResult>(`${id}/results.json`)) ??
		{};

	return { entry, metadata, result };
}
