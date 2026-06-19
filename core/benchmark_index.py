"""Build an index.json manifest aggregating every benchmark run on disk.

The manifest is the bridge to the web dashboard: a single, light JSON listing
all runs with their headline metrics and the relative path to their artifacts.
"""

import json
import os

_RUN_FILES = ("summary.json", "results.json", "results_full.json")


def _read_json(path: str) -> dict | None:
    """Read a JSON file, returning None when it is missing or invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _headline_reward(run_dir: str) -> float | None:
    """Extract a comparable headline eval reward from a run's JSON artifacts.

    Handles the aggregate shape (multi-seed), the flat shape (single run),
    the variants shape (comparison) and the leaderboard shape (gridsearch).
    """
    summary = _read_json(os.path.join(run_dir, "summary.json"))
    if summary and "aggregate" in summary:
        stat = summary["aggregate"].get("eval_mean_reward")
        if stat:
            return stat.get("mean")

    results = _read_json(os.path.join(run_dir, "results.json"))
    if results:
        if "eval_mean_reward" in results:
            return results["eval_mean_reward"]
        if "variants" in results:
            rewards = [
                v["aggregate"].get("eval_mean_reward", {}).get("mean")
                for v in results["variants"].values()
            ]
            rewards = [r for r in rewards if r is not None]
            return max(rewards) if rewards else None
        if results.get("best"):
            return results["best"].get("eval_mean_reward")
    return None


def _build_entry(benchmarks_dir: str, run_id: str) -> dict | None:
    """Build a single index entry from a run directory, or None if invalid."""
    run_dir = os.path.join(benchmarks_dir, run_id)
    if not os.path.isdir(run_dir):
        return None
    metadata = _read_json(os.path.join(run_dir, "metadata.json"))
    if metadata is None:
        return None

    has_artifacts = any(os.path.exists(os.path.join(run_dir, name)) for name in _RUN_FILES)
    if not has_artifacts:
        return None

    return {
        "id": run_id,
        "benchmark": metadata.get("benchmark"),
        "algo": metadata.get("algo"),
        "env": metadata.get("env"),
        "seeds": metadata.get("seeds"),
        "n_seeds": metadata.get("n_seeds"),
        "git_commit": metadata.get("git_commit"),
        "device": metadata.get("device"),
        "multiseed": "multiseed" in run_id,
        "headline_eval_reward": _headline_reward(run_dir),
        "path": run_id,
    }


def build_index(benchmarks_dir: str = "outputs/benchmarks") -> dict:
    """Scan the benchmarks directory and write an index.json manifest.

    Args:
        benchmarks_dir: Directory containing one sub-folder per benchmark run.

    Returns:
        The manifest dict ({"runs": [...]}); also written to
        ``<benchmarks_dir>/index.json``. Runs are sorted newest-first by id.
    """
    if not os.path.isdir(benchmarks_dir):
        return {"runs": []}

    entries = []
    for run_id in os.listdir(benchmarks_dir):
        entry = _build_entry(benchmarks_dir, run_id)
        if entry is not None:
            entries.append(entry)

    entries.sort(key=lambda e: e["id"], reverse=True)
    manifest = {"runs": entries}

    with open(os.path.join(benchmarks_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


if __name__ == "__main__":
    manifest = build_index()
    print(f"Indexed {len(manifest['runs'])} run(s) into outputs/benchmarks/index.json")
