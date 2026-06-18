"""Sync benchmark JSON outputs into the web dashboard's public/data directory.

Rebuilds the index manifest, then mirrors every run's JSON artifacts into
``web/public/data`` so the Next.js dashboard can statically export them.
Run from anywhere: ``python scripts/sync_web_data.py``.
"""

import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from core.benchmark_index import build_index  # noqa: E402

SRC = os.path.join(REPO_ROOT, "outputs", "benchmarks")
DST = os.path.join(REPO_ROOT, "web", "public", "data")


def sync() -> int:
    """Copy benchmark JSON artifacts into the dashboard data directory.

    Returns:
        The number of run directories synced.
    """
    build_index(SRC)

    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST, exist_ok=True)

    index_path = os.path.join(SRC, "index.json")
    if os.path.exists(index_path):
        shutil.copy(index_path, os.path.join(DST, "index.json"))

    count = 0
    for name in os.listdir(SRC):
        run_dir = os.path.join(SRC, name)
        if not os.path.isdir(run_dir):
            continue
        target_dir = os.path.join(DST, name)
        os.makedirs(target_dir, exist_ok=True)
        for filename in os.listdir(run_dir):
            if filename.endswith(".json"):
                shutil.copy(os.path.join(run_dir, filename), os.path.join(target_dir, filename))
        count += 1

    print(f"Synced {count} run(s) into web/public/data")
    return count


if __name__ == "__main__":
    sync()
