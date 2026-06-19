"""Collect reproducibility metadata for benchmark runs.

Embedding the git commit, library versions and device in every report makes a
result auditable and rerunnable instead of an opaque number.
"""

import platform
import subprocess

import numpy as np


def git_commit() -> str:
    """Return the short git commit hash, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def collect_metadata(extra: dict | None = None) -> dict:
    """Collect environment metadata for a benchmark run.

    Args:
        extra: Extra run-specific fields (e.g. seeds) to merge in.

    Returns:
        Dict with git commit, python/numpy/torch versions, platform and device.
    """
    metadata = {
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }

    try:
        import torch
    except ImportError:
        metadata["torch"] = None
        metadata["device"] = "cpu"
    else:
        metadata["torch"] = torch.__version__
        metadata["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    if extra:
        metadata.update(extra)
    return metadata
