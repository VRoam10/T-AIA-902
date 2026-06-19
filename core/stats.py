"""Statistical aggregation helpers for multi-seed benchmark runs.

Turning a set of single-seed results into mean / std / confidence intervals is
what separates a trustworthy benchmark from a single lucky (or unlucky) run.
"""

import math
from collections.abc import Iterable

import numpy as np

_BOOL = bool


def is_scalar(value) -> bool:
    """Return True for plain numeric scalars (ints/floats but not booleans)."""
    return isinstance(value, int | float) and not isinstance(value, _BOOL)


def numeric_keys(run: dict) -> list[str]:
    """Return the keys of a result dict whose values are numeric scalars."""
    return [key for key, value in run.items() if is_scalar(value)]


def aggregate(runs: Iterable[dict], scalar_keys: list[str] | None = None) -> dict:
    """Aggregate scalar metrics across multiple runs.

    Args:
        runs: An iterable of result dicts (one per seed).
        scalar_keys: Keys to aggregate. When None, the numeric keys of the
            first run are used.

    Returns:
        Dict mapping each key to a stats dict with mean, std, ci95, min, max
        and n (the number of non-null samples). Keys absent or null in every
        run are skipped. ci95 is the 95% confidence interval half-width.
    """
    runs = list(runs)
    if not runs:
        return {}

    if scalar_keys is None:
        scalar_keys = numeric_keys(runs[0])

    out: dict = {}
    for key in scalar_keys:
        values = [run[key] for run in runs if is_scalar(run.get(key))]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        n = int(arr.size)
        std = float(np.std(arr))
        ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        out[key] = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(std, 4),
            "ci95": round(ci95, 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "n": n,
        }
    return out


def summary_line(stat: dict) -> str:
    """Format an aggregated stat dict as a ``mean ± std`` string."""
    return f"{stat['mean']} ± {stat['std']}"
