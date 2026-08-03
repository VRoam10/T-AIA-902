"""Where a car sits along its path: arc length, cross-track error and tangent.

Checkpoints answer "which target is next". This module answers "where am I on the
line between them", which is the question a reward can differentiate over a
kilometre-long segment. Everything here is a function of position alone, so the
numbers never jump when the target checkpoint advances, and following the road
round a bend can never read as backward progress — which is exactly what
"straight-line distance to the next checkpoint" got wrong.

Pure math: no beamngpy, no ``self``, no logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathPosition:
    """A projection result, in the terms the envs and the reward consume."""

    progress_m: float  # arc length from the polyline start to the projection point
    cross_track_m: float  # signed perpendicular offset; + = left of travel
    tangent_rad: float  # heading of the segment projected onto
    segment_index: int  # which segment that was (0 = first)
    segment_len_m: float  # its length


# What "no usable path" reads as. Zeros are already how the observation and the
# reward spell "no data", so a degenerate polyline needs no special casing upstream.
NEUTRAL = PathPosition(0.0, 0.0, 0.0, 0, 0.0)


def path_length(polyline) -> float:
    """Total XY length of a polyline. 0.0 for fewer than two points."""
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return 0.0
    return float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())


def project_onto_path(polyline, pos) -> PathPosition:
    """Project ``pos`` onto ``polyline`` in the XY plane and describe where it landed.

    The segment chosen is the one whose clamped perpendicular distance is smallest,
    so a car cutting a corner still projects onto that corner; ties go to the
    earlier segment. Positions before the start or past the end clamp to the ends,
    which is what keeps progress bounded by the path length.

    Returns :data:`NEUTRAL` for an empty or single-point polyline.

    Caveat for closed circuits: a track that passes close to itself can project
    onto the wrong lap of the same geometry. Laps are 1 today; lap counting belongs
    to the caller, which adds ``laps_done * path_length(polyline)``.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return NEUTRAL

    a = pts[:-1, :2]
    seg = pts[1:, :2] - a
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    # A repeated point would divide by zero; its t collapses to 0 and its cross
    # product is 0, so substituting 1.0 keeps the arithmetic finite and harmless.
    safe_len = np.where(seg_len > 0.0, seg_len, 1.0)

    p = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)
    rel = p - a
    t = np.clip(
        (rel[:, 0] * seg[:, 0] + rel[:, 1] * seg[:, 1]) / (safe_len * safe_len), 0.0, 1.0
    )
    foot = a + seg * t[:, None]
    i = int(np.argmin(np.hypot(p[0] - foot[:, 0], p[1] - foot[:, 1])))

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    cross = seg[i, 0] * rel[i, 1] - seg[i, 1] * rel[i, 0]
    return PathPosition(
        progress_m=float(cum[i] + t[i] * seg_len[i]),
        cross_track_m=float(cross / safe_len[i]),
        tangent_rad=float(np.arctan2(seg[i, 1], seg[i, 0])),
        segment_index=i,
        segment_len_m=float(seg_len[i]),
    )
