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

# How far either side of the last known arc length the search looks when it is
# seeded. It must exceed the distance a car can cover in one env step, or a fast
# car outruns its own window and progress freezes: the race car tops out near
# 81 m/s, and one step is beamng_spec.SECONDS_PER_ENV_STEP (0.333 s), so 27 m.
# The rest is margin for a car that is tumbling rather than driving.
SEARCH_WINDOW_M = 60.0


def path_length(polyline) -> float:
    """Total XY length of a polyline. 0.0 for fewer than two points."""
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return 0.0
    return float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())


def waypoint_arcs(polyline) -> list[float]:
    """Arc length of every vertex after the first — one entry per waypoint.

    The guide line is ``[spawn, *waypoints]``, so entry k is how far along the
    path waypoint k sits. This is what "has the car reached checkpoint k" should
    be measured against. Proximity is not the same question: with the dense
    waypoint spacing (8 m) equal to the old arrival radius (8 m) and four times
    the spawn clearance (2 m), a car parked at the spawn was already inside
    checkpoint 0's ring, and every next checkpoint on the dense chain was already
    inside the current one's.

    Empty for a degenerate polyline, so it zips with an empty waypoint list.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return []
    return [float(v) for v in np.cumsum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))]


def project_onto_path(polyline, pos, *, near_m: float | None = None) -> PathPosition:
    """Project ``pos`` onto ``polyline`` in the XY plane and describe where it landed.

    The segment chosen is the one whose clamped perpendicular distance is smallest,
    so a car cutting a corner still projects onto that corner; ties go to the
    earlier segment. Positions before the start or past the end clamp to the ends,
    which is what keeps progress bounded by the path length.

    ``near_m`` seeds the search with the arc length the car was last measured at,
    restricting it to :data:`SEARCH_WINDOW_M` either side. Pass it whenever a
    previous reading exists. Without it the search is global, and a path that
    passes close to itself projects onto whichever part happens to be nearest —
    which on a closed circuit means a car sitting at the start/finish line reads
    either arc 0 or a full lap. Since the reward pays ``PROGRESS_COEF`` times the
    *change* in arc length, that ambiguity is worth 3 x the path length in a
    single step: measured at +5301 on gridmap_v2's 1767 m default path, for a car
    that had not moved. A seed whose window leaves the car further away than the
    window is wide is treated as stale (a teleport, or a car off the map) and the
    search falls back to global rather than reporting a confidently wrong arc.

    Returns :data:`NEUTRAL` for an empty or single-point polyline.

    Still not lap-aware: progress is a function of position alone, so a second lap
    of a circuit reads the same as the first. A real lap counter needs a
    lap-crossing *event*, which no projection can supply.
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
    gap = np.hypot(p[0] - foot[:, 0], p[1] - foot[:, 1])

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    i = int(np.argmin(gap))
    if near_m is not None:
        in_window = (cum[1:] >= near_m - SEARCH_WINDOW_M) & (cum[:-1] <= near_m + SEARCH_WINDOW_M)
        if in_window.any():
            local = int(np.argmin(np.where(in_window, gap, np.inf)))
            if gap[local] <= SEARCH_WINDOW_M:
                i = local

    cross = seg[i, 0] * rel[i, 1] - seg[i, 1] * rel[i, 0]
    return PathPosition(
        progress_m=float(cum[i] + t[i] * seg_len[i]),
        cross_track_m=float(cross / safe_len[i]),
        tangent_rad=float(np.arctan2(seg[i, 1], seg[i, 0])),
        segment_index=i,
        segment_len_m=float(seg_len[i]),
    )
