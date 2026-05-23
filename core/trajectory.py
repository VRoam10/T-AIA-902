"""Automatic per-map trajectory generation for BeamNG environments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrajectoryData:
    """A spawn pose plus two pre-sampled waypoint sequences for one map."""

    spawn_pos: Vec3
    spawn_rot: Quat
    sparse_waypoints: list[Vec3]
    dense_waypoints: list[Vec3]
    map_name: str
    generated_at: str
    source: str

    def to_json(self) -> str:
        d = asdict(self)
        d["spawn_pos"] = list(self.spawn_pos)
        d["spawn_rot"] = list(self.spawn_rot)
        d["sparse_waypoints"] = [list(p) for p in self.sparse_waypoints]
        d["dense_waypoints"] = [list(p) for p in self.dense_waypoints]
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "TrajectoryData":
        d = json.loads(payload)
        return cls(
            spawn_pos=tuple(d["spawn_pos"]),
            spawn_rot=tuple(d["spawn_rot"]),
            sparse_waypoints=[tuple(p) for p in d["sparse_waypoints"]],
            dense_waypoints=[tuple(p) for p in d["dense_waypoints"]],
            map_name=d["map_name"],
            generated_at=d["generated_at"],
            source=d["source"],
        )


def _segment_length(a: Vec3, b: Vec3) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def resample(path: list[Vec3], spacing: float) -> list[Vec3]:
    """Resample a polyline at uniform arc-length intervals.

    The first and last original points are always included.  Internal samples
    are placed every `spacing` metres along the polyline measured in the XY
    plane (Z is linearly interpolated).
    """
    if len(path) < 2:
        raise ValueError("resample requires at least 2 points")
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0")

    # Cumulative arc length per original vertex
    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + _segment_length(path[i - 1], path[i]))
    total = cum[-1]

    out: list[Vec3] = [path[0]]
    target = spacing
    seg = 1  # index of the original vertex at the END of the current segment

    while target < total:
        # Advance until target falls inside [cum[seg-1], cum[seg]]
        while seg < len(path) and cum[seg] < target:
            seg += 1
        if seg >= len(path):
            break
        seg_start_d = cum[seg - 1]
        seg_len = cum[seg] - seg_start_d
        t = (target - seg_start_d) / seg_len if seg_len > 0 else 0.0
        a, b = path[seg - 1], path[seg]
        out.append((
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ))
        target += spacing

    # Always include the last original point
    if out[-1] != path[-1]:
        out.append(path[-1])
    return out


def heading_to_quat(p0: Vec3, p1: Vec3) -> Quat:
    """Quaternion (x, y, z, w) matching BeamNG's vehicle-spawn convention.

    Returns the rotation that orients a vehicle's forward axis (+Y in BeamNG)
    to point from `p0` toward `p1` in the XY plane.  The identity quaternion
    (0, 0, 0, 1) corresponds to facing North (+Y).  See `scenario_creator.md`
    for the cardinal-direction reference table.  Vertical delta is ignored.
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if dx == 0.0 and dy == 0.0:
        raise ValueError("p0 and p1 must differ in the XY plane")
    heading = math.atan2(dx, dy)
    return (0.0, 0.0, math.sin(heading / 2.0), math.cos(heading / 2.0))
