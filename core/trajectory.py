"""Automatic per-map trajectory generation for BeamNG environments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


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
    """Resample a polyline at uniform arc-length intervals per segment.

    Original vertices are always preserved (so corners survive resampling).
    Within each straight segment between two adjacent vertices, samples are
    placed every `spacing` metres measured in the XY plane (Z is linearly
    interpolated).  If a segment is shorter than `spacing`, no interior
    samples are added for it.  For a closed polyline (path[0] == path[-1])
    the duplicate closing vertex is dropped from the output so the loop
    starts and ends at distinct positions in the sample sequence.
    """
    if len(path) < 2:
        raise ValueError("resample requires at least 2 points")
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0")

    closed = path[0] == path[-1]
    last_idx = len(path) - 1

    out: list[Vec3] = [path[0]]
    for i in range(1, len(path)):
        a, b = path[i - 1], path[i]
        seg_len = _segment_length(a, b)
        if seg_len > 0.0:
            d = spacing
            while d < seg_len:
                t = d / seg_len
                out.append((
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                ))
                d += spacing
        # Append the segment endpoint (next vertex), unless it's the closing
        # duplicate of a closed loop.
        if closed and i == last_idx:
            continue
        if b != out[-1]:
            out.append(b)
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


SPARSE_SPACING_M = 25.0
DENSE_SPACING_M = 8.0
SPAWN_Z_OFFSET_M = 1.0
FALLBACK_SIDE_M = 80.0
FALLBACK_GROUND_Z = 1.0


def _square_loop_fallback(map_name: str) -> TrajectoryData:
    """Generate an 80 m square loop centered on the world origin.

    Used as a last-resort trajectory for maps where get_road_network() returns
    nothing usable (typically `smallgrid`).
    """
    half = FALLBACK_SIDE_M / 2.0
    z = FALLBACK_GROUND_Z
    corners: list[Vec3] = [
        (half, -half, z),
        (half, half, z),
        (-half, half, z),
        (-half, -half, z),
        (half, -half, z),  # close the loop
    ]
    sparse = resample(corners, SPARSE_SPACING_M)
    dense = resample(corners, DENSE_SPACING_M)
    spawn_pos = (sparse[0][0], sparse[0][1], sparse[0][2] + SPAWN_Z_OFFSET_M)
    spawn_rot = heading_to_quat(sparse[0], sparse[1])
    return TrajectoryData(
        spawn_pos=spawn_pos,
        spawn_rot=spawn_rot,
        sparse_waypoints=sparse,
        dense_waypoints=dense,
        map_name=map_name,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source="fallback:square_loop",
    )
