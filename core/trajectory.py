"""Automatic per-map trajectory generation for BeamNG environments."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    def from_json(cls, payload: str) -> TrajectoryData:
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
    plane (Z is linearly interpolated).  Original interior vertices are NOT
    preserved — use this when you want a uniform subsampling regardless of
    the input polyline's vertex density.  Functions that need vertex
    preservation (e.g. closed-loop corners) should sample per-segment
    themselves; see _square_loop_fallback for an example.
    """
    if len(path) < 2:
        raise ValueError("resample requires at least 2 points")
    if spacing <= 0.0:
        raise ValueError("spacing must be > 0")

    cum = [0.0]
    for i in range(1, len(path)):
        cum.append(cum[-1] + _segment_length(path[i - 1], path[i]))
    total = cum[-1]

    out: list[Vec3] = [path[0]]
    target = spacing
    seg = 1  # index of the original vertex at the END of the current segment

    while target < total:
        while seg < len(path) and cum[seg] < target:
            seg += 1
        if seg >= len(path):
            break
        seg_start_d = cum[seg - 1]
        seg_len = cum[seg] - seg_start_d
        t = (target - seg_start_d) / seg_len if seg_len > 0 else 0.0
        a, b = path[seg - 1], path[seg]
        out.append(
            (
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            )
        )
        target += spacing

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


SPARSE_SPACING_M = 25.0
DENSE_SPACING_M = 8.0
SPAWN_Z_OFFSET_M = 1.0
FALLBACK_SIDE_M = 80.0
FALLBACK_GROUND_Z = 1.0

CACHE_DIR = Path("outputs/trajectories")


def _square_loop_fallback(map_name: str) -> TrajectoryData:
    """Generate an 80 m square loop centered on the world origin.

    Used as a last-resort trajectory for maps where get_road_network() returns
    nothing usable (typically `smallgrid`).  Each side is uniformly subdivided
    so that every corner appears exactly in the output.
    """
    half = FALLBACK_SIDE_M / 2.0
    z = FALLBACK_GROUND_Z
    corners: list[Vec3] = [
        (half, -half, z),
        (half, half, z),
        (-half, half, z),
        (-half, -half, z),
    ]

    def loop_samples(spacing: float) -> list[Vec3]:
        n_per_side = max(1, round(FALLBACK_SIDE_M / spacing))
        out: list[Vec3] = []
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            for k in range(n_per_side):
                t = k / n_per_side
                out.append(
                    (
                        a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        z,
                    )
                )
        return out

    sparse = loop_samples(SPARSE_SPACING_M)
    dense = loop_samples(DENSE_SPACING_M)
    spawn_pos = (sparse[0][0], sparse[0][1], sparse[0][2] + SPAWN_Z_OFFSET_M)
    spawn_rot = heading_to_quat(sparse[0], sparse[1])
    return TrajectoryData(
        spawn_pos=spawn_pos,
        spawn_rot=spawn_rot,
        sparse_waypoints=sparse,
        dense_waypoints=dense,
        map_name=map_name,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source="fallback:square_loop",
    )


def _edge_center(edge: dict) -> Vec3:
    """Pull the centerline point out of a single road-edge dict.

    BeamNGpy versions sometimes expose this under `"middle"` and sometimes only
    as `"left"` + `"right"` — we accept either shape.  Raises ValueError if
    neither is present.
    """
    if "middle" in edge:
        return tuple(edge["middle"])  # type: ignore[return-value]
    if "left" in edge and "right" in edge:
        left, right = edge["left"], edge["right"]
        return (
            (left[0] + right[0]) / 2.0,
            (left[1] + right[1]) / 2.0,
            (left[2] + right[2]) / 2.0,
        )
    raise ValueError(f"edge dict missing centerline keys: {sorted(edge.keys())}")


def _extract_longest_road(network: dict) -> tuple[str | None, list[Vec3] | None]:
    """Return (road_id, centerline) of the longest drivable road in `network`.

    `network` is the dict returned by `bng.scenario.get_road_network(...)`.
    Returns (None, None) if no road has at least two edges with non-zero
    cumulative length.
    """
    best_id: str | None = None
    best_centerline: list[Vec3] | None = None
    best_length = 0.0

    for road_id, road in network.items():
        edges = road.get("edges", []) if isinstance(road, dict) else []
        if len(edges) < 2:
            continue
        try:
            centerline = [_edge_center(e) for e in edges]
        except ValueError:
            continue
        length = sum(
            _segment_length(centerline[i], centerline[i + 1]) for i in range(len(centerline) - 1)
        )
        if length > best_length:
            best_length = length
            best_id = road_id
            best_centerline = centerline

    if best_centerline is None or best_length == 0.0:
        return (None, None)
    return (best_id, best_centerline)


def generate(bng, map_name: str) -> TrajectoryData:
    """Probe BeamNG for the map's road network and build a TrajectoryData.

    Requires `bng` to be already connected with the target map's scenario loaded
    (any scenario on the right map is fine — only get_road_network is called).
    """
    network = bng.scenario.get_road_network(include_edges=True, drivable_only=True)
    road_id, centerline = _extract_longest_road(network)

    if centerline is None:
        return _square_loop_fallback(map_name=map_name)

    sparse = resample(centerline, SPARSE_SPACING_M)
    dense = resample(centerline, DENSE_SPACING_M)
    spawn_pos = (sparse[0][0], sparse[0][1], sparse[0][2] + SPAWN_Z_OFFSET_M)
    spawn_rot = heading_to_quat(sparse[0], sparse[1])
    return TrajectoryData(
        spawn_pos=spawn_pos,
        spawn_rot=spawn_rot,
        sparse_waypoints=sparse,
        dense_waypoints=dense,
        map_name=map_name,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=f"road_network:{road_id}",
    )


def load_or_generate(map_name: str, bng) -> TrajectoryData:
    """Return the cached trajectory for `map_name` or generate one via BeamNG.

    Raises RuntimeError if no cache exists and `bng` is None.
    A corrupt cache file is logged, deleted, and regenerated (if `bng` is given).
    """
    cache_path = CACHE_DIR / f"{map_name}.json"
    if cache_path.exists():
        try:
            return TrajectoryData.from_json(cache_path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[trajectory] cache for '{map_name}' is corrupt ({exc}); regenerating")
            cache_path.unlink(missing_ok=True)

    if bng is None:
        raise RuntimeError(
            f"No cached trajectory for '{map_name}'. Launch BeamNG and run "
            "'Generate trajectories' from the main menu first."
        )

    data = generate(bng, map_name)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(data.to_json())
    return data
