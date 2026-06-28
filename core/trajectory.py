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


@dataclass(frozen=True)
class MapTrajectories:
    """All generated paths for one map: one TrajectoryData per teleport point."""

    map_name: str
    generated_at: str
    paths: list[TrajectoryData]

    def to_json(self) -> str:
        return json.dumps(
            {
                "map_name": self.map_name,
                "generated_at": self.generated_at,
                "paths": [json.loads(p.to_json()) for p in self.paths],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, payload: str) -> MapTrajectories:
        d = json.loads(payload)
        # Back-compat: an old cache is a single TrajectoryData object.
        if "paths" not in d and "spawn_pos" in d:
            traj = TrajectoryData.from_json(payload)
            return cls(map_name=traj.map_name, generated_at=traj.generated_at, paths=[traj])
        paths = [TrajectoryData.from_json(json.dumps(p)) for p in d["paths"]]
        return cls(map_name=d["map_name"], generated_at=d["generated_at"], paths=paths)


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
    (0, 0, 0, 1) corresponds to facing North (+Y); positive yaw is CCW around
    +Z (standard right-handed math convention, verified against beamngpy's
    angle_to_quat helper).  Vertical delta is ignored.
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if dx == 0.0 and dy == 0.0:
        raise ValueError("p0 and p1 must differ in the XY plane")
    # BeamNG's yaw is CCW around +Z with identity facing +Y, so the standard
    # atan2(y, x) angle from +X needs a -π/2 shift; equivalently atan2(-dx, dy).
    heading = math.atan2(-dx, dy)
    return (0.0, 0.0, math.sin(heading / 2.0), math.cos(heading / 2.0))


SPARSE_SPACING_M = 25.0
DENSE_SPACING_M = 8.0
SPAWN_Z_OFFSET_M = 1.0
FALLBACK_SIDE_M = 80.0
FALLBACK_GROUND_Z = 1.0
MIN_PATH_SEPARATION_M = 30.0
SPAWN_CLEARANCE_M = 2.0  # keep the first checkpoint at least this far from the spawn
# Advisory only: the healthy checkpoint count a connected path is expected to
# reach (asserted in tests). Generation no longer enforces it — paths follow the
# road as far as it goes, so well-connected spawns far exceed it while genuinely
# short isolated roads carry fewer.
MIN_CHECKPOINTS = 10
ROAD_CONNECT_M = 10.0  # road endpoints this close are treated as a junction
# A teleport whose path can't reach this length (no road / dead-end stub) is
# dropped instead of emitting a pile of waypoints crammed at one spot.
MIN_USABLE_PATH_LENGTH_M = 2 * SPARSE_SPACING_M

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
    # Drop the first sample (corner under spawn) so the first checkpoint
    # ring is ahead of the car rather than around it.
    return TrajectoryData(
        spawn_pos=spawn_pos,
        spawn_rot=spawn_rot,
        sparse_waypoints=sparse[1:],
        dense_waypoints=dense[1:],
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


def _road_centerlines(network: dict) -> list[tuple[str, list[Vec3]]]:
    """Every road in `network` with >= 2 valid edges, as (road_id, centerline)."""
    out: list[tuple[str, list[Vec3]]] = []
    for road_id, road in network.items():
        edges = road.get("edges", []) if isinstance(road, dict) else []
        if len(edges) < 2:
            continue
        try:
            centerline = [_edge_center(e) for e in edges]
        except ValueError:
            continue
        out.append((road_id, centerline))
    return out


def _quat_to_forward(rot: Quat) -> tuple[float, float]:
    """XY forward unit vector for a pure-Z-yaw quaternion (identity -> +Y)."""
    yaw = 2.0 * math.atan2(rot[2], rot[3])
    return (-math.sin(yaw), math.cos(yaw))


def _nearest_road(
    point: Vec3, roads: list[tuple[str, list[Vec3]]]
) -> tuple[str, list[Vec3]] | None:
    """Road whose closest centerline vertex is nearest `point` in the XY plane."""
    best: tuple[str, list[Vec3]] | None = None
    best_d = float("inf")
    for road_id, centerline in roads:
        d = min(math.hypot(v[0] - point[0], v[1] - point[1]) for v in centerline)
        if d < best_d:
            best_d = d
            best = (road_id, centerline)
    return best


def _road_path_from_teleport(
    centerline: list[Vec3], tele_pos: Vec3, forward_xy: tuple[float, float]
) -> list[Vec3]:
    """Sub-polyline from the vertex nearest `tele_pos`, walking with `forward_xy`.

    Picks the traversal direction whose road tangent best aligns with the
    teleport heading, so the car always drives forward along the returned path.
    Falls back to the whole oriented centerline if the snap vertex leaves fewer
    than two points ahead.
    """
    k = min(
        range(len(centerline)),
        key=lambda i: math.hypot(centerline[i][0] - tele_pos[0], centerline[i][1] - tele_pos[1]),
    )
    # Tangent at k (forward along increasing index).
    j = k + 1 if k + 1 < len(centerline) else k - 1
    tangent = (centerline[j][0] - centerline[k][0], centerline[j][1] - centerline[k][1])
    if j < k:  # tangent was measured backward; flip it to point forward-in-index
        tangent = (-tangent[0], -tangent[1])
    aligned = tangent[0] * forward_xy[0] + tangent[1] * forward_xy[1] >= 0.0

    forward_path = centerline[k:] if aligned else list(reversed(centerline[: k + 1]))
    if len(forward_path) >= 2:
        return forward_path
    # Snap sat at the far end; use the whole centerline oriented to the heading.
    return centerline if aligned else list(reversed(centerline))


def _teleport_points(bng) -> list[tuple[Vec3, Quat, str]]:
    """Map quick-travel points as (pos, rot_quat, name).

    Prefers named quick-travel waypoints (`find_waypoints`); falls back to
    `SpawnSphere` spawn points. Returns [] if neither is available (older
    beamngpy or a map without them).
    """
    for getter, _label in (
        (lambda: bng.scenario.find_waypoints(), "waypoints"),
        (lambda: bng.scenario.find_objects_class("SpawnSphere"), "spawnspheres"),
    ):
        try:
            objs = getter()
        except Exception:
            continue
        pts: list[tuple[Vec3, Quat, str]] = []
        for i, o in enumerate(objs or []):
            pos = tuple(getattr(o, "pos", None)) if getattr(o, "pos", None) else None
            if pos is None:
                continue
            rot = getattr(o, "rot_quat", None) or (0.0, 0.0, 0.0, 1.0)
            name = getattr(o, "name", None) or getattr(o, "oid", None) or f"point_{i}"
            pts.append((pos, tuple(rot), str(name)))
        if pts:
            return pts
    return []


def _path_length(centerline: list[Vec3]) -> float:
    return sum(
        _segment_length(centerline[i], centerline[i + 1]) for i in range(len(centerline) - 1)
    )


def _drop_waypoints_near_spawn(
    spawn_pos: Vec3, waypoints: list[Vec3], clearance: float
) -> list[Vec3]:
    """Drop leading waypoints within `clearance` metres (XY) of the spawn.

    Keeps the first checkpoint off the spawn so it isn't auto-registered at
    episode start. Always keeps at least the final waypoint.
    """
    for i, wp in enumerate(waypoints):
        if math.hypot(wp[0] - spawn_pos[0], wp[1] - spawn_pos[1]) >= clearance:
            return waypoints[i:]
    return waypoints[-1:]


def _extend_path_along_network(
    path: list[Vec3],
    roads: list[tuple[str, list[Vec3]]],
    used_ids: set[str],
    connect_m: float = ROAD_CONNECT_M,
) -> list[Vec3]:
    """Grow `path` forward through connected roads for as long as the network runs.

    After the snapped starting road runs out, hop to whichever unused road joins
    the current end (within `connect_m`) and best continues the current heading,
    repeating until no forward road connects (a dead end). This makes a teleport
    on a short road yield a long, well-spaced trajectory (checkpoints at the
    default spacing) that follows the road as far as it goes. `used_ids` bounds
    the walk to each road once, so it always terminates.
    """
    path = list(path)
    while True:
        end, prev = path[-1], path[-2]
        dx, dy = end[0] - prev[0], end[1] - prev[1]
        best: list[Vec3] | None = None
        best_id: str | None = None
        best_score = 0.0  # require a forward-ish continuation (cos > 0)
        for road_id, centerline in roads:
            if road_id in used_ids or len(centerline) < 2:
                continue
            d_start = math.hypot(centerline[0][0] - end[0], centerline[0][1] - end[1])
            d_end = math.hypot(centerline[-1][0] - end[0], centerline[-1][1] - end[1])
            if min(d_start, d_end) > connect_m:
                continue
            oriented = centerline if d_start <= d_end else list(reversed(centerline))
            tx, ty = oriented[1][0] - oriented[0][0], oriented[1][1] - oriented[0][1]
            norm = math.hypot(tx, ty)
            if norm == 0.0:
                continue
            score = (dx * tx + dy * ty) / norm  # |dir| * cos(turn angle)
            if score > best_score:
                best_score, best, best_id = score, oriented, road_id
        if best is None:
            break
        used_ids.add(best_id)
        path.extend(best[1:])  # skip the junction vertex (≈ current end)
    return path


def _spawn_rot_towards(spawn_pos: Vec3, waypoints: list[Vec3], fallback: Quat) -> Quat:
    """Quaternion facing the first waypoint that differs from spawn in the XY plane.

    Used so a spawned vehicle looks toward its next checkpoint rather than at a
    teleport marker's own (often identity) heading. Falls back to `fallback`
    when no waypoint is distinct from the spawn position.
    """
    for wp in waypoints:
        if abs(wp[0] - spawn_pos[0]) > 1e-6 or abs(wp[1] - spawn_pos[1]) > 1e-6:
            return heading_to_quat(spawn_pos, wp)
    return fallback


def _path_from_teleport(
    tele_pos: Vec3, tele_rot: Quat, roads: list[tuple[str, list[Vec3]]], map_name: str
) -> tuple[TrajectoryData, float] | None:
    """Build one TrajectoryData by snapping a teleport point to its nearest road.

    The snapped road is extended forward through connected roads for as long as
    the network runs, so a teleport on a short road yields a long path with
    checkpoints at the default spacing. Paths shorter than MIN_USABLE_PATH_LENGTH_M
    (no real road here) are dropped.

    Returns (trajectory, path_length) for length-based sorting, or None when no
    usable road yields a 2+ point path.
    """
    nearest = _nearest_road(tele_pos, roads)
    if nearest is None:
        return None
    road_id, centerline = nearest
    forward = _quat_to_forward(tele_rot)
    path = _road_path_from_teleport(centerline, tele_pos, forward)
    if len(path) < 2:
        return None
    path = _extend_path_along_network(path, roads, {road_id})
    if _path_length(path) < MIN_USABLE_PATH_LENGTH_M:
        # No real road here (dead-end stub / off-road spawn): drop it instead of
        # emitting a path whose few waypoints all sit at the same place.
        return None
    spawn_pos = (tele_pos[0], tele_pos[1], tele_pos[2] + SPAWN_Z_OFFSET_M)
    sparse = _drop_waypoints_near_spawn(
        spawn_pos, resample(path, SPARSE_SPACING_M), SPAWN_CLEARANCE_M
    )
    dense = _drop_waypoints_near_spawn(
        spawn_pos, resample(path, DENSE_SPACING_M), SPAWN_CLEARANCE_M
    )
    traj = TrajectoryData(
        spawn_pos=spawn_pos,
        spawn_rot=_spawn_rot_towards(spawn_pos, sparse, tele_rot),
        sparse_waypoints=sparse,
        dense_waypoints=dense,
        map_name=map_name,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=f"teleport:{road_id}",
    )
    return traj, _path_length(path)


def generate(bng, map_name: str) -> MapTrajectories:
    """Probe BeamNG for the map's roads + teleport points and build all paths.

    One path per teleport point (snapped to its nearest road, oriented to the
    teleport heading), deduped by MIN_PATH_SEPARATION_M and sorted longest-road
    first. Falls back to a single longest-road path, then a square loop.
    """
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    network = bng.scenario.get_road_network(include_edges=True, drivable_only=True)
    roads = _road_centerlines(network)
    teleports = _teleport_points(bng)
    if teleports:
        print(
            f"[trajectory] {map_name}: {len(teleports)} quick-travel points: "
            + ", ".join(t[2] for t in teleports)
        )

    if roads and teleports:
        scored: list[tuple[TrajectoryData, float]] = []
        accepted_spawns: list[Vec3] = []
        dropped = 0
        for pos, rot, _name in teleports:
            built = _path_from_teleport(pos, rot, roads, map_name)
            if built is None:
                dropped += 1  # no usable road at this teleport
                continue
            traj, length = built
            if any(
                _segment_length(traj.spawn_pos, s) < MIN_PATH_SEPARATION_M for s in accepted_spawns
            ):
                continue
            accepted_spawns.append(traj.spawn_pos)
            scored.append((traj, length))
        if dropped:
            print(
                f"[trajectory] {map_name}: dropped {dropped} teleport(s) with no usable road "
                f"(< {MIN_USABLE_PATH_LENGTH_M:.0f} m of connected path)"
            )
        if scored:
            scored.sort(key=lambda t: t[1], reverse=True)
            return MapTrajectories(
                map_name=map_name,
                generated_at=generated_at,
                paths=[t[0] for t in scored],
            )

    # Fallback 1: single longest road.
    road_id, centerline = _extract_longest_road(network)
    if centerline is not None:
        sparse = resample(centerline, SPARSE_SPACING_M)
        dense = resample(centerline, DENSE_SPACING_M)
        spawn_pos = (sparse[0][0], sparse[0][1], sparse[0][2] + SPAWN_Z_OFFSET_M)
        spawn_rot = heading_to_quat(sparse[0], sparse[1])
        traj = TrajectoryData(
            spawn_pos=spawn_pos,
            spawn_rot=spawn_rot,
            sparse_waypoints=sparse[1:],
            dense_waypoints=dense[1:],
            map_name=map_name,
            generated_at=generated_at,
            source=f"road_network:{road_id}",
        )
        return MapTrajectories(map_name=map_name, generated_at=generated_at, paths=[traj])

    # Fallback 2: square loop.
    return MapTrajectories(
        map_name=map_name,
        generated_at=generated_at,
        paths=[_square_loop_fallback(map_name=map_name)],
    )


def load_or_generate(map_name: str, bng) -> MapTrajectories:
    """Return the cached MapTrajectories for `map_name` or generate via BeamNG.

    Raises RuntimeError if no cache exists and `bng` is None.
    A corrupt cache file is logged, deleted, and regenerated (if `bng` is given).
    """
    cache_path = CACHE_DIR / f"{map_name}.json"
    if cache_path.exists():
        try:
            return MapTrajectories.from_json(cache_path.read_text())
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
