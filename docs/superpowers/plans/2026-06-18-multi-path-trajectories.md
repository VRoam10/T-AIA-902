# Multi-Path Trajectories from Teleport Points Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one road-snapped path per map teleport point so multi-agent vehicles each train on their own path in their own location without colliding.

**Architecture:** `core/trajectory.py` enumerates a map's BeamNG `SpawnSphere` teleport points, snaps each to the nearest road, orients that road to the teleport heading, and resamples it into a `TrajectoryData`. All paths for a map are wrapped in a new `MapTrajectories` and cached. `environments/beamng_multi.py` moves from one shared waypoint list to per-slot waypoints and assigns path `i` to vehicle `i`, hard-erroring when vehicles outnumber paths. The single-agent env takes `paths[0]` and is unchanged in behavior.

**Tech Stack:** Python 3, `beamngpy`, `numpy`, `pytest`. Pure-function logic is unit-tested with mocked BeamNG (no simulator needed), matching the existing test style.

## Global Constraints

- Lint with `ruff` (config in `ruff.toml`); no new lint errors.
- Run tests with `python -m pytest` from the repo root (`c:\Epitech\T-AIA-902`).
- BeamNG-free unit tests only: mock `bng` with `unittest.mock.MagicMock`, never launch the simulator.
- `Vec3 = tuple[float, float, float]`, `Quat = tuple[float, float, float, float]` (already defined in `core/trajectory.py`).
- Single-agent training behavior must not change (it consumes `paths[0]`, and paths are sorted longest-road-first).
- The trajectory cache (`outputs/trajectories/*.json`) is regenerable/gitignored; changing its format is allowed, but `from_json` must still read the old single-object format.

---

### Task 1: `MapTrajectories` dataclass with JSON + old-format back-compat

**Files:**
- Modify: `core/trajectory.py`
- Test: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: existing `TrajectoryData` (unchanged).
- Produces:
  - `MapTrajectories(map_name: str, generated_at: str, paths: list[TrajectoryData])` — frozen dataclass.
  - `MapTrajectories.to_json(self) -> str`
  - `MapTrajectories.from_json(cls, payload: str) -> MapTrajectories` — also accepts the old single-`TrajectoryData` object shape (a dict with a top-level `"spawn_pos"` key), wrapping it as a 1-element `paths` list.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trajectory.py` (extend the existing import from `core.trajectory` to include `MapTrajectories`):

```python
def _sample_traj(map_name="italy", source="road_network:r1"):
    return TrajectoryData(
        spawn_pos=(1.0, 2.0, 3.0),
        spawn_rot=(0.0, 0.0, 0.707, 0.707),
        sparse_waypoints=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        dense_waypoints=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        map_name=map_name,
        generated_at="2026-06-18T12:00:00+00:00",
        source=source,
    )


def test_maptrajectories_json_roundtrip():
    mt = MapTrajectories(
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        paths=[_sample_traj(source="teleport:r1"), _sample_traj(source="teleport:r2")],
    )
    restored = MapTrajectories.from_json(mt.to_json())
    assert restored == mt
    assert len(restored.paths) == 2
    assert restored.paths[1].source == "teleport:r2"


def test_maptrajectories_from_json_accepts_old_single_object_format():
    # Old caches stored a single TrajectoryData object at the top level.
    old_payload = _sample_traj(map_name="gridmap_v2").to_json()
    mt = MapTrajectories.from_json(old_payload)
    assert mt.map_name == "gridmap_v2"
    assert len(mt.paths) == 1
    assert mt.paths[0] == _sample_traj(map_name="gridmap_v2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trajectory.py::test_maptrajectories_json_roundtrip tests/test_trajectory.py::test_maptrajectories_from_json_accepts_old_single_object_format -v`
Expected: FAIL with `ImportError: cannot import name 'MapTrajectories'`.

- [ ] **Step 3: Implement `MapTrajectories`**

In `core/trajectory.py`, after the `TrajectoryData` class (before `_segment_length`), add:

```python
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
    def from_json(cls, payload: str) -> "MapTrajectories":
        d = json.loads(payload)
        # Back-compat: an old cache is a single TrajectoryData object.
        if "paths" not in d and "spawn_pos" in d:
            traj = TrajectoryData.from_json(payload)
            return cls(map_name=traj.map_name, generated_at=traj.generated_at, paths=[traj])
        paths = [TrajectoryData.from_json(json.dumps(p)) for p in d["paths"]]
        return cls(map_name=d["map_name"], generated_at=d["generated_at"], paths=paths)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trajectory.py -v`
Expected: PASS (new tests green; all existing trajectory tests still green — `generate`/`load_or_generate` are untouched in this task).

- [ ] **Step 5: Commit**

```bash
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat: add MapTrajectories wrapper with old-format back-compat"
```

---

### Task 2: Geometry helpers — road list, nearest road, heading orientation

**Files:**
- Modify: `core/trajectory.py`
- Test: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: existing `_edge_center`, `_segment_length`, `Vec3`, `Quat`.
- Produces:
  - `_road_centerlines(network: dict) -> list[tuple[str, list[Vec3]]]` — every road with ≥2 valid edges, as `(road_id, centerline)`.
  - `_quat_to_forward(rot: Quat) -> tuple[float, float]` — XY forward unit vector for a pure-Z-yaw quaternion (identity → `(0.0, 1.0)`).
  - `_nearest_road(point: Vec3, roads: list[tuple[str, list[Vec3]]]) -> tuple[str, list[Vec3]] | None` — road whose closest centerline vertex is nearest `point` (XY); `None` if `roads` empty.
  - `_road_path_from_teleport(centerline: list[Vec3], tele_pos: Vec3, forward_xy: tuple[float, float]) -> list[Vec3]` — sub-polyline starting at the centerline vertex nearest `tele_pos`, walking in whichever direction aligns with `forward_xy`; returns ≥2 points (falls back to the whole oriented centerline if the snap sits at the far end).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trajectory.py` (extend the `core.trajectory` import to include `_road_centerlines`, `_quat_to_forward`, `_nearest_road`, `_road_path_from_teleport`):

```python
def test_road_centerlines_lists_all_multi_edge_roads():
    network = {
        "a": {"edges": [{"middle": (0.0, 0.0, 0.0)}, {"middle": (10.0, 0.0, 0.0)}]},
        "b": {"edges": [{"middle": (0.0, 0.0, 0.0)}]},  # single edge -> skipped
        "c": {"edges": [{"middle": (0.0, 5.0, 0.0)}, {"middle": (0.0, 15.0, 0.0)}]},
    }
    roads = _road_centerlines(network)
    ids = {rid for rid, _ in roads}
    assert ids == {"a", "c"}


def test_quat_to_forward_identity_is_north():
    fx, fy = _quat_to_forward((0.0, 0.0, 0.0, 1.0))
    assert fx == pytest.approx(0.0, abs=1e-6)
    assert fy == pytest.approx(1.0, abs=1e-6)


def test_quat_to_forward_east():
    # yaw -pi/2 faces +X (East): forward = (1, 0)
    rot = (0.0, 0.0, math.sin(-math.pi / 4), math.cos(-math.pi / 4))
    fx, fy = _quat_to_forward(rot)
    assert fx == pytest.approx(1.0, abs=1e-6)
    assert fy == pytest.approx(0.0, abs=1e-6)


def test_nearest_road_picks_closest():
    roads = [
        ("far", [(100.0, 100.0, 0.0), (200.0, 100.0, 0.0)]),
        ("near", [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]),
    ]
    rid, centerline = _nearest_road((1.0, 1.0, 0.0), roads)
    assert rid == "near"
    assert centerline[0] == (0.0, 0.0, 0.0)


def test_nearest_road_none_when_empty():
    assert _nearest_road((0.0, 0.0, 0.0), []) is None


def test_road_path_from_teleport_walks_in_heading_direction():
    # Straight east-west road; teleport mid-road facing East -> path heads +X.
    centerline = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0)]
    path = _road_path_from_teleport(centerline, (10.0, 0.0, 0.0), (1.0, 0.0))
    assert path[0] == (10.0, 0.0, 0.0)
    assert path[-1] == (30.0, 0.0, 0.0)


def test_road_path_from_teleport_reverses_when_facing_back():
    centerline = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0)]
    # Same snap vertex but facing West -> path heads -X.
    path = _road_path_from_teleport(centerline, (20.0, 0.0, 0.0), (-1.0, 0.0))
    assert path[0] == (20.0, 0.0, 0.0)
    assert path[-1] == (0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trajectory.py -k "road_centerlines or quat_to_forward or nearest_road or road_path_from_teleport" -v`
Expected: FAIL with `ImportError` for the new names.

- [ ] **Step 3: Implement the helpers**

In `core/trajectory.py`, after `_extract_longest_road` add:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trajectory.py -k "road_centerlines or quat_to_forward or nearest_road or road_path_from_teleport" -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat: add nearest-road snap and heading-orientation helpers"
```

---

### Task 3: Teleport enumeration + multi-path `generate`/`load_or_generate`

**Files:**
- Modify: `core/trajectory.py`
- Test: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: Task 1 `MapTrajectories`, Task 2 helpers, existing `resample`, `heading_to_quat`, `_extract_longest_road`, `_square_loop_fallback`, constants `SPARSE_SPACING_M`, `DENSE_SPACING_M`, `SPAWN_Z_OFFSET_M`.
- Produces:
  - `MIN_PATH_SEPARATION_M = 30.0` (module constant).
  - `_teleport_points(bng) -> list[tuple[Vec3, Quat]]` — reads `bng.scenario.find_objects_class("SpawnSphere")`; falls back to `find_waypoints()`; returns `[]` on any error/empty.
  - `_path_from_teleport(tele_pos, tele_rot, roads, map_name) -> TrajectoryData | None` — builds one path; `None` if no usable road.
  - **Changed signatures:** `generate(bng, map_name) -> MapTrajectories` and `load_or_generate(map_name, bng) -> MapTrajectories`.

- [ ] **Step 1: Write the failing tests**

Replace the existing `test_generate_from_road_network`, `test_generate_falls_back_when_network_empty`, `test_load_or_generate_uses_cache`, `test_load_or_generate_generates_and_writes_when_missing`, and `test_load_or_generate_regenerates_on_corrupt_cache` in `tests/test_trajectory.py` with the versions below, and add the new teleport tests. Extend the `core.trajectory` import to include `MapTrajectories`, `_teleport_points`, `MIN_PATH_SEPARATION_M`.

```python
def _spawn_obj(pos, rot=(0.0, 0.0, 0.0, 1.0)):
    obj = MagicMock()
    obj.pos = pos
    obj.rot_quat = rot
    return obj


def _two_road_network():
    return {
        "north": {
            "edges": [
                {"middle": (0.0, 0.0, 0.0)},
                {"middle": (0.0, 50.0, 0.0)},
                {"middle": (0.0, 120.0, 0.0)},
            ],
        },
        "east": {
            "edges": [
                {"middle": (200.0, 0.0, 0.0)},
                {"middle": (250.0, 0.0, 0.0)},
            ],
        },
    }


def test_teleport_points_reads_spawnspheres():
    bng = MagicMock()
    bng.scenario.find_objects_class.return_value = [
        _spawn_obj((1.0, 2.0, 3.0)),
        _spawn_obj((4.0, 5.0, 6.0)),
    ]
    pts = _teleport_points(bng)
    bng.scenario.find_objects_class.assert_called_once_with("SpawnSphere")
    assert pts[0][0] == (1.0, 2.0, 3.0)
    assert len(pts) == 2


def test_teleport_points_empty_on_error():
    bng = MagicMock()
    bng.scenario.find_objects_class.side_effect = RuntimeError("boom")
    bng.scenario.find_waypoints.side_effect = RuntimeError("boom")
    assert _teleport_points(bng) == []


def test_generate_builds_one_path_per_teleport():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_objects_class.return_value = [
        _spawn_obj((0.0, 0.0, 0.0)),     # snaps to "north"
        _spawn_obj((201.0, 0.0, 0.0)),   # snaps to "east"
    ]
    mt = generate(bng, map_name="italy")
    assert isinstance(mt, MapTrajectories)
    assert mt.map_name == "italy"
    assert len(mt.paths) == 2
    # Sorted longest-road-first: the "north" road (120 m) precedes "east" (50 m).
    assert mt.paths[0].source == "teleport:north"
    assert mt.paths[1].source == "teleport:east"
    assert all(len(p.sparse_waypoints) >= 1 for p in mt.paths)


def test_generate_dedupes_nearby_teleports():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_objects_class.return_value = [
        _spawn_obj((0.0, 0.0, 0.0)),
        _spawn_obj((1.0, 0.0, 0.0)),  # within MIN_PATH_SEPARATION_M of the first
    ]
    mt = generate(bng, map_name="italy")
    assert len(mt.paths) == 1


def test_generate_falls_back_to_longest_road_without_teleports():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_objects_class.return_value = []
    bng.scenario.find_waypoints.return_value = []
    mt = generate(bng, map_name="italy")
    assert len(mt.paths) == 1
    assert mt.paths[0].source.startswith("road_network:")


def test_generate_falls_back_to_square_loop_without_roads():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = {}
    bng.scenario.find_objects_class.return_value = []
    bng.scenario.find_waypoints.return_value = []
    mt = generate(bng, map_name="smallgrid")
    assert len(mt.paths) == 1
    assert mt.paths[0].source == "fallback:square_loop"


def test_load_or_generate_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    mt = MapTrajectories(
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        paths=[_sample_traj(source="teleport:cached")],
    )
    (tmp_path / "italy.json").write_text(mt.to_json())
    bng = MagicMock()
    out = load_or_generate("italy", bng)
    bng.scenario.get_road_network.assert_not_called()
    assert out == mt


def test_load_or_generate_reads_old_single_object_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    (tmp_path / "italy.json").write_text(_sample_traj().to_json())  # old format
    out = load_or_generate("italy", MagicMock())
    assert isinstance(out, MapTrajectories)
    assert len(out.paths) == 1


def test_load_or_generate_generates_and_writes_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_objects_class.return_value = [_spawn_obj((0.0, 0.0, 0.0))]
    out = load_or_generate("italy", bng)
    assert (tmp_path / "italy.json").exists()
    assert MapTrajectories.from_json((tmp_path / "italy.json").read_text()) == out


def test_load_or_generate_raises_when_no_cache_and_no_bng(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="No cached trajectory"):
        load_or_generate("italy", bng=None)


def test_load_or_generate_regenerates_on_corrupt_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    (tmp_path / "italy.json").write_text("{not valid json")
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_objects_class.return_value = [_spawn_obj((0.0, 0.0, 0.0))]
    out = load_or_generate("italy", bng)
    assert out.map_name == "italy"
    MapTrajectories.from_json((tmp_path / "italy.json").read_text())
```

Also update the old `test_load_or_generate_uses_cache` helper expectations: it previously compared against a `TrajectoryData`; the new version above compares against a `MapTrajectories`. Delete the obsolete originals so there are no duplicate function names.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trajectory.py -v`
Expected: FAIL — `ImportError` for `_teleport_points`/`MIN_PATH_SEPARATION_M`, and the rewritten `generate`/`load_or_generate` tests fail because `generate` still returns a single `TrajectoryData`.

- [ ] **Step 3: Implement teleport enumeration + multi-path generation**

In `core/trajectory.py`:

Add the constant near the other spacing constants (after `FALLBACK_GROUND_Z`):

```python
MIN_PATH_SEPARATION_M = 30.0
```

Add helpers after `_road_path_from_teleport` (from Task 2):

```python
def _teleport_points(bng) -> list[tuple[Vec3, Quat]]:
    """Map teleport/spawn points as (pos, rot_quat).

    Reads BeamNG `SpawnSphere` objects; falls back to named waypoints. Returns
    an empty list if neither is available (older beamngpy or map without them).
    """
    for getter, args in (
        (lambda: bng.scenario.find_objects_class("SpawnSphere"), None),
        (lambda: bng.scenario.find_waypoints(), None),
    ):
        try:
            objs = getter()
        except Exception:
            continue
        pts: list[tuple[Vec3, Quat]] = []
        for o in objs or []:
            pos = tuple(getattr(o, "pos", None)) if getattr(o, "pos", None) else None
            if pos is None:
                continue
            rot = getattr(o, "rot_quat", None) or (0.0, 0.0, 0.0, 1.0)
            pts.append((pos, tuple(rot)))
        if pts:
            return pts
    return []


def _path_length(centerline: list[Vec3]) -> float:
    return sum(_segment_length(centerline[i], centerline[i + 1]) for i in range(len(centerline) - 1))


def _path_from_teleport(
    tele_pos: Vec3, tele_rot: Quat, roads: list[tuple[str, list[Vec3]]], map_name: str
) -> tuple[TrajectoryData, float] | None:
    """Build one TrajectoryData by snapping a teleport point to its nearest road.

    Returns (trajectory, road_length) for length-based sorting, or None when no
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
    sparse = resample(path, SPARSE_SPACING_M)
    dense = resample(path, DENSE_SPACING_M)
    spawn_pos = (tele_pos[0], tele_pos[1], tele_pos[2] + SPAWN_Z_OFFSET_M)
    traj = TrajectoryData(
        spawn_pos=spawn_pos,
        spawn_rot=tele_rot,
        sparse_waypoints=sparse,
        dense_waypoints=dense,
        map_name=map_name,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=f"teleport:{road_id}",
    )
    return traj, _path_length(path)
```

Replace the existing `generate(bng, map_name)` body with:

```python
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

    if roads and teleports:
        scored: list[tuple[TrajectoryData, float]] = []
        accepted_spawns: list[Vec3] = []
        for pos, rot in teleports:
            built = _path_from_teleport(pos, rot, roads, map_name)
            if built is None:
                continue
            traj, length = built
            if any(
                _segment_length(traj.spawn_pos, s) < MIN_PATH_SEPARATION_M
                for s in accepted_spawns
            ):
                continue
            accepted_spawns.append(traj.spawn_pos)
            scored.append((traj, length))
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
```

Replace the `load_or_generate` body so it round-trips `MapTrajectories`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trajectory.py -v`
Expected: PASS (all trajectory tests, old and new).

- [ ] **Step 5: Lint and commit**

```bash
python -m ruff check core/trajectory.py tests/test_trajectory.py
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat: generate one road-snapped path per teleport point"
```

---

### Task 4: Single-agent env consumes `paths[0]`

**Files:**
- Modify: `environments/beamng.py:404-420` (the `_resolve_trajectory` method)
- Test: `tests/test_trajectory.py` (a focused env test; no simulator)

**Interfaces:**
- Consumes: `load_or_generate(map_name, bng) -> MapTrajectories` (Task 3).
- Produces: `BeamNGDrivingEnv._resolve_trajectory()` still returns a single `TrajectoryData` (the longest road, `paths[0]`), so the rest of the env is unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trajectory.py`:

```python
def test_single_env_resolve_trajectory_takes_first_path(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    mt = MapTrajectories(
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        paths=[_sample_traj(source="teleport:first"), _sample_traj(source="teleport:second")],
    )
    (tmp_path / "italy.json").write_text(mt.to_json())

    from environments.beamng import BeamNGDrivingEnv

    env = BeamNGDrivingEnv(beamng_home="unused", map_name="italy")
    traj = env._resolve_trajectory()
    assert traj.source == "teleport:first"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_trajectory.py::test_single_env_resolve_trajectory_takes_first_path -v`
Expected: FAIL — `_resolve_trajectory` currently returns the `MapTrajectories` (no `.source`), raising `AttributeError`.

- [ ] **Step 3: Update `_resolve_trajectory` in `environments/beamng.py`**

The method has two `return load_or_generate(...)` sites (cache-hit and post-probe). Change both to take `.paths[0]`:

```python
    def _resolve_trajectory(self) -> TrajectoryData:
        """Return cached trajectory (longest road) or probe the map to generate one."""
        from core.trajectory import CACHE_DIR

        cache_path = CACHE_DIR / f"{self.map_name}.json"
        if cache_path.exists():
            return load_or_generate(self.map_name, bng=None).paths[0]

        probe = Scenario(self.map_name, "trajectory_probe", description="Road probe")
        probe_vehicle = Vehicle("probe_vehicle", model="etk800")
        probe.add_vehicle(probe_vehicle, pos=(0.0, 0.0, 100.0), rot_quat=(0.0, 0.0, 0.0, 1.0))
        probe.make(self.bng)
        self.bng.load_scenario(probe)
        self.bng.start_scenario()
        import time

        time.sleep(0.5)
        return load_or_generate(self.map_name, self.bng).paths[0]
```

Match the existing body exactly except for the two `.paths[0]` additions — keep whatever probe/sleep lines already exist; do not duplicate imports already present at the top of the method.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trajectory.py::test_single_env_resolve_trajectory_takes_first_path -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environments/beamng.py tests/test_trajectory.py
git commit -m "feat: single-agent env consumes paths[0] from MapTrajectories"
```

---

### Task 5: Migrate multi-agent env to per-slot waypoints (mechanical refactor)

**Files:**
- Modify: `environments/beamng_multi.py`
- Test: `tests/test_beamng_multi.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `VehicleSlot.waypoints: list[tuple[float, float, float]]` (new field, defaults to empty).
  - All per-vehicle path logic now reads `slot.waypoints` instead of the shared `self.waypoints`: `_path_errors`, `observe`, `_waypoint_hints`, `_update_slot_marker`, `_reward_default`, `_reward_ddpg`.
  - Behavior is unchanged this task: `launch()` still computes one shared path and copies it into every `slot.waypoints` (bridge removed in Task 6).

- [ ] **Step 1: Update the existing tests to use per-slot waypoints**

In `tests/test_beamng_multi.py`, every test that sets `env.waypoints = [...]` must instead set the slot it exercises. Apply these replacements:

In `TestPathErrorsAndReward`:
- `test_path_errors_advance_waypoint_when_close`: replace `env.waypoints = [...]` with `slot = env.slots[0]; slot.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]` (move the slot assignment above, drop `env.waypoints`).
- `test_default_reward_gives_checkpoint_bonus`: replace `env.waypoints = [...]` with `env.slots[0].waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]`.
- `test_default_reward_terminates_on_max_damage`: replace `env.waypoints = [(0.0, 0.0, 0.0)]` with `env.slots[0].waypoints = [(0.0, 0.0, 0.0)]`.
- `test_ddpg_reward_rewards_progress`: replace `env.waypoints = [...]` with `env.slots[1].waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]`.

In `TestObserve`, set the specific slot's waypoints (the one passed to `observe`):
- `test_observe_returns_vector_of_n_states`, `test_observe_polls_each_slot_sensor`, `test_observe_appends_extras_when_flags_on`: replace `env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]` with `slot.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]` (place after `slot = env.slots[0]`).

In `TestMarkers`:
- `test_update_slot_marker_adds_sphere_in_slot_color`: replace `env.waypoints = [(10.0, 20.0, 1.0), (30.0, 40.0, 1.0)]` with `slot.waypoints = [(10.0, 20.0, 1.0), (30.0, 40.0, 1.0)]` (after `slot = env.slots[0]`).
- `test_update_slot_marker_removes_previous`: replace `env.waypoints = [(0.0, 0.0, 0.0)]` with `slot.waypoints = [(0.0, 0.0, 0.0)]` (after `slot = env.slots[0]`).
- `test_update_slot_marker_noop_without_bng`: replace `env.waypoints = [(0.0, 0.0, 0.0)]` with `slot.waypoints = [(0.0, 0.0, 0.0)]` (after `slot = env.slots[0]`).

Add one new test asserting the field exists:

```python
def test_vehicle_slot_has_waypoints_field():
    s = _slot()
    assert s.waypoints == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beamng_multi.py -v`
Expected: FAIL — `observe`/`_path_errors`/`_update_slot_marker`/rewards still read `self.waypoints` (now unset on the env), and `VehicleSlot` has no `waypoints` field.

- [ ] **Step 3: Add the field and migrate the consumers**

In `environments/beamng_multi.py`, add to `VehicleSlot` (next to `spawn_pos`/`spawn_rot`, around line 69):

```python
    waypoints: list = field(default_factory=list)
```

In `_path_errors` (around line 320), change the guard and target lookups from `self.waypoints` to `slot.waypoints`:

```python
    def _path_errors(self, slot, pos, state):
        if not slot.waypoints or not state:
            slot.current_dist = 0.0
            return 0.0, 0.0, 0.0

        target = slot.waypoints[slot.waypoint_idx % len(slot.waypoints)]
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = float(np.hypot(dx, dy))
        slot.current_dist = dist

        if dist < self.WAYPOINT_RADIUS:
            slot.waypoint_idx += 1
            slot.checkpoint_hit = True
            self._update_slot_marker(slot)
            if slot.waypoint_idx < len(slot.waypoints):
                new_t = slot.waypoints[slot.waypoint_idx]
                slot.current_dist = float(np.hypot(new_t[0] - pos[0], new_t[1] - pos[1]))
        ...
```
(Leave the heading/lateral block below unchanged.)

In `observe` (around line 464), change the checkpoint-distance block:

```python
        slot.current_pos = pos
        if slot.waypoints:
            target = slot.waypoints[slot.waypoint_idx % len(slot.waypoints)]
            slot.checkpoint_dist = float(np.hypot(pos[0] - target[0], pos[1] - target[1]))
```

In `_waypoint_hints` (around line 520), change the guard and loop:

```python
        if not slot.trajectory_hints or not slot.waypoints:
            return np.empty(0, dtype=np.float32)
        ...
        for i in range(slot.trajectory_hints):
            idx = (slot.waypoint_idx + i) % len(slot.waypoints)
            wp = slot.waypoints[idx]
            ...
```

In `_reward_default` (around line 378) change `len(self.waypoints)` → `len(slot.waypoints)`:

```python
        if slot.waypoint_idx >= len(slot.waypoints):
            reward += 200.0
            done = True
```

In `_reward_ddpg` (around line 436) change `len(self.waypoints)` → `len(slot.waypoints)`:

```python
        if slot.waypoint_idx >= len(slot.waypoints):
            reward += 200.0
            slot.waypoint_idx = 0
            done = True
```

In `_update_slot_marker` (around line 670), change the guard and target lookup:

```python
        if self.bng is None or not slot.waypoints:
            return
        ...
            target = slot.waypoints[slot.waypoint_idx % len(slot.waypoints)]
```

In `launch()` (around line 549), bridge the shared path into every slot so behavior is unchanged this task:

```python
        self.trajectory = self._resolve_trajectory()
        self.waypoints = list(self.trajectory.sparse_waypoints)
        for slot in self.slots:
            slot.waypoints = list(self.waypoints)
        self._load_scenario()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng_multi.py -v`
Expected: PASS (all multi tests, including the new field test).

- [ ] **Step 5: Lint and commit**

```bash
python -m ruff check environments/beamng_multi.py tests/test_beamng_multi.py
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "refactor: multi-agent env reads per-slot waypoints"
```

---

### Task 6: Assign one path per vehicle, hard-error on overflow, drop the starting grid

**Files:**
- Modify: `environments/beamng_multi.py`
- Test: `tests/test_beamng_multi.py`

**Interfaces:**
- Consumes: `MapTrajectories`/`TrajectoryData` (Task 1/3), `load_or_generate -> MapTrajectories`.
- Produces:
  - `BeamNGMultiEnv.trajectories: MapTrajectories | None` (replaces the single `self.trajectory`).
  - `BeamNGMultiEnv._assign_paths()` — sets each slot's `waypoints`/`spawn_pos`/`spawn_rot` from `trajectories.paths[i]`; raises `ValueError` when `len(slots) > len(paths)`.
  - Removed: `_grid_pose`, `_spawn_axes`, `GRID_LANE_OFFSET`, and the shared `self.waypoints` attribute.

- [ ] **Step 1: Update tests — drop grid tests, add assignment tests**

In `tests/test_beamng_multi.py`:

Delete the entire `class TestStartingGrid` (it tests `_grid_pose`/`_spawn_axes`/`GRID_LANE_OFFSET`, all removed).

Add a new class (uses real `MapTrajectories`/`TrajectoryData`):

```python
class TestPathAssignment:
    def _mt(self, n_paths):
        from core.trajectory import MapTrajectories, TrajectoryData

        paths = [
            TrajectoryData(
                spawn_pos=(float(i) * 100.0, 0.0, 1.0),
                spawn_rot=(0.0, 0.0, 0.0, 1.0),
                sparse_waypoints=[(float(i) * 100.0, 10.0, 0.0), (float(i) * 100.0, 20.0, 0.0)],
                dense_waypoints=[(float(i) * 100.0, 10.0, 0.0)],
                map_name="italy",
                generated_at="2026-06-18T12:00:00+00:00",
                source=f"teleport:r{i}",
            )
            for i in range(n_paths)
        ]
        return MapTrajectories(
            map_name="italy", generated_at="2026-06-18T12:00:00+00:00", paths=paths
        )

    def test_each_slot_gets_its_own_path(self):
        env = _env()  # 3 slots
        env.trajectories = self._mt(3)
        env._assign_paths()
        assert env.slots[0].spawn_pos == (0.0, 0.0, 1.0)
        assert env.slots[1].spawn_pos == (100.0, 0.0, 1.0)
        assert env.slots[2].spawn_pos == (200.0, 0.0, 1.0)
        assert env.slots[0].waypoints == [(0.0, 10.0, 0.0), (0.0, 20.0, 0.0)]
        assert env.slots[1].waypoints[0] == (100.0, 10.0, 0.0)
        # Distinct spawns -> no shared start line.
        assert len({s.spawn_pos for s in env.slots}) == 3

    def test_more_vehicles_than_paths_raises(self):
        env = _env()  # 3 slots
        env.trajectories = self._mt(2)
        with pytest.raises(ValueError, match="only 2 distinct path"):
            env._assign_paths()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beamng_multi.py::TestPathAssignment -v`
Expected: FAIL — `_assign_paths` and `env.trajectories` do not exist yet.

- [ ] **Step 3: Implement assignment, drop the grid**

In `environments/beamng_multi.py`:

In `__init__` (around line 283-286), replace the trajectory/waypoints attributes:

```python
        self.bng: BeamNGpy = None
        self.scenario: Scenario = None
        self.trajectories: MapTrajectories | None = None
```
(Remove the `self.trajectory` and `self.waypoints` lines.)

Update the import at the top (around line 23):

```python
from core.trajectory import MapTrajectories, load_or_generate
```

Remove the `GRID_LANE_OFFSET = 4.0` class constant (around line 255). Keep `HALF_TRACK_WIDTH`.

Replace `launch()` (around line 539) so it resolves all paths and assigns them:

```python
    def launch(self):
        """Start BeamNG, resolve all map paths, and load the multi-vehicle scenario."""
        self.bng = BeamNGpy(
            self.host,
            self.port,
            home=self.beamng_home,
            user=self.beamng_user,
            headless=self.headless,
        )
        self.bng.open(launch=True)
        self.trajectories = self._resolve_trajectory()
        self._assign_paths()
        self._load_scenario()

    def _assign_paths(self):
        """Give each vehicle its own path; error if vehicles outnumber paths."""
        paths = self.trajectories.paths
        if len(self.slots) > len(paths):
            raise ValueError(
                f"{len(self.slots)} vehicles requested but map '{self.map_name}' has "
                f"only {len(paths)} distinct path(s). Reduce the vehicle count to "
                f"<= {len(paths)} or pick a map with more teleport points."
            )
        for slot, path in zip(self.slots, paths):
            slot.waypoints = list(path.sparse_waypoints)
            slot.spawn_pos = path.spawn_pos
            slot.spawn_rot = path.spawn_rot
```

Replace `_resolve_trajectory` (around line 553) to return the full `MapTrajectories`:

```python
    def _resolve_trajectory(self):
        import time

        from core.trajectory import CACHE_DIR

        cache_path = CACHE_DIR / f"{self.map_name}.json"
        if cache_path.exists():
            return load_or_generate(self.map_name, bng=None)

        probe = Scenario(self.map_name, "trajectory_probe", description="Road probe")
        probe_vehicle = Vehicle("probe_vehicle", model="etk800")
        probe.add_vehicle(probe_vehicle, pos=(0.0, 0.0, 100.0), rot_quat=(0.0, 0.0, 0.0, 1.0))
        probe.make(self.bng)
        self.bng.load_scenario(probe)
        self.bng.start_scenario()
        time.sleep(0.5)
        return load_or_generate(self.map_name, self.bng)
```

In `_load_scenario` (around line 571), use each slot's already-assigned spawn (remove the `_grid_pose` call) and add the union of all waypoints as checkpoints:

```python
        for slot in self.slots:
            vcfg = self.VEHICLES.get(slot.vehicle_id, self.VEHICLES["taxi"])
            vcfg = {**vcfg, "color": slot.color}
            slot.vehicle = Vehicle(slot.name, **vcfg)
            slot.electrics = Electrics()
            slot.damage_sensor = Damage()
            slot.vehicle.attach_sensor("electrics", slot.electrics)
            slot.vehicle.attach_sensor("damage", slot.damage_sensor)
            self.scenario.add_vehicle(
                slot.vehicle,
                pos=slot.spawn_pos,
                rot_quat=slot.spawn_rot,
                cling=True,
            )

        all_waypoints = [wp for slot in self.slots for wp in slot.waypoints]
        scales = [(5.0, 5.0, 1.0)] * len(all_waypoints)
        self.scenario.add_checkpoints(all_waypoints, scales)
```
(Drop the `slot.spawn_pos, slot.spawn_rot = self._grid_pose(i)` line and the `enumerate`; iterate `self.slots` directly.)

Delete the `_spawn_axes` method (around line 690) and the `_grid_pose` method (around line 702) entirely.

- [ ] **Step 4: Run the full multi-agent test file to verify it passes**

Run: `python -m pytest tests/test_beamng_multi.py -v`
Expected: PASS — `TestPathAssignment` green, `TestStartingGrid` gone, all other multi tests still green.

- [ ] **Step 5: Lint and commit**

```bash
python -m ruff check environments/beamng_multi.py tests/test_beamng_multi.py
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "feat: assign one path per vehicle, hard-error on overflow"
```

---

### Task 7: CLI — report path count after generation

**Files:**
- Modify: `core/cli.py`
- Test: `tests/test_cli_trajectory_summary.py` (new)

**Interfaces:**
- Consumes: `MapTrajectories` (Task 1).
- Produces: `format_trajectory_summary(mt: MapTrajectories) -> str` in `core/cli.py` — a one-line human summary of how many paths a map has and their sources.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_trajectory_summary.py`:

```python
from core.cli import format_trajectory_summary
from core.trajectory import MapTrajectories, TrajectoryData


def _traj(source):
    return TrajectoryData(
        spawn_pos=(0.0, 0.0, 1.0),
        spawn_rot=(0.0, 0.0, 0.0, 1.0),
        sparse_waypoints=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        dense_waypoints=[(0.0, 0.0, 0.0)],
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        source=source,
    )


def test_summary_reports_path_count():
    mt = MapTrajectories(
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        paths=[_traj("teleport:r1"), _traj("teleport:r2")],
    )
    summary = format_trajectory_summary(mt)
    assert "2 path" in summary
    assert "italy" in summary
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_trajectory_summary.py -v`
Expected: FAIL — `ImportError: cannot import name 'format_trajectory_summary'`.

- [ ] **Step 3: Implement the helper and wire it into the trajectory menu**

In `core/cli.py`, add near the top-level helpers (after the imports):

```python
def format_trajectory_summary(mt) -> str:
    """One-line summary of a MapTrajectories: path count + per-path sources."""
    n = len(mt.paths)
    sources = ", ".join(p.source for p in mt.paths)
    return f"{mt.map_name}: {n} path(s) [{sources}]"
```

In `_trajectory_menu` (around line 409-423), after generating, load the cache and print the summary. Replace the success print with:

```python
        try:
            env.reset()
            from core.trajectory import load_or_generate

            mt = load_or_generate(map_name, bng=None)
            print(f"    Done. {format_trajectory_summary(mt)}")
        finally:
            env.close()
```
(The single-agent `env.reset()` writes the cache via `_resolve_trajectory`; `load_or_generate(..., bng=None)` then reads it back as a `MapTrajectories` for the summary.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli_trajectory_summary.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
python -m ruff check core/cli.py tests/test_cli_trajectory_summary.py
git add core/cli.py tests/test_cli_trajectory_summary.py
git commit -m "feat: CLI reports generated path count per map"
```

---

### Task 8: Full suite + docs note

**Files:**
- Modify: `docs/romain.md`
- Test: whole suite.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS (no regressions across the repo).

- [ ] **Step 2: Run the linter over all touched files**

Run: `python -m ruff check core/trajectory.py environments/beamng.py environments/beamng_multi.py core/cli.py tests/`
Expected: no errors.

- [ ] **Step 3: Append a note to `docs/romain.md`**

Add under the "Sixth issue" paragraph (which mentions vehicles colliding and shared LiDAR):

```markdown
Follow-up to the sixth issue: trajectory generation now emits one road-snapped
path per map teleport point (`SpawnSphere`). Multi-agent training assigns one
path per vehicle in a different part of the map, so vehicles no longer share a
start line or collide. If more vehicles than paths are requested, the session
errors out rather than doubling up.
```

- [ ] **Step 4: Commit**

```bash
git add docs/romain.md
git commit -m "docs: note per-teleport multi-path training in romain.md"
```

---

## Self-Review

**Spec coverage:**
- Teleport-point source (`SpawnSphere` + `find_waypoints` fallback) → Task 3 `_teleport_points`. ✓
- Snap to nearest road + orient to heading → Task 2 `_nearest_road`/`_road_path_from_teleport`, used in Task 3. ✓
- Dedup (30 m) + longest-road sort → Task 3 `generate`. ✓
- Fallbacks (longest road, square loop) → Task 3 `generate`. ✓
- `MapTrajectories` + old-format back-compat → Task 1. ✓
- `load_or_generate` returns `MapTrajectories` → Task 3. ✓
- Single env takes `paths[0]` → Task 4. ✓
- Per-slot waypoints across `_path_errors`/`observe`/`_waypoint_hints`/`_update_slot_marker`/rewards → Task 5. ✓
- Hard-error cap + drop grid + checkpoint union + path assignment → Task 6. ✓
- CLI path-count messaging → Task 7. ✓
- Testing (pure functions, mocked BeamNG) → Tasks 1-7. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `generate`/`load_or_generate` return `MapTrajectories` everywhere after Task 3; `_resolve_trajectory` returns `TrajectoryData` (single env) vs `MapTrajectories` (multi env) — intentional and matched to each env's call sites. `_assign_paths`/`_path_from_teleport`/`_road_path_from_teleport` signatures match their consumers. `VehicleSlot.waypoints` added in Task 5, consumed in Tasks 5-6. ✓
