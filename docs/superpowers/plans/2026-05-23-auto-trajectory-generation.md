# Auto-Trajectory Generation per Map — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `DEFAULT_WAYPOINTS` / `DDPG_WAYPOINTS` / `SPAWN_POS` / `SPAWN_ROT` in `environments/beamng.py` with a per-map automatic generation system. For each BeamNG map the wrapper produces a deterministic spawn pose and two waypoint lists (sparse for discrete algos, dense for continuous), extracted once from `scenario.get_road_network(...)` and cached on disk as JSON.

**Architecture:** A new self-contained module `core/trajectory.py` owns the dataclass, JSON (de)serialization, the resampling/heading math, the road-extraction logic and the cache. `environments/beamng.py` does the two-phase BeamNG launch: probe → save cache → real scenario load using `trajectory.spawn_*` and `trajectory.*_waypoints`. A new menu entry pre-warms the cache for one or all maps.

**Tech Stack:** Python 3.11, `beamngpy>=1.28`, `numpy`, `pytest` (added to `requirements.txt`).

**Spec:** See `docs/superpowers/specs/2026-05-23-auto-trajectory-generation-design.md`.

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `core/trajectory.py` | `TrajectoryData` dataclass, resample, heading→quat, road extraction, fallback, cache I/O |
| Create | `tests/__init__.py` | makes `tests/` a package |
| Create | `tests/test_trajectory.py` | unit tests for the pure functions in `core/trajectory.py` (no BeamNG required) |
| Modify | `environments/beamng.py` | drop the four hardcoded constants, two-phase launch, derive `self.waypoints` from `self.trajectory` |
| Modify | `core/cli.py` | new `_trajectory_menu()` function + new option in `main_menu()` |
| Modify | `requirements.txt` | add `pytest>=8.0` |
| Modify | `scenario_creator.md` | point to the auto flow as primary, keep the manual flow as fallback |
| Modify | `README.md` | mention `outputs/trajectories/` cache + new menu option |
| Modify | `.github/workflows/ci.yml` | add a pytest step that runs `pytest tests/` |

---

## Task 1: Add pytest to dependencies & create `tests/` package

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add pytest to `requirements.txt`**

Edit `requirements.txt` — append below the existing dependencies (before the BeamNG section):

```
pytest>=8.0
```

- [ ] **Step 2: Create empty `tests/__init__.py`**

Create file `tests/__init__.py` with a single line:
```python
"""Pytest test suite for the RL pipeline."""
```

- [ ] **Step 3: Install pytest**

Run: `pip install pytest>=8.0`
Expected: pytest successfully installed.

- [ ] **Step 4: Confirm pytest discovers the empty suite**

Run: `pytest tests/ -v`
Expected: `no tests ran in 0.0xs` (no error).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "chore: add pytest dependency and tests package"
```

---

## Task 2: `TrajectoryData` dataclass + JSON round-trip

**Files:**
- Create: `core/trajectory.py`
- Create: `tests/test_trajectory.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trajectory.py`:

```python
"""Unit tests for core.trajectory."""
import json

from core.trajectory import TrajectoryData


def test_trajectorydata_json_roundtrip():
    data = TrajectoryData(
        spawn_pos=(1.0, 2.0, 3.0),
        spawn_rot=(0.0, 0.0, 0.707, 0.707),
        sparse_waypoints=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        dense_waypoints=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        map_name="gridmap_v2",
        generated_at="2026-05-23T12:00:00Z",
        source="road_network:road_42",
    )

    payload = data.to_json()
    parsed = json.loads(payload)
    assert parsed["map_name"] == "gridmap_v2"
    assert parsed["spawn_pos"] == [1.0, 2.0, 3.0]
    assert parsed["sparse_waypoints"][1] == [10.0, 0.0, 0.0]

    restored = TrajectoryData.from_json(payload)
    assert restored == data
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_trajectory.py::test_trajectorydata_json_roundtrip -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.trajectory'`.

- [ ] **Step 3: Implement `TrajectoryData`**

Create `core/trajectory.py`:

```python
"""Automatic per-map trajectory generation for BeamNG environments."""

from __future__ import annotations

import json
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_trajectory.py::test_trajectorydata_json_roundtrip -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat(trajectory): add TrajectoryData dataclass with JSON I/O"
```

---

## Task 3: `resample()` — arc-length resampling

**Files:**
- Modify: `core/trajectory.py`
- Modify: `tests/test_trajectory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trajectory.py`:

```python
import math

import pytest

from core.trajectory import resample


def test_resample_straight_line_uniform_spacing():
    # A 30 m straight line on the X axis, resampled at 10 m → 4 points (0, 10, 20, 30)
    path = [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)]
    out = resample(path, spacing=10.0)
    assert len(out) == 4
    assert out[0] == (0.0, 0.0, 0.0)
    assert out[-1] == (30.0, 0.0, 0.0)
    for i in range(len(out) - 1):
        d = math.hypot(out[i + 1][0] - out[i][0], out[i + 1][1] - out[i][1])
        assert d == pytest.approx(10.0, abs=1e-6)


def test_resample_preserves_endpoints_with_remainder():
    # 25 m line, spacing 10 m → samples at 0, 10, 20, and 25 (last point preserved)
    path = [(0.0, 0.0, 0.0), (25.0, 0.0, 0.0)]
    out = resample(path, spacing=10.0)
    assert out[0] == (0.0, 0.0, 0.0)
    assert out[-1] == (25.0, 0.0, 0.0)
    # Inner points spaced 10 m
    assert out[1] == pytest.approx((10.0, 0.0, 0.0), abs=1e-6)
    assert out[2] == pytest.approx((20.0, 0.0, 0.0), abs=1e-6)


def test_resample_two_segment_polyline():
    # L-shape: (0,0)→(10,0)→(10,10), 20 m total, spacing 5 m → 5 points
    path = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)]
    out = resample(path, spacing=5.0)
    assert len(out) == 5
    assert out[0] == (0.0, 0.0, 0.0)
    assert out[2] == pytest.approx((10.0, 0.0, 0.0), abs=1e-6)
    assert out[-1] == (10.0, 10.0, 0.0)


def test_resample_rejects_short_path():
    with pytest.raises(ValueError):
        resample([(0.0, 0.0, 0.0)], spacing=5.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_trajectory.py -v -k resample`
Expected: 4 FAILs with `ImportError: cannot import name 'resample'`.

- [ ] **Step 3: Implement `resample`**

Append to `core/trajectory.py`:

```python
import math


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_trajectory.py -v -k resample`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat(trajectory): add arc-length resample() with unit tests"
```

---

## Task 4: `heading_to_quat()` — cardinal directions

**Files:**
- Modify: `core/trajectory.py`
- Modify: `tests/test_trajectory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trajectory.py`:

```python
from core.trajectory import heading_to_quat


def test_heading_to_quat_north():
    # +Y direction → identity-ish quaternion in scenario_creator.md table
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (0.0, 10.0, 0.0))
    assert (qx, qy) == (0.0, 0.0)
    # heading = +pi/2 → z = sin(pi/4) ≈ 0.707, w = cos(pi/4) ≈ 0.707
    assert qz == pytest.approx(math.sin(math.pi / 4), abs=1e-6)
    assert qw == pytest.approx(math.cos(math.pi / 4), abs=1e-6)


def test_heading_to_quat_east():
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    # heading = 0 → z = 0, w = 1
    assert qz == pytest.approx(0.0, abs=1e-6)
    assert qw == pytest.approx(1.0, abs=1e-6)


def test_heading_to_quat_west():
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (-10.0, 0.0, 0.0))
    # heading = pi → z = 1, w = 0
    assert abs(qz) == pytest.approx(1.0, abs=1e-6)
    assert qw == pytest.approx(0.0, abs=1e-6)


def test_heading_to_quat_rejects_zero_delta():
    with pytest.raises(ValueError):
        heading_to_quat((1.0, 1.0, 0.0), (1.0, 1.0, 0.0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_trajectory.py -v -k heading`
Expected: 4 FAILs (function not defined).

- [ ] **Step 3: Implement `heading_to_quat`**

Append to `core/trajectory.py`:

```python
def heading_to_quat(p0: Vec3, p1: Vec3) -> Quat:
    """Quaternion (x, y, z, w) that rotates the +X axis to point from p0 to p1.

    Uses only the XY components; vertical delta is ignored.  Matches the
    convention documented in `scenario_creator.md` (East = (0,0,0,1),
    North = (0,0,0.707,0.707), etc.).
    """
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    if dx == 0.0 and dy == 0.0:
        raise ValueError("p0 and p1 must differ in the XY plane")
    heading = math.atan2(dy, dx)
    return (0.0, 0.0, math.sin(heading / 2.0), math.cos(heading / 2.0))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_trajectory.py -v -k heading`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat(trajectory): add heading_to_quat() with cardinal-direction tests"
```

---

## Task 5: `_square_loop_fallback()` — geometric fallback for empty maps

**Files:**
- Modify: `core/trajectory.py`
- Modify: `tests/test_trajectory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trajectory.py`:

```python
from core.trajectory import _square_loop_fallback


def test_square_loop_fallback_topology():
    traj = _square_loop_fallback(map_name="smallgrid")
    # 80 m square, perimeter 320 m
    # Sparse 25 m → ~13 samples; dense 8 m → ~40 samples
    assert traj.map_name == "smallgrid"
    assert traj.source == "fallback:square_loop"
    assert len(traj.sparse_waypoints) >= 12
    assert len(traj.sparse_waypoints) <= 16
    assert len(traj.dense_waypoints) >= 35
    # First waypoint is the spawn point
    assert traj.spawn_pos[:2] == traj.sparse_waypoints[0][:2]
    # Spawn is above the road (z offset)
    assert traj.spawn_pos[2] > traj.sparse_waypoints[0][2]


def test_square_loop_corners_are_at_expected_positions():
    traj = _square_loop_fallback(map_name="smallgrid")
    # Corners of an 80 m square around origin → expect points near (40,-40), (40,40), (-40,40), (-40,-40)
    xy = [(p[0], p[1]) for p in traj.sparse_waypoints]
    for cx, cy in [(40.0, -40.0), (40.0, 40.0), (-40.0, 40.0), (-40.0, -40.0)]:
        assert any(
            math.hypot(x - cx, y - cy) < 1e-3 for x, y in xy
        ), f"missing corner near ({cx}, {cy})"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_trajectory.py -v -k fallback`
Expected: 2 FAILs.

- [ ] **Step 3: Implement `_square_loop_fallback`**

Append to `core/trajectory.py`:

```python
from datetime import datetime, timezone

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_trajectory.py -v -k fallback`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat(trajectory): add square-loop fallback for empty maps"
```

---

## Task 6: `_extract_longest_road()` — pick longest drivable road

**Files:**
- Modify: `core/trajectory.py`
- Modify: `tests/test_trajectory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trajectory.py`:

```python
from core.trajectory import _extract_longest_road, _edge_center


def test_edge_center_prefers_middle_key():
    edge = {"middle": (5.0, 5.0, 1.0), "left": (0.0, 0.0, 1.0), "right": (10.0, 10.0, 1.0)}
    assert _edge_center(edge) == (5.0, 5.0, 1.0)


def test_edge_center_falls_back_to_left_right_midpoint():
    edge = {"left": (0.0, 0.0, 1.0), "right": (10.0, 4.0, 1.0)}
    assert _edge_center(edge) == (5.0, 2.0, 1.0)


def test_extract_longest_road_picks_longest():
    network = {
        "short_road": {
            "edges": [
                {"middle": (0.0, 0.0, 0.0)},
                {"middle": (5.0, 0.0, 0.0)},
            ],
        },
        "long_road": {
            "edges": [
                {"middle": (0.0, 0.0, 0.0)},
                {"middle": (50.0, 0.0, 0.0)},
                {"middle": (100.0, 0.0, 0.0)},
            ],
        },
    }
    road_id, centerline = _extract_longest_road(network)
    assert road_id == "long_road"
    assert centerline[0] == (0.0, 0.0, 0.0)
    assert centerline[-1] == (100.0, 0.0, 0.0)


def test_extract_longest_road_returns_none_for_empty_network():
    assert _extract_longest_road({}) == (None, None)


def test_extract_longest_road_skips_single_edge_roads():
    network = {"degenerate": {"edges": [{"middle": (0.0, 0.0, 0.0)}]}}
    assert _extract_longest_road(network) == (None, None)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_trajectory.py -v -k "edge or extract"`
Expected: 5 FAILs.

- [ ] **Step 3: Implement `_edge_center` and `_extract_longest_road`**

Append to `core/trajectory.py`:

```python
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
        length = sum(_segment_length(centerline[i], centerline[i + 1]) for i in range(len(centerline) - 1))
        if length > best_length:
            best_length = length
            best_id = road_id
            best_centerline = centerline

    if best_centerline is None or best_length == 0.0:
        return (None, None)
    return (best_id, best_centerline)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_trajectory.py -v -k "edge or extract"`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat(trajectory): extract longest drivable road from road_network dict"
```

---

## Task 7: `generate()` + `load_or_generate()` cache orchestration

**Files:**
- Modify: `core/trajectory.py`
- Modify: `tests/test_trajectory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_trajectory.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock

from core.trajectory import generate, load_or_generate, CACHE_DIR


def test_generate_from_road_network():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = {
        "main_road": {
            "edges": [
                {"middle": (0.0, 0.0, 0.0)},
                {"middle": (50.0, 0.0, 0.0)},
                {"middle": (100.0, 0.0, 0.0)},
            ],
        },
    }
    traj = generate(bng, map_name="italy")
    bng.scenario.get_road_network.assert_called_once_with(include_edges=True, drivable_only=True)
    assert traj.map_name == "italy"
    assert traj.source.startswith("road_network:main_road")
    assert len(traj.sparse_waypoints) >= 4
    assert len(traj.dense_waypoints) > len(traj.sparse_waypoints)
    assert traj.spawn_pos[2] > traj.sparse_waypoints[0][2]  # z offset above road


def test_generate_falls_back_when_network_empty():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = {}
    traj = generate(bng, map_name="smallgrid")
    assert traj.source == "fallback:square_loop"


def test_load_or_generate_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    cached = TrajectoryData(
        spawn_pos=(1.0, 2.0, 3.0),
        spawn_rot=(0.0, 0.0, 0.707, 0.707),
        sparse_waypoints=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        dense_waypoints=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        map_name="italy",
        generated_at="2026-05-23T12:00:00+00:00",
        source="road_network:cached",
    )
    (tmp_path / "italy.json").write_text(cached.to_json())

    bng = MagicMock()
    out = load_or_generate("italy", bng)
    bng.scenario.get_road_network.assert_not_called()
    assert out == cached


def test_load_or_generate_generates_and_writes_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = {
        "r": {
            "edges": [
                {"middle": (0.0, 0.0, 0.0)},
                {"middle": (100.0, 0.0, 0.0)},
            ],
        },
    }
    out = load_or_generate("italy", bng)
    assert (tmp_path / "italy.json").exists()
    # Round-trip the file
    assert TrajectoryData.from_json((tmp_path / "italy.json").read_text()) == out


def test_load_or_generate_raises_when_no_cache_and_no_bng(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="No cached trajectory"):
        load_or_generate("italy", bng=None)


def test_load_or_generate_regenerates_on_corrupt_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    (tmp_path / "italy.json").write_text("{not valid json")
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = {
        "r": {
            "edges": [
                {"middle": (0.0, 0.0, 0.0)},
                {"middle": (100.0, 0.0, 0.0)},
            ],
        },
    }
    out = load_or_generate("italy", bng)
    assert out.map_name == "italy"
    # Cache rewritten with valid content
    TrajectoryData.from_json((tmp_path / "italy.json").read_text())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_trajectory.py -v -k "generate or load_or"`
Expected: 6 FAILs.

- [ ] **Step 3: Implement `generate` + `load_or_generate`**

Append to `core/trajectory.py`:

```python
from pathlib import Path

CACHE_DIR = Path("outputs/trajectories")


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
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_trajectory.py -v -k "generate or load_or"`
Expected: 6 PASS. Run full suite to confirm: `pytest tests/ -v` (all green).

- [ ] **Step 5: Commit**

```bash
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat(trajectory): add generate() + load_or_generate() with disk cache"
```

---

## Task 8: Wire `core/trajectory` into `environments/beamng.py`

**Files:**
- Modify: `environments/beamng.py`

- [ ] **Step 1: Remove the four hardcoded constants**

In `environments/beamng.py`, **delete** the following blocks (between lines 60 and 91 of the current file):

```python
# Original sparse waypoints (for discrete-action algos: DQN, Q-learning)
DEFAULT_WAYPOINTS = [
    (61.0, -755.0, 100.0),
    (90.0, -734.0, 100.0),
    (116.0, -612.0, 100.0),
]

# Dense waypoints for continuous-action algos (DDPG, TD3)
DDPG_WAYPOINTS = [
    (61.0, -773.0, 100.0),
    ...
    (116.0, -612.0, 100.0),
]
```

and

```python
SPAWN_POS = (61.0, -788.0, 101.0)
SPAWN_ROT = (0.0, 0.0, 1.0, 0.0)
```

Keep `WAYPOINT_RADIUS`, `MAX_STEPS`, `MAX_DAMAGE`, `CHECKPOINT_WARN_DIST`, `CHECKPOINT_RESET_DIST` unchanged.

- [ ] **Step 2: Import the trajectory module**

Add at the top of `environments/beamng.py` (alongside the other imports):

```python
from core.trajectory import TrajectoryData, load_or_generate
```

- [ ] **Step 3: Initialise `self.trajectory` in `__init__`**

In `BeamNGDrivingEnv.__init__`, replace the existing block:

```python
        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._last_dist = 0.0
        self._steps = 0
        self._active_marker_id: str | None = None
        self.waypoints = list(self.DEFAULT_WAYPOINTS)
        self._current_pos = self.SPAWN_POS
        self._checkpoint_dist = 0.0
        self.headless = headless

        if self.reward_mode in ("ddpg"):
            self.waypoints = list(self.DEFAULT_WAYPOINTS)
        else:
            self.waypoints = list(self.DEFAULT_WAYPOINTS)
```

with:

```python
        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._last_dist = 0.0
        self._steps = 0
        self._active_marker_id: str | None = None
        self._checkpoint_dist = 0.0
        self.headless = headless

        # Filled on first _launch() — either read from cache or generated then.
        self.trajectory: TrajectoryData | None = None
        self.waypoints: list[tuple[float, float, float]] = []
        self._current_pos = (0.0, 0.0, 0.0)
```

- [ ] **Step 4: Add a helper that picks sparse vs dense**

Add this method just below `__init__`:

```python
    def _select_waypoints(self) -> list[tuple[float, float, float]]:
        assert self.trajectory is not None
        use_dense = self.reward_mode == "ddpg" or isinstance(self, BeamNGContinuousEnv)
        return list(self.trajectory.dense_waypoints if use_dense else self.trajectory.sparse_waypoints)
```

(Note: `BeamNGContinuousEnv` is defined further down in the same file — Python resolves the name at call time, so the forward reference is fine.)

- [ ] **Step 5: Modify `_launch` to load the trajectory before the scenario**

Replace the body of `_launch`:

```python
    def _launch(self, human_control=False):
        """Start BeamNG.drive and load the scenario for the first time."""
        self.bng = BeamNGpy(
            self.host,
            self.port,
            home=self.beamng_home,
            user=self.beamng_user,
            headless=self.headless,
        )
        self.bng.open(launch=True)
        self.trajectory = self._resolve_trajectory()
        self.waypoints = self._select_waypoints()
        self._current_pos = self.trajectory.spawn_pos
        self._load_scenario(human_control=human_control)

    def _resolve_trajectory(self) -> TrajectoryData:
        """Return cached trajectory or probe the map to generate one."""
        from core.trajectory import CACHE_DIR
        cache_path = CACHE_DIR / f"{self.map_name}.json"
        if cache_path.exists():
            return load_or_generate(self.map_name, bng=None)

        # No cache → run a probe scenario so we can call get_road_network
        probe = Scenario(self.map_name, "trajectory_probe", description="Road probe")
        probe_vehicle = Vehicle("probe_vehicle", model="etk800")
        probe.add_vehicle(probe_vehicle, pos=(0.0, 0.0, 100.0), rot_quat=(0.0, 0.0, 0.0, 1.0))
        probe.make(self.bng)
        self.bng.load_scenario(probe)
        self.bng.start_scenario()
        time.sleep(0.5)
        try:
            return load_or_generate(self.map_name, self.bng)
        finally:
            # The next call to _load_scenario will replace this probe.
            pass
```

- [ ] **Step 6: Replace `SPAWN_POS` / `SPAWN_ROT` usages in `_load_scenario`**

In `BeamNGDrivingEnv._load_scenario`, replace:

```python
        self.scenario.add_vehicle(
            self.vehicle,
            pos=self.SPAWN_POS,
            rot_quat=self.SPAWN_ROT,
        )
```

with:

```python
        self.scenario.add_vehicle(
            self.vehicle,
            pos=self.trajectory.spawn_pos,
            rot_quat=self.trajectory.spawn_rot,
        )
```

Do the same edit inside `BeamNGRadarEnv._load_scenario` and `BeamNGCameraEnv._load_scenario` (two more occurrences of `pos=self.SPAWN_POS, rot_quat=self.SPAWN_ROT`).

- [ ] **Step 7: Smoke-import the module**

Run:
```bash
python -c "import environments; from environments.beamng import BeamNGDrivingEnv; print('imports OK')"
```
Expected: `imports OK` (no `AttributeError` referencing the removed constants).

- [ ] **Step 8: Run the existing pipeline smoke test (Taxi path)**

Run:
```bash
python -c "
import algorithms, environments
from core.registry import registry
from core.runner import PipelineRunner
algo = registry.get_algorithm('q_learning')
env_info = registry.get_environment('taxi')
meta = env_info['metadata']
agent = algo['class'](n_states=meta['n_states'], n_actions=meta['n_actions'], **algo['default_config'])
env = env_info['factory']()
runner = PipelineRunner()
runner.train(agent, env, n_episodes=10)
env.close()
print('Taxi smoke OK')
"
```
Expected: `Taxi smoke OK` (this confirms we didn't break the Taxi side of the pipeline).

- [ ] **Step 9: Run the unit tests**

Run: `pytest tests/ -v`
Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add environments/beamng.py
git commit -m "feat(beamng): use auto-generated per-map trajectory instead of constants"
```

---

## Task 9: New CLI menu entry — pre-warm trajectory cache

**Files:**
- Modify: `core/cli.py`

- [ ] **Step 1: Add a trajectory-generation menu function**

Add this new function in `core/cli.py` (just above `def main_menu()`):

```python
def _trajectory_menu():
    """Pre-warm the trajectory cache for one or all BeamNG maps."""
    print("\n--- Generate Trajectories ---")
    print("This will launch BeamNG and probe each map's road network.")
    print("The result is cached in outputs/trajectories/<map>.json.\n")

    options = _BEAMNG_MAPS + ["all"]
    choice = _pick(options, "Map")

    targets = _BEAMNG_MAPS if choice == "all" else [choice]

    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGDrivingEnv

    for map_name in targets:
        print(f"\n>>> Generating trajectory for '{map_name}' ...")
        cache_path = os.path.join("outputs", "trajectories", f"{map_name}.json")
        if os.path.exists(cache_path):
            ans = input(f"    Cache already exists at {cache_path}. Overwrite? [y/N]: ").strip().lower()
            if ans != "y":
                print("    Skipped.")
                continue
            os.remove(cache_path)

        env = BeamNGDrivingEnv(
            beamng_home=BEAMNG_HOME,
            beamng_user=BEAMNG_USER,
            headless=HEADLESS,
            map_name=map_name,
        )
        try:
            env.reset()
            print(f"    Done. Source: {env.trajectory.source}, "
                  f"{len(env.trajectory.sparse_waypoints)} sparse / "
                  f"{len(env.trajectory.dense_waypoints)} dense waypoints.")
        finally:
            env.close()
```

- [ ] **Step 2: Wire it into `main_menu`**

Replace the existing menu in `main_menu`:

```python
        print("1. Train an agent")
        print("2. Evaluate an agent")
        print("3. Run a benchmark")
        print("4. Human play (BeamNG)")
        print("5. Quit")
```

with:

```python
        print("1. Train an agent")
        print("2. Evaluate an agent")
        print("3. Run a benchmark")
        print("4. Human play (BeamNG)")
        print("5. Generate trajectories (BeamNG)")
        print("6. Quit")
```

And update the dispatcher:

```python
        if choice == "1":
            _train_menu()
        elif choice == "2":
            _eval_menu()
        elif choice == "3":
            _benchmark_menu()
        elif choice == "4":
            _human_play_menu()
        elif choice == "5":
            _trajectory_menu()
        elif choice == "6":
            print("Bye!")
            break
        else:
            print("  Invalid choice.")
```

- [ ] **Step 3: Remove the now-misleading warning in `_pick_beamng_options`**

In `core/cli.py`, replace:

```python
    if map_name != "gridmap_v2":
        print("  Note: spawn position and checkpoints are defined for gridmap_v2 only.")
        print("        The map will load but waypoints may be misplaced.")
```

with:

```python
    cache_path = os.path.join("outputs", "trajectories", f"{map_name}.json")
    if not os.path.exists(cache_path):
        print(f"  Note: no cached trajectory for '{map_name}'. It will be generated on first launch.")
```

- [ ] **Step 4: Smoke-import the menu module**

Run: `python -c "from core.cli import main_menu; print('cli import OK')"`
Expected: `cli import OK`.

- [ ] **Step 5: Commit**

```bash
git add core/cli.py
git commit -m "feat(cli): add 'Generate trajectories' menu entry; drop hardcoded-map warning"
```

---

## Task 10: Update docs

**Files:**
- Modify: `scenario_creator.md`
- Modify: `README.md`

- [ ] **Step 1: Mark `scenario_creator.md` as the legacy / manual flow**

At the very top of `scenario_creator.md`, **insert** the following preamble before the existing first line:

```markdown
> **Status:** Legacy / fallback. Trajectories are now generated automatically for each map via the "Generate trajectories" menu entry — see `README.md`. The procedure below is only needed if you want to hand-tune a spawn point or override the auto-generated waypoints.

```

- [ ] **Step 2: Document the new flow in `README.md`**

In `README.md`, replace the existing section starting at `## Algorithmes disponibles` (around line 263) — add **just before** that section the following:

```markdown
## Trajectoires automatiques (BeamNG)

Les waypoints, spawn position et spawn rotation sont desormais generes
automatiquement pour chaque map a partir du reseau routier (DecalRoads)
de BeamNG.

- Pre-calcul depuis le menu principal : option `5. Generate trajectories (BeamNG)`
- Cache sur disque : `outputs/trajectories/<map>.json`
- Pour regenerer : supprimer le fichier JSON ou reactiver l'option 5
- Pour les maps sans routes (`smallgrid`), une boucle carree de 80 m sert de fallback

Le format du cache JSON :

```json
{
  "spawn_pos": [x, y, z],
  "spawn_rot": [qx, qy, qz, qw],
  "sparse_waypoints": [[x, y, z], ...],
  "dense_waypoints":  [[x, y, z], ...],
  "map_name": "...",
  "generated_at": "...",
  "source": "road_network:<id>" | "fallback:square_loop"
}
```

Si vous voulez surcharger la trajectoire pour une map particuliere, editez
le JSON a la main ou utilisez la procedure decrite dans `scenario_creator.md`.

---

```

- [ ] **Step 3: Commit**

```bash
git add scenario_creator.md README.md
git commit -m "docs: document auto-trajectory generation; mark scenario_creator as legacy"
```

---

## Task 11: Add `pytest` to CI

**Files:**
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Append a pytest step to the `test` job**

In `.github/workflows/ci.yml`, replace the dependency-install step in the `test` job:

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install python-dotenv tqdm gymnasium numpy matplotlib pygame
          pip install torch --index-url https://download.pytorch.org/whl/cpu
```

with:

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install python-dotenv tqdm gymnasium numpy matplotlib pygame pytest
          pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then **add** a new step at the end of the `test` job (after the two smoke-test blocks):

```yaml
      - name: Unit tests (trajectory)
        run: pytest tests/ -v
```

- [ ] **Step 2: Lint check locally**

Run: `ruff check .`
Expected: no errors. If ruff complains about the new files, run `ruff check --fix .` then `ruff format .`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run pytest in the test job"
```

---

## Task 12: Manual BeamNG smoke test (acceptance)

**Files:** none — manual verification only.

- [ ] **Step 1: Start BeamNG and run the menu**

Run: `python main.py`, choose `5. Generate trajectories (BeamNG)`, then choose `gridmap_v2`.
Expected: BeamNG launches, the script prints
`Done. Source: road_network:<id>, N sparse / M dense waypoints.`
and `outputs/trajectories/gridmap_v2.json` exists with the expected shape.

- [ ] **Step 2: Repeat for the other maps**

Re-run option 5 for `italy`, `west_coast_usa`, `smallgrid`.
Expected: three more JSON files appear in `outputs/trajectories/`. `smallgrid` should have `source = "fallback:square_loop"`.

- [ ] **Step 3: Short DQN training run on gridmap_v2**

From the main menu: `1. Train an agent` → `dqn` → `beamng` → map `gridmap_v2`. Train 5 episodes.
Expected: the vehicle spawns on the road (Z agrees with the cached `spawn_pos`), the green active-waypoint marker appears on the first sparse waypoint, training runs without `AttributeError` about removed constants.

- [ ] **Step 4: Short DDPG training run on italy**

From the main menu: `1. Train an agent` → `ddpg` → `beamng_continuous` → map `italy`. Train 3 episodes.
Expected: vehicle uses the dense waypoints (markers spaced ~8 m apart).

- [ ] **Step 5: Confirm the cache is loaded on second launch**

Re-run any training option with a previously-cached map.
Expected: no probe phase visible in the logs; the trajectory is loaded directly from JSON.

- [ ] **Step 6: Final commit (if any cleanup needed)**

If any small adjustments were needed during the smoke test, commit them. Otherwise this task ends with nothing to commit.

---

## Self-review checklist

| Spec section | Implemented in |
|---|---|
| Goal: replace hardcoded constants | Task 8 (steps 1, 6) |
| `TrajectoryData` dataclass | Task 2 |
| Resampling (sparse 25 m + dense 8 m) | Task 3, Task 7 |
| Spawn pose from heading | Task 4, Task 7 |
| Empty-map fallback | Task 5, Task 7 |
| Road extraction with version-tolerant edge keys | Task 6 |
| Cache JSON in `outputs/trajectories/{map}.json` | Task 7 |
| Two-phase BeamNG launch (probe → real) | Task 8 |
| Sparse vs dense selection by reward_mode/class | Task 8 (step 4) |
| New CLI menu entry | Task 9 |
| Doc updates (`scenario_creator.md`, `README.md`) | Task 10 |
| Tests reachable from CI | Task 11 |
| Manual acceptance | Task 12 |

**Placeholders:** none. **Type/name drift:** `TrajectoryData`, `resample`, `heading_to_quat`, `_extract_longest_road`, `_edge_center`, `_square_loop_fallback`, `generate`, `load_or_generate`, `CACHE_DIR`, `SPARSE_SPACING_M`, `DENSE_SPACING_M`, `SPAWN_Z_OFFSET_M`, `FALLBACK_SIDE_M`, `FALLBACK_GROUND_Z` — all defined in Task 2/3/4/5/6/7 and used consistently in Task 8/9.
