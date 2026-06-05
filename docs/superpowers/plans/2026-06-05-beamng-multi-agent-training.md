# BeamNG Multi-Agent Simultaneous Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spawn N vehicles in one BeamNG scenario (collisions disabled), each driven by its own RL algorithm/checkpoint, all training in parallel on the same trajectory with independent per-vehicle episodes.

**Architecture:** Extract the pure LiDAR geometry from `environments/beamng.py` into a shared, stateless `environments/beamng_geometry.py` (used by both the single- and multi-vehicle envs). Add `environments/beamng_multi.py` holding a per-vehicle `VehicleSlot` state container and a `BeamNGMultiEnv` that owns one shared scene and steps every vehicle with a single `bng.step()`. Add `core/multi_runner.py` whose loop gathers each agent's action, steps physics once, then updates every agent; a finished vehicle is teleported back to spawn and continues while others keep driving. Wire it into the CLI.

**Tech Stack:** Python, numpy, PyTorch (existing agents), beamngpy, pytest.

**Reference spec:** `docs/superpowers/specs/2026-06-05-beamng-multi-agent-training-design.md`

---

## File Structure

- **Create `environments/beamng_geometry.py`** — pure, stateless LiDAR geometry: `LidarConfig`, `ego_local_extents_from_bbox`, `world_to_local`, `lidar_keep_mask`, `process_lidar`. No `self`, no BeamNG.
- **Create `tests/test_beamng_geometry.py`** — unit tests for the helpers.
- **Modify `environments/beamng.py`** — delete NPC code; route `_cache_ego_local_bbox` / `_lidar_keep_mask` / `_process_lidar` through the helpers. Single-agent behaviour unchanged.
- **Create `tests/test_beamng.py`** — tests that the refactored env methods delegate correctly (bare instance, no sim).
- **Create `environments/beamng_multi.py`** — `VehicleSlot` dataclass + `BeamNGMultiEnv`.
- **Create `tests/test_beamng_multi.py`** — tests for slot state, action mapping, reward, observation (mock sensors).
- **Create `core/multi_runner.py`** — `MultiAgentRunner` parallel training loop.
- **Create `tests/test_multi_runner.py`** — loop tests with a fake env + fake agents (no BeamNG).
- **Modify `core/cli.py`** — add a testable `build_multi_session(...)` helper + a `_multi_train_menu()` entry.
- **Create `tests/test_cli_multi.py`** — tests for the spec→(env,agents) builder helper.

---

## Task 1: Extract pure LiDAR geometry helpers

**Files:**
- Create: `environments/beamng_geometry.py`
- Test: `tests/test_beamng_geometry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_beamng_geometry.py`:

```python
"""Unit tests for environments.beamng_geometry — pure LiDAR geometry."""

import numpy as np
import pytest

from environments.beamng_geometry import (
    LidarConfig,
    ego_local_extents_from_bbox,
    lidar_keep_mask,
    process_lidar,
    world_to_local,
)

CFG = LidarConfig(
    rays=8,
    v_bins=1,
    channels=1,
    fov_deg=120.0,
    vert_angle=6.0,
    max_dist=50.0,
    self_margin=0.30,
    ground_clearance=0.30,
)


class TestEgoLocalExtents:
    def test_returns_none_without_bbox(self):
        assert ego_local_extents_from_bbox({}, {"pos": (0, 0, 0)}, 0.3) is None

    def test_returns_none_without_pos(self):
        bbox = {"a": (1.0, 1.0, 1.0)}
        assert ego_local_extents_from_bbox(bbox, {}, 0.3) is None

    def test_axis_aligned_box_extents_include_margin(self):
        # Vehicle at origin, heading +X (dir=(1,0,0)). A unit box's local extents
        # should equal world extents (no rotation) expanded by the margin.
        bbox = {
            "c0": (-1.0, -0.5, 0.0),
            "c1": (1.0, 0.5, 1.5),
        }
        state = {"pos": (0.0, 0.0, 0.0), "dir": (1.0, 0.0, 0.0)}
        ext = ego_local_extents_from_bbox(bbox, state, margin=0.3)
        x_min, x_max, y_min, y_max, z_min, z_max = ext
        assert x_min == pytest.approx(-1.3)
        assert x_max == pytest.approx(1.3)
        assert y_min == pytest.approx(-0.8)
        assert y_max == pytest.approx(0.8)
        assert z_min == pytest.approx(-0.3)
        assert z_max == pytest.approx(1.8)


class TestWorldToLocal:
    def test_identity_heading_translates_only(self):
        pts = np.array([[5.0, 0.0, 1.0]], dtype=np.float32)
        lx, ly, lz = world_to_local(pts, (1.0, 0.0, 0.0), heading=0.0)
        assert lx[0] == pytest.approx(4.0)
        assert ly[0] == pytest.approx(0.0)
        assert lz[0] == pytest.approx(1.0)

    def test_heading_90deg_rotates_into_local(self):
        # Heading +90deg (facing +Y). A point straight ahead in world +Y maps to local +X.
        pts = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)
        lx, ly, lz = world_to_local(pts, (0.0, 0.0, 0.0), heading=np.pi / 2)
        assert lx[0] == pytest.approx(10.0, abs=1e-5)
        assert ly[0] == pytest.approx(0.0, abs=1e-5)


class TestLidarKeepMask:
    def test_rejects_points_inside_ego_box(self):
        lx = np.array([0.0, 5.0], dtype=np.float32)
        ly = np.array([0.0, 0.0], dtype=np.float32)
        lz = np.array([1.0, 1.0], dtype=np.float32)
        ext = (-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
        keep, dbg = lidar_keep_mask(lx, ly, lz, ext, self_margin=0.3, ground_clearance=0.3)
        assert keep.tolist() == [False, True]
        assert dbg["self"] == 1
        assert dbg["kept"] == 1

    def test_rejects_ground_points(self):
        lx = np.array([5.0, 5.0], dtype=np.float32)
        ly = np.array([0.0, 0.0], dtype=np.float32)
        lz = np.array([-0.5, 2.0], dtype=np.float32)  # first is below ground
        ext = (-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
        keep, dbg = lidar_keep_mask(lx, ly, lz, ext, self_margin=0.3, ground_clearance=0.3)
        assert keep.tolist() == [False, True]
        assert dbg["ground"] == 1

    def test_no_extents_uses_flat_ground_threshold(self):
        lx = np.array([5.0], dtype=np.float32)
        ly = np.array([0.0], dtype=np.float32)
        lz = np.array([1.0], dtype=np.float32)
        keep, dbg = lidar_keep_mask(lx, ly, lz, None, self_margin=0.3, ground_clearance=0.3)
        assert keep.tolist() == [True]
        assert dbg["extents_none"] is True


class TestProcessLidar:
    def test_empty_cloud_returns_all_clear(self):
        out, dbg = process_lidar(None, (0, 0, 0), 0.0, None, CFG)
        assert out.shape == (8,)
        assert np.all(out == 1.0)

    def test_single_obstacle_lands_in_one_bin_and_is_normalized(self):
        # One point 25 m straight ahead (local +X), above ground, no ego box.
        # Distance 25 / max 50 = 0.5 in the centre bin; others stay clear (1.0).
        cloud = np.array([[25.0, 0.0, 1.0]], dtype=np.float32)
        out, dbg = process_lidar(cloud, (0, 0, 0), 0.0, None, CFG)
        assert out.min() == pytest.approx(0.5, abs=1e-3)
        assert (out == 1.0).sum() == 7
        assert dbg["fov"] == 1

    def test_point_outside_fov_is_ignored(self):
        # Point directly behind (local -X) is outside the 120deg forward FOV.
        cloud = np.array([[-25.0, 0.0, 1.0]], dtype=np.float32)
        out, dbg = process_lidar(cloud, (0, 0, 0), 0.0, None, CFG)
        assert np.all(out == 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beamng_geometry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'environments.beamng_geometry'`.

- [ ] **Step 3: Write the implementation**

Create `environments/beamng_geometry.py`:

```python
"""Pure, stateless LiDAR geometry helpers shared by the BeamNG environments.

Extracted from environments.beamng so the single-vehicle env and the
multi-vehicle env use one implementation. No `self`, no BeamNG connection,
no logging side effects — callers handle those.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarConfig:
    """Binning + filtering parameters for a LiDAR sensor.

    Built once from an environment's class constants and passed to
    :func:`process_lidar`.
    """

    rays: int  # horizontal azimuth bins
    v_bins: int  # vertical elevation bins (1 = legacy single row)
    channels: int  # values stored per cell (currently 1: distance)
    fov_deg: float  # total forward azimuth field of view
    vert_angle: float  # total vertical field of view (used when v_bins > 1)
    max_dist: float  # metres — normalization range
    self_margin: float  # metres — ego OBB expansion for self-hit rejection
    ground_clearance: float  # metres above floor before a point counts as obstacle


def ego_local_extents_from_bbox(bbox, state, margin):
    """Return ego OBB extents in vehicle-local frame, or None.

    Tuple layout: (x_min, x_max, y_min, y_max, z_min, z_max), each already
    expanded by ``margin``. Returns None when bbox or pos is missing — callers
    fall back to a flat ground threshold.
    """
    if not bbox or "pos" not in state:
        return None

    corners = np.asarray(list(bbox.values()), dtype=np.float32)
    pos = np.asarray(state.get("pos", (0.0, 0.0, 0.0)), dtype=np.float32)
    dir_vec = np.asarray(state.get("dir", (1.0, 0.0, 0.0)), dtype=np.float32)
    heading = float(np.arctan2(dir_vec[1], dir_vec[0]))

    rel = corners - pos
    c, s = np.cos(-heading), np.sin(-heading)
    lx = rel[:, 0] * c - rel[:, 1] * s
    ly = rel[:, 0] * s + rel[:, 1] * c
    lz = rel[:, 2]
    return (
        float(lx.min() - margin),
        float(lx.max() + margin),
        float(ly.min() - margin),
        float(ly.max() + margin),
        float(lz.min() - margin),
        float(lz.max() + margin),
    )


def world_to_local(points, pos, heading):
    """Transform Nx3 world points into the vehicle-local frame.

    Returns (local_x, local_y, local_z) as separate 1-D arrays.
    """
    rel = points - np.asarray(pos, dtype=np.float32)
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    local_x = rel[:, 0] * cos_h - rel[:, 1] * sin_h
    local_y = rel[:, 0] * sin_h + rel[:, 1] * cos_h
    local_z = rel[:, 2]
    return local_x, local_y, local_z


def lidar_keep_mask(local_x, local_y, local_z, ego_extents, self_margin, ground_clearance):
    """Reject points inside the ego OBB or below the ground threshold.

    Returns (keep_mask, debug_dict). ``ground_clearance`` is measured above the
    true bbox floor (z_min + self_margin) when extents are known, else above 0.
    """
    n_total = int(local_x.size)
    inside_self = np.zeros(n_total, dtype=bool)

    if ego_extents is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = ego_extents
        inside_self = (
            (local_x >= x_min)
            & (local_x <= x_max)
            & (local_y >= y_min)
            & (local_y <= y_max)
            & (local_z >= z_min)
            & (local_z <= z_max)
        )
        floor = z_min + self_margin
        ground_z = floor + ground_clearance
    else:
        ground_z = ground_clearance

    below_ground = local_z <= ground_z
    keep = ~inside_self & ~below_ground

    debug = {
        "total": n_total,
        "self": int(inside_self.sum()),
        "ground": int((below_ground & ~inside_self).sum()),
        "kept": int(keep.sum()),
        "extents_none": ego_extents is None,
        "ground_z": float(ground_z),
    }
    return keep, debug


def process_lidar(point_cloud, vehicle_pos, vehicle_heading, ego_extents, cfg):
    """Bin a raw LiDAR point cloud into a (v_bins x rays x channels) grid.

    Returns (distances, debug). ``distances`` is a flat float32 array in [0, 1]
    where 0 means an obstacle is right there and 1 means clear. ``debug`` holds
    filtering counts plus the nearest in-FOV point's distance/height.
    """
    v_bins = cfg.v_bins
    h_bins = cfg.rays
    ch = cfg.channels
    n_out = v_bins * h_bins * ch
    distances = np.ones(n_out, dtype=np.float32)
    debug = {}

    if point_cloud is None or len(point_cloud) == 0:
        return distances, debug

    pts = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
    local_x, local_y, local_z = world_to_local(pts, vehicle_pos, vehicle_heading)

    keep, debug = lidar_keep_mask(
        local_x, local_y, local_z, ego_extents, cfg.self_margin, cfg.ground_clearance
    )
    local_x = local_x[keep]
    local_y = local_y[keep]
    local_z = local_z[keep]
    if local_x.size == 0:
        return distances, debug

    angles = np.arctan2(local_y, local_x)
    dists = np.hypot(local_x, local_y)

    half_fov = np.radians(cfg.fov_deg / 2.0)
    in_fov = np.abs(angles) <= half_fov
    angles = angles[in_fov]
    dists = dists[in_fov]
    local_z = local_z[in_fov]
    if angles.size == 0:
        return distances, debug

    nearest = int(np.argmin(dists))
    debug["fov"] = int(angles.size)
    debug["min_dist_m"] = float(dists[nearest])
    debug["min_dist_z"] = float(local_z[nearest])

    h_edges = np.linspace(-half_fov, half_fov, h_bins + 1)
    h_idx = np.clip(np.digitize(angles, h_edges) - 1, 0, h_bins - 1)

    if v_bins == 1:
        v_idx = np.zeros(angles.shape, dtype=np.intp)
    else:
        half_vfov = np.radians(cfg.vert_angle / 2.0)
        elevation = np.arctan2(local_z, dists)
        v_edges = np.linspace(-half_vfov, half_vfov, v_bins + 1)
        v_idx = np.clip(np.digitize(elevation, v_edges) - 1, 0, v_bins - 1)

    for v in range(v_bins):
        for h in range(h_bins):
            sel = dists[(v_idx == v) & (h_idx == h)]
            if sel.size:
                distances[(v * h_bins + h) * ch] = np.clip(sel.min() / cfg.max_dist, 0.0, 1.0)

    return distances, debug
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng_geometry.py -v`
Expected: PASS (all tests green).

- [ ] **Step 5: Commit**

```bash
git add environments/beamng_geometry.py tests/test_beamng_geometry.py
git commit -m "feat: extract pure LiDAR geometry helpers"
```

---

## Task 2: Refactor beamng.py onto the helpers and delete NPC code

**Files:**
- Modify: `environments/beamng.py`
- Test: `tests/test_beamng.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_beamng.py`. These instantiate the env **without launching** (its `__init__` does not connect) and exercise the refactored LiDAR methods. `LOG_LIDAR` defaults to False, so `self.bng` is never touched.

```python
"""Tests for environments.beamng — refactored LiDAR delegation (no sim)."""

import numpy as np
import pytest

from environments.beamng import BeamNGDrivingEnv


def _bare_env():
    # __init__ only stores config; no BeamNG connection is opened.
    return BeamNGDrivingEnv(beamng_home="unused")


class TestProcessLidarDelegation:
    def test_empty_cloud_all_clear(self):
        env = _bare_env()
        env._ego_local_extents = None
        out = env._process_lidar(None, (0.0, 0.0, 0.0), 0.0)
        assert out.shape == (BeamNGDrivingEnv.LIDAR_RAYS,)
        assert np.all(out == 1.0)

    def test_single_obstacle_normalized_into_one_bin(self):
        env = _bare_env()
        env._ego_local_extents = None
        cloud = np.array([[25.0, 0.0, 1.0]], dtype=np.float32)
        out = env._process_lidar(cloud, (0.0, 0.0, 0.0), 0.0)
        assert out.min() == pytest.approx(0.5, abs=1e-3)
        assert (out == 1.0).sum() == BeamNGDrivingEnv.LIDAR_RAYS - 1
        # debug populated as a side effect, as before
        assert env._lidar_debug["fov"] == 1


class TestNoNpcApi:
    def test_npc_helpers_removed(self):
        assert not hasattr(BeamNGDrivingEnv, "_spawn_npc_vehicles")
        assert not hasattr(BeamNGDrivingEnv, "NPC_COUNT")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beamng.py -v`
Expected: FAIL — `test_npc_helpers_removed` fails (NPC code still present) and/or `_process_lidar` still works the old way but `_spawn_npc_vehicles` exists.

- [ ] **Step 3a: Delete the NPC code**

In `environments/beamng.py`, delete the NPC configuration block and method (currently lines ~372-423):

```python
    # NPC configuration
    NPC_COUNT = 5          # number of parked NPC vehicles to spawn
    NPC_LATERAL_OFFSET = 4.0   # metres to the right of the road direction
    NPC_COLORS = ["Blue", "Green", "Orange", "White", "Black"]
    NPC_MODELS = ["etk800", "etk800", "etk800", "etk800", "etk800"]

    def _spawn_npc_vehicles(self):
        ...  # entire method
```

Remove the whole block (the four `NPC_*` constants and the entire `_spawn_npc_vehicles` method).

In `_load_scenario`, delete the call and its comment (currently lines ~451-453):

```python
        # Spawn NPC vehicles parked to the side of the trajectory.
        # They are static (no AI) and laterally offset so they never block the ego's path.
        self._spawn_npc_vehicles()
```

The unused `math` and `random` imports may remain if still used elsewhere — `random` is used by `_randomize_waypoints`; `math` is now unused after removing NPC code, so delete `import math` at the top.

- [ ] **Step 3b: Route LiDAR math through the helpers**

Add the import near the existing imports in `environments/beamng.py`:

```python
from environments.beamng_geometry import (
    LidarConfig,
    ego_local_extents_from_bbox,
    process_lidar,
)
```

Add a helper that builds a `LidarConfig` from the class constants. Place it as a method on `BeamNGDrivingEnv` (so subclasses with different `LIDAR_*` constants get the right config):

```python
    def _lidar_config(self) -> LidarConfig:
        return LidarConfig(
            rays=self.LIDAR_RAYS,
            v_bins=self.LIDAR_V_BINS,
            channels=self.LIDAR_CHANNELS_PER_RAY,
            fov_deg=self.LIDAR_FOV_DEG,
            vert_angle=self.LIDAR_VERT_ANGLE,
            max_dist=self.LIDAR_MAX_DIST,
            self_margin=self.LIDAR_SELF_MARGIN,
            ground_clearance=self.LIDAR_GROUND_CLEARANCE,
        )
```

Replace the body of `_cache_ego_local_bbox` from `corners = ...` onward (keep the poll + bbox fetch + the early `if not bbox or "pos" not in state` guard) so the extent math delegates:

```python
        self._ego_local_extents = ego_local_extents_from_bbox(
            bbox, state, self.LIDAR_SELF_MARGIN
        )
```

That single call replaces the manual `corners`/`rel`/`lx/ly/lz`/extents computation. Keep the surrounding `try/except`, the `poll_sensors()`, the `get_bbox()`, and the `state = self.vehicle.state or {}` lines.

Delete the now-unused `_lidar_keep_mask` method entirely (its logic now lives in the helper, called inside `process_lidar`).

Replace the entire body of `_process_lidar` with a delegation that preserves the existing side effects (set `self._lidar_debug`, optional Lua logging):

```python
    def _process_lidar(self, point_cloud, vehicle_pos, vehicle_heading) -> np.ndarray:
        """Bin a raw LiDAR point cloud via the shared geometry helper.

        Preserves the previous side effects: stores the filtering breakdown in
        self._lidar_debug and optionally logs the bins to the BeamNG console.
        """
        distances, debug = process_lidar(
            point_cloud,
            vehicle_pos,
            vehicle_heading,
            self._ego_local_extents,
            self._lidar_config(),
        )
        self._lidar_debug = debug

        if LOG_LIDAR:
            if point_cloud is None or len(point_cloud) == 0:
                self.bng.queue_lua_command("log('I', 'RL', 'Lidar: no points')")
            else:
                self.bng.queue_lua_command(
                    "log('I', 'RL', 'Lidar: [{}]')".format(
                        ", ".join(f"{v:.3f}" for v in distances)
                    )
                )
        return distances
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng.py tests/test_beamng_geometry.py -v`
Expected: PASS.

Run the full existing suite to confirm no regressions:
Run: `python -m pytest -q`
Expected: PASS (existing trajectory/dqn tests still green).

- [ ] **Step 5: Lint**

Run: `ruff check environments/beamng.py environments/beamng_geometry.py`
Expected: no errors (fix any unused-import warnings, e.g. removed `import math`).

- [ ] **Step 6: Commit**

```bash
git add environments/beamng.py tests/test_beamng.py
git commit -m "refactor: route beamng LiDAR through shared helpers; remove NPC test code"
```

---

## Task 3: VehicleSlot state container

**Files:**
- Create: `environments/beamng_multi.py` (first piece — `VehicleSlot`)
- Test: `tests/test_beamng_multi.py` (first tests)

- [ ] **Step 1: Write the failing test**

Create `tests/test_beamng_multi.py`:

```python
"""Tests for environments.beamng_multi."""

from environments.beamng_multi import VehicleSlot


def _slot(**kw):
    defaults = dict(
        name="ego_0",
        color="Red",
        vehicle_id="taxi",
        agent=object(),
        reward_mode="default",
        action_space="discrete",
        save_path="outputs/dqn.pth",
    )
    defaults.update(kw)
    return VehicleSlot(**defaults)


class TestVehicleSlot:
    def test_episode_state_defaults_to_zero(self):
        s = _slot()
        assert s.waypoint_idx == 0
        assert s.steps == 0
        assert s.last_damage == 0.0
        assert s.checkpoint_hit is False
        assert s.done is False
        assert s.episode == 0
        assert s.reward_history == []

    def test_reset_episode_zeros_running_state_but_keeps_episode_count(self):
        s = _slot()
        s.waypoint_idx = 4
        s.steps = 123
        s.last_damage = 50.0
        s.ep_reward = 99.0
        s.checkpoint_hit = True
        s.done = True
        s.episode = 7
        s.reward_history.append(99.0)

        s.reset_episode()

        assert s.waypoint_idx == 0
        assert s.steps == 0
        assert s.last_damage == 0.0
        assert s.ep_reward == 0.0
        assert s.checkpoint_hit is False
        assert s.done is False
        # history + episode counter survive a reset
        assert s.episode == 7
        assert s.reward_history == [99.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_beamng_multi.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'environments.beamng_multi'`.

- [ ] **Step 3: Write the implementation**

Create `environments/beamng_multi.py` with the slot (the env class is added in Task 4):

```python
"""Multi-vehicle BeamNG environment for simultaneous parallel training.

One scenario holds N vehicles (collisions disabled). A single physics step
advances every vehicle; each vehicle keeps its own episode state in a
VehicleSlot so several algorithms train in parallel on one trajectory.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VehicleSlot:
    """All per-vehicle state: identity, sensors, episode state, training stats.

    Nothing here is shared between vehicles — the multi env reads and writes
    these fields per slot so two algorithms never alias each other's state.
    """

    # Identity / config
    name: str
    color: str
    vehicle_id: str
    agent: Any
    reward_mode: str  # "default" (DQN) or "ddpg" (DDPG/TD3)
    action_space: str  # "discrete" or "continuous"
    save_path: str

    # Sensors (assigned during scenario load)
    vehicle: Any = None
    electrics: Any = None
    damage_sensor: Any = None
    lidar: Any = None

    # Episode state
    waypoint_idx: int = 0
    last_damage: float = 0.0
    last_dist: float = 0.0
    current_dist: float = 0.0
    current_pos: tuple = (0.0, 0.0, 0.0)
    checkpoint_dist: float = 0.0
    checkpoint_hit: bool = False
    steps: int = 0
    ego_local_extents: tuple | None = None
    last_obs: np.ndarray | None = None
    done: bool = False

    # Per-episode running accumulators
    ep_reward: float = 0.0
    ep_losses: list = field(default_factory=list)

    # Cross-episode training stats
    episode: int = 0
    reward_history: list = field(default_factory=list)
    steps_history: list = field(default_factory=list)

    def reset_episode(self) -> None:
        """Zero running episode state. Keeps episode counter + histories."""
        self.waypoint_idx = 0
        self.last_damage = 0.0
        self.last_dist = 0.0
        self.current_dist = 0.0
        self.checkpoint_dist = 0.0
        self.checkpoint_hit = False
        self.steps = 0
        self.ep_reward = 0.0
        self.ep_losses = []
        self.done = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_beamng_multi.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "feat: add VehicleSlot per-vehicle state container"
```

---

## Task 4: BeamNGMultiEnv — construction, action mapping, reward, observation

This task builds the env in testable pieces. The BeamNG-coupled lifecycle methods (`launch`, `_load_scenario`, `reset_all`, `reset_vehicle`, `close`) are written here too but verified with mocks; the pure-logic methods get real assertions.

**Files:**
- Modify: `environments/beamng_multi.py`
- Test: `tests/test_beamng_multi.py`

### Subtask 4a: `__init__` + slot building from specs

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beamng_multi.py`:

```python
from environments.beamng_multi import BeamNGMultiEnv, build_slots


class _FakeAgent:
    pass


SPECS = [
    {"algo": "dqn", "agent": _FakeAgent(), "vehicle_id": "taxi", "color": "Yellow",
     "save_path": "outputs/dqn.pth"},
    {"algo": "ddpg", "agent": _FakeAgent(), "vehicle_id": "ibishu_pigeon", "color": "Red",
     "save_path": "outputs/ddpg.pth"},
    {"algo": "td3", "agent": _FakeAgent(), "vehicle_id": "taxi", "color": "Blue",
     "save_path": "outputs/td3.pth"},
]


class TestBuildSlots:
    def test_names_are_unique_and_indexed(self):
        slots = build_slots(SPECS)
        assert [s.name for s in slots] == ["ego_0", "ego_1", "ego_2"]

    def test_reward_mode_and_action_space_derived_from_algo(self):
        slots = build_slots(SPECS)
        assert slots[0].reward_mode == "default"
        assert slots[0].action_space == "discrete"
        assert slots[1].reward_mode == "ddpg"
        assert slots[1].action_space == "continuous"
        assert slots[2].reward_mode == "ddpg"
        assert slots[2].action_space == "continuous"

    def test_carries_color_and_save_path(self):
        slots = build_slots(SPECS)
        assert slots[1].color == "Red"
        assert slots[1].save_path == "outputs/ddpg.pth"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_beamng_multi.py::TestBuildSlots -v`
Expected: FAIL — `ImportError: cannot import name 'BeamNGMultiEnv'` / `build_slots`.

- [ ] **Step 3: Write the implementation**

Append to `environments/beamng_multi.py` (after `VehicleSlot`). First the algo→behaviour mapping and `build_slots`:

```python
# Algorithms whose action space is continuous (actor outputs in [-1, 1]).
_CONTINUOUS_ALGOS = {"ddpg", "td3"}


def build_slots(specs: list[dict]) -> list[VehicleSlot]:
    """Turn a list of vehicle specs into VehicleSlots.

    Each spec dict: {"algo", "agent", "vehicle_id", "color", "save_path"}.
    reward_mode and action_space are derived from the algorithm name.
    """
    slots = []
    for i, spec in enumerate(specs):
        algo = spec["algo"]
        continuous = algo in _CONTINUOUS_ALGOS
        slots.append(
            VehicleSlot(
                name=f"ego_{i}",
                color=spec["color"],
                vehicle_id=spec["vehicle_id"],
                agent=spec["agent"],
                reward_mode="ddpg" if continuous else "default",
                action_space="continuous" if continuous else "discrete",
                save_path=spec["save_path"],
            )
        )
    return slots
```

Then the env class. Import the base env's tunables and the geometry helpers at the top of the file (add to existing imports):

```python
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics, Lidar

from config import LIDAR_VISUALISE, LOG_LIDAR
from core.trajectory import TrajectoryData, load_or_generate
from environments.beamng import BeamNGDrivingEnv
from environments.beamng_geometry import (
    LidarConfig,
    ego_local_extents_from_bbox,
    process_lidar,
)
```

```python
class BeamNGMultiEnv:
    """Owns one BeamNG scenario shared by N vehicles, each with its own slot.

    Reuses the single-vehicle env's constants (ACTIONS table, LiDAR config,
    waypoint/reward thresholds) via BeamNGDrivingEnv class attributes, but keeps
    every mutable bit of episode state in per-vehicle VehicleSlots.
    """

    # Reuse the discrete action table and tunables from the single-vehicle env.
    ACTIONS = BeamNGDrivingEnv.ACTIONS
    WAYPOINT_RADIUS = BeamNGDrivingEnv.WAYPOINT_RADIUS
    MAX_STEPS = BeamNGDrivingEnv.MAX_STEPS
    MAX_DAMAGE = BeamNGDrivingEnv.MAX_DAMAGE
    CHECKPOINT_WARN_DIST = BeamNGDrivingEnv.CHECKPOINT_WARN_DIST
    CHECKPOINT_RESET_DIST = BeamNGDrivingEnv.CHECKPOINT_RESET_DIST

    # LiDAR geometry constants (single forward row, same as base env).
    LIDAR_RAYS = BeamNGDrivingEnv.LIDAR_RAYS
    LIDAR_V_BINS = BeamNGDrivingEnv.LIDAR_V_BINS
    LIDAR_CHANNELS_PER_RAY = BeamNGDrivingEnv.LIDAR_CHANNELS_PER_RAY
    LIDAR_FOV_DEG = BeamNGDrivingEnv.LIDAR_FOV_DEG
    LIDAR_VERT_ANGLE = BeamNGDrivingEnv.LIDAR_VERT_ANGLE
    LIDAR_MAX_DIST = BeamNGDrivingEnv.LIDAR_MAX_DIST
    LIDAR_GROUND_CLEARANCE = BeamNGDrivingEnv.LIDAR_GROUND_CLEARANCE
    LIDAR_SELF_MARGIN = BeamNGDrivingEnv.LIDAR_SELF_MARGIN
    LIDAR_MOUNT_POS = BeamNGDrivingEnv.LIDAR_MOUNT_POS
    LIDAR_MOUNT_DIR = BeamNGDrivingEnv.LIDAR_MOUNT_DIR
    LIDAR_MOUNT_UP = BeamNGDrivingEnv.LIDAR_MOUNT_UP
    LIDAR_VERT_RES = BeamNGDrivingEnv.LIDAR_VERT_RES

    VEHICLES = BeamNGDrivingEnv.VEHICLES

    def __init__(
        self,
        slots: list[VehicleSlot],
        beamng_home: str,
        beamng_user: str = None,
        host: str = "localhost",
        port: int = 25252,
        headless: bool = False,
        map_name: str = "gridmap_v2",
        trajectory_hints: int = 0,
    ):
        self.slots = slots
        self.beamng_home = beamng_home
        self.beamng_user = beamng_user
        self.host = host
        self.port = port
        self.headless = headless
        self.map_name = map_name
        self.trajectory_hints = trajectory_hints

        self.bng: BeamNGpy = None
        self.scenario: Scenario = None
        self.trajectory: TrajectoryData | None = None
        self.waypoints: list[tuple[float, float, float]] = []

    def _lidar_config(self) -> LidarConfig:
        return LidarConfig(
            rays=self.LIDAR_RAYS,
            v_bins=self.LIDAR_V_BINS,
            channels=self.LIDAR_CHANNELS_PER_RAY,
            fov_deg=self.LIDAR_FOV_DEG,
            vert_angle=self.LIDAR_VERT_ANGLE,
            max_dist=self.LIDAR_MAX_DIST,
            self_margin=self.LIDAR_SELF_MARGIN,
            ground_clearance=self.LIDAR_GROUND_CLEARANCE,
        )

    @property
    def n_states(self) -> int:
        base = 6 + self.LIDAR_RAYS * self.LIDAR_V_BINS * self.LIDAR_CHANNELS_PER_RAY
        return base + self.trajectory_hints * 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_beamng_multi.py::TestBuildSlots -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "feat: BeamNGMultiEnv construction + slot building from specs"
```

### Subtask 4b: `apply_action` discrete/continuous mapping

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beamng_multi.py`:

```python
from unittest.mock import MagicMock

import numpy as np


def _env(slots=None):
    return BeamNGMultiEnv(slots=slots or build_slots(SPECS), beamng_home="unused")


class TestApplyAction:
    def test_discrete_action_maps_through_actions_table(self):
        env = _env()
        slot = env.slots[0]  # discrete
        slot.vehicle = MagicMock()
        env.apply_action(slot, 1)  # ACTIONS[1] = full throttle straight
        slot.vehicle.control.assert_called_once_with(throttle=1.0, steering=0.0, brake=0.0)

    def test_continuous_action_positive_accel_is_throttle(self):
        env = _env()
        slot = env.slots[1]  # continuous
        slot.vehicle = MagicMock()
        env.apply_action(slot, np.array([0.8, -0.5], dtype=np.float32))
        slot.vehicle.control.assert_called_once_with(throttle=0.8, steering=-0.5, brake=0.0)

    def test_continuous_action_negative_accel_is_brake(self):
        env = _env()
        slot = env.slots[1]
        slot.vehicle = MagicMock()
        env.apply_action(slot, np.array([-0.6, 0.2], dtype=np.float32))
        slot.vehicle.control.assert_called_once_with(throttle=0.0, steering=0.2, brake=0.6)

    def test_continuous_three_dim_action_is_throttle_steer_brake(self):
        env = _env()
        slot = env.slots[2]
        slot.vehicle = MagicMock()
        env.apply_action(slot, np.array([0.5, 0.1, 0.3], dtype=np.float32))
        slot.vehicle.control.assert_called_once_with(throttle=0.5, steering=0.1, brake=0.3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_beamng_multi.py::TestApplyAction -v`
Expected: FAIL — `AttributeError: 'BeamNGMultiEnv' object has no attribute 'apply_action'`.

- [ ] **Step 3: Write the implementation**

Add to `BeamNGMultiEnv`:

```python
    def apply_action(self, slot: VehicleSlot, action) -> None:
        """Map an agent action to vehicle controls. Does not step physics."""
        if slot.action_space == "discrete" or isinstance(action, (int, np.integer)):
            ctrl = self.ACTIONS[int(action)]
            throttle, steering, brake = ctrl["throttle"], ctrl["steering"], ctrl["brake"]
        else:
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            if action.shape[0] == 2:
                accel = float(action[0])
                steering = float(action[1])
                throttle = max(0.0, accel)
                brake = max(0.0, -accel)
            else:
                throttle = float(max(0.0, action[0]))
                steering = float(action[1])
                brake = float(max(0.0, action[2]))
        slot.vehicle.control(throttle=throttle, steering=steering, brake=brake)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_beamng_multi.py::TestApplyAction -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "feat: BeamNGMultiEnv.apply_action discrete/continuous mapping"
```

### Subtask 4c: per-slot path errors + reward

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beamng_multi.py`:

```python
class TestPathErrorsAndReward:
    def test_path_errors_advance_waypoint_when_close(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]
        slot.waypoint_idx = 0
        # Vehicle sits right on waypoint 0 -> should advance to 1 and flag a hit.
        state = {"vel": (1.0, 0.0, 0.0)}
        env._path_errors(slot, pos=(0.0, 0.0, 0.0), state=state)
        assert slot.waypoint_idx == 1
        assert slot.checkpoint_hit is True

    def test_default_reward_gives_checkpoint_bonus(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]  # reward_mode "default"
        slot.checkpoint_hit = True
        slot.waypoint_idx = 1
        slot.checkpoint_dist = 0.0
        obs = np.zeros(env.n_states, dtype=np.float32)
        obs[0] = 0.5  # moving (speed) so no stationary penalty
        reward, done = env.compute_reward(slot, obs)
        assert reward >= 100.0
        assert slot.checkpoint_hit is False  # consumed

    def test_default_reward_terminates_on_max_damage(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0)]
        slot = env.slots[0]
        obs = np.zeros(env.n_states, dtype=np.float32)
        obs[0] = 0.5
        obs[4] = 1.0  # damage_norm = 1.0 -> 1000 damage == MAX_DAMAGE
        reward, done = env.compute_reward(slot, obs)
        assert done is True

    def test_ddpg_reward_rewards_progress(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[1]  # reward_mode "ddpg"
        slot.last_dist = 50.0
        slot.current_dist = 40.0  # got 10 m closer
        obs = np.zeros(env.n_states, dtype=np.float32)
        obs[0] = 0.4  # speed
        reward, done = env.compute_reward(slot, obs)
        assert reward > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_beamng_multi.py::TestPathErrorsAndReward -v`
Expected: FAIL — `_path_errors` / `compute_reward` not defined.

- [ ] **Step 3: Write the implementation**

Add to `BeamNGMultiEnv`. These mirror `BeamNGDrivingEnv` but read/write `slot.*`. `_update_active_marker` is intentionally a no-op hook here (per-vehicle markers are Subtask 4f / optional).

```python
    def _path_errors(self, slot, pos, state):
        """Heading/lateral error to slot's current waypoint; advances on arrival.

        Sets slot.current_dist for the DDPG progress reward.
        """
        if not self.waypoints or not state:
            slot.current_dist = 0.0
            return 0.0, 0.0, 0.0

        target = self.waypoints[slot.waypoint_idx % len(self.waypoints)]
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = float(np.hypot(dx, dy))
        slot.current_dist = dist

        if dist < self.WAYPOINT_RADIUS:
            slot.waypoint_idx += 1
            slot.checkpoint_hit = True
            if slot.waypoint_idx < len(self.waypoints):
                new_t = self.waypoints[slot.waypoint_idx]
                slot.current_dist = float(np.hypot(new_t[0] - pos[0], new_t[1] - pos[1]))

        vel = state.get("vel", (1.0, 0.0, 0.0))
        vehicle_heading = np.arctan2(vel[1], vel[0])
        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
        lateral_err = dist * np.sin(heading_err)
        return float(heading_err), float(lateral_err), dist

    def compute_reward(self, slot, obs):
        if slot.reward_mode == "ddpg":
            return self._reward_ddpg(slot, obs)
        return self._reward_default(slot, obs)

    def _reward_default(self, slot, obs):
        speed, steering, _heading_err, _lateral_err, damage_norm = obs[:5]
        damage = damage_norm * 1000.0
        done = False
        reward = 0.0

        if speed < 0.05:
            reward -= 2.0
        reward -= abs(steering) * 0.2

        if damage > slot.last_damage + 50:
            reward -= 50.0
        if damage >= self.MAX_DAMAGE:
            done = True
        slot.last_damage = damage

        if slot.steps >= self.MAX_STEPS:
            done = True

        if slot.checkpoint_hit:
            reward += 100.0 * slot.waypoint_idx
            slot.checkpoint_hit = False

        if slot.waypoint_idx >= len(self.waypoints):
            reward += 200.0
            done = True

        dist = slot.checkpoint_dist
        if dist >= self.CHECKPOINT_RESET_DIST:
            reward -= 100.0
            done = True
        elif dist >= self.CHECKPOINT_WARN_DIST:
            reward -= (
                (dist - self.CHECKPOINT_WARN_DIST)
                / (self.CHECKPOINT_RESET_DIST - self.CHECKPOINT_WARN_DIST)
                * 10.0
            )

        return float(reward), done

    def _reward_ddpg(self, slot, obs):
        speed, _steering, heading_err, _lateral_err, damage_norm = obs[:5]
        lidar_bins = obs[5:]
        damage = damage_norm * 1000.0
        alignment = np.cos(heading_err * np.pi)
        done = False
        reward = 0.0

        dist_delta = slot.last_dist - slot.current_dist
        reward += dist_delta * 3.0
        slot.last_dist = slot.current_dist

        reward += speed * alignment * 3.0
        reward += alignment * 0.5

        if speed < 0.05:
            reward -= 1.0

        min_lidar = float(np.min(lidar_bins)) if lidar_bins.size else 1.0
        if min_lidar < 0.2:
            reward -= (1.0 - min_lidar) * 5.0
        elif min_lidar < 0.4:
            reward -= (1.0 - min_lidar) * 2.0

        damage_delta = damage - slot.last_damage
        if damage_delta > 0:
            reward -= damage_delta * 0.3
        if damage_delta > 150:
            reward -= 30.0
            done = True
        if damage >= self.MAX_DAMAGE:
            done = True
        slot.last_damage = damage

        if slot.steps >= self.MAX_STEPS:
            done = True

        if slot.checkpoint_hit:
            reward += 50.0
            slot.checkpoint_hit = False

        if slot.waypoint_idx >= len(self.waypoints):
            reward += 200.0
            slot.waypoint_idx = 0
            done = True

        return float(reward), done
```

Note: the DDPG reward indexes `obs[5:]` as LiDAR bins, matching the base env's `_compute_reward_ddpg`. Since `n_states` here uses the `6 + ...` kinematic layout, `obs[5]` is the distance-to-checkpoint feature followed by LiDAR — identical to the base env, so reward parity is preserved.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_beamng_multi.py::TestPathErrorsAndReward -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "feat: BeamNGMultiEnv per-slot path errors + reward"
```

### Subtask 4d: `observe` with mocked sensors

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beamng_multi.py`:

```python
class TestObserve:
    def _wire_slot_sensors(self, slot, *, speed, steering, damage, pos, vel, lidar_points):
        slot.vehicle = MagicMock()
        slot.vehicle.state = {"pos": pos, "vel": vel, "dir": vel}
        slot.electrics = MagicMock()
        slot.electrics.data = {"wheelspeed": speed, "steering": steering}
        slot.damage_sensor = MagicMock()
        slot.damage_sensor.data = {"damage": damage}
        slot.lidar = MagicMock()
        slot.lidar.poll.return_value = {"pointCloud": lidar_points}
        slot.ego_local_extents = None

    def test_observe_returns_vector_of_n_states(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]
        self._wire_slot_sensors(
            slot, speed=10.0, steering=0.0, damage=0.0,
            pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), lidar_points=None,
        )
        obs = env.observe(slot)
        assert obs.shape == (env.n_states,)
        # speed normalized by 50
        assert obs[0] == pytest.approx(0.2, abs=1e-3)

    def test_observe_polls_each_slot_sensor(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]
        self._wire_slot_sensors(
            slot, speed=0.0, steering=0.0, damage=0.0,
            pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), lidar_points=None,
        )
        env.observe(slot)
        slot.vehicle.poll_sensors.assert_called_once()
        slot.lidar.poll.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_beamng_multi.py::TestObserve -v`
Expected: FAIL — `observe` not defined.

- [ ] **Step 3: Write the implementation**

Add to `BeamNGMultiEnv`:

```python
    def observe(self, slot: VehicleSlot) -> np.ndarray:
        """Poll a slot's sensors and return its normalized observation vector."""
        slot.vehicle.poll_sensors()

        elec = slot.electrics.data or {}
        dmg = slot.damage_sensor.data or {}
        speed = float(elec.get("wheelspeed", 0.0))
        steering = float(elec.get("steering", 0.0))
        damage = float(dmg.get("damage", 0.0))

        state = slot.vehicle.state or {}
        pos = state.get("pos", (0.0, 0.0, 0.0))
        vel = state.get("vel", (1.0, 0.0, 0.0))
        dir_vec = state.get("dir", vel)
        vehicle_heading = float(np.arctan2(dir_vec[1], dir_vec[0]))

        heading_err, lateral_err, dist = self._path_errors(slot, pos, state)

        point_cloud = slot.lidar.poll().get("pointCloud", None) if slot.lidar is not None else None
        lidar_bins, debug = process_lidar(
            point_cloud, pos, vehicle_heading, slot.ego_local_extents, self._lidar_config()
        )

        slot.current_pos = pos
        if self.waypoints:
            target = self.waypoints[slot.waypoint_idx % len(self.waypoints)]
            slot.checkpoint_dist = float(np.hypot(pos[0] - target[0], pos[1] - target[1]))

        waypoint_hints = self._waypoint_hints(slot, pos, vehicle_heading)

        return np.concatenate(
            [
                np.array(
                    [
                        np.clip(speed / 50.0, -1.0, 1.0),
                        np.clip(steering, -1.0, 1.0),
                        np.clip(heading_err / np.pi, -1.0, 1.0),
                        np.clip(lateral_err / 5.0, -1.0, 1.0),
                        np.clip(damage / 1000.0, 0.0, 1.0),
                        np.clip(dist / self.CHECKPOINT_WARN_DIST, 0.0, 2.0),
                    ],
                    dtype=np.float32,
                ),
                lidar_bins,
                waypoint_hints,
            ]
        )

    def _waypoint_hints(self, slot, pos, vehicle_heading) -> np.ndarray:
        """Vehicle-local (forward, left) coords for the next trajectory_hints waypoints."""
        if not self.trajectory_hints or not self.waypoints:
            return np.empty(0, dtype=np.float32)
        NORM = 100.0
        cos_h = np.cos(-vehicle_heading)
        sin_h = np.sin(-vehicle_heading)
        hints: list[float] = []
        for i in range(self.trajectory_hints):
            idx = (slot.waypoint_idx + i) % len(self.waypoints)
            wp = self.waypoints[idx]
            rel_x = wp[0] - pos[0]
            rel_y = wp[1] - pos[1]
            local_x = rel_x * cos_h - rel_y * sin_h
            local_y = rel_x * sin_h + rel_y * cos_h
            hints.append(float(np.clip(local_x / NORM, -1.0, 1.0)))
            hints.append(float(np.clip(local_y / NORM, -1.0, 1.0)))
        return np.array(hints, dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_beamng_multi.py::TestObserve -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "feat: BeamNGMultiEnv.observe with per-slot sensors"
```

### Subtask 4e: scene lifecycle (mock-verified)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_beamng_multi.py`:

```python
class TestLifecycle:
    def test_step_physics_steps_once_for_all(self):
        env = _env()
        env.bng = MagicMock()
        env.step_physics()
        env.bng.step.assert_called_once_with(10)

    def test_reset_vehicle_teleports_and_resets_state(self):
        env = _env()
        env.trajectory = MagicMock()
        env.trajectory.spawn_pos = (1.0, 2.0, 3.0)
        env.trajectory.spawn_rot = (0.0, 0.0, 0.0, 1.0)
        slot = env.slots[0]
        slot.vehicle = MagicMock()
        slot.waypoint_idx = 5
        slot.steps = 99
        env.reset_vehicle(slot)
        slot.vehicle.teleport.assert_called_once_with(
            (1.0, 2.0, 3.0), rot_quat=(0.0, 0.0, 0.0, 1.0), reset=True
        )
        assert slot.waypoint_idx == 0
        assert slot.steps == 0

    def test_close_removes_lidars_and_closes_bng(self):
        env = _env()
        env.bng = MagicMock()
        for s in env.slots:
            s.lidar = MagicMock()
        env.close()
        for s in env.slots:
            assert s.lidar is None
        env.bng.close.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_beamng_multi.py::TestLifecycle -v`
Expected: FAIL — `step_physics` / `reset_vehicle` / `close` not defined.

- [ ] **Step 3: Write the implementation**

Add to `BeamNGMultiEnv`. The launch / scenario-load path mirrors `BeamNGDrivingEnv` but adds every slot's vehicle at the same spawn and creates one LiDAR per slot.

```python
    def launch(self):
        """Start BeamNG and load the shared multi-vehicle scenario."""
        self.bng = BeamNGpy(
            self.host,
            self.port,
            home=self.beamng_home,
            user=self.beamng_user,
            headless=self.headless,
        )
        self.bng.open(launch=True)
        self.trajectory = self._resolve_trajectory()
        self.waypoints = list(self.trajectory.sparse_waypoints)
        self._load_scenario()

    def _resolve_trajectory(self):
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
        import time

        time.sleep(0.5)
        return load_or_generate(self.map_name, self.bng)

    def _load_scenario(self):
        import time

        self.scenario = Scenario(self.map_name, "rl_multi_driving", description="RL Multi-Agent")

        for slot in self.slots:
            vcfg = self.VEHICLES.get(slot.vehicle_id, self.VEHICLES["taxi"])
            # Per-slot colour override; vcfg already carries a default colour.
            vcfg = {**vcfg, "color": slot.color}
            slot.vehicle = Vehicle(slot.name, **vcfg)
            slot.electrics = Electrics()
            slot.damage_sensor = Damage()
            slot.vehicle.attach_sensor("electrics", slot.electrics)
            slot.vehicle.attach_sensor("damage", slot.damage_sensor)
            self.scenario.add_vehicle(
                slot.vehicle,
                pos=self.trajectory.spawn_pos,
                rot_quat=self.trajectory.spawn_rot,
                cling=True,
            )

        scales = [(5.0, 5.0, 1.0)] * len(self.waypoints)
        self.scenario.add_checkpoints(self.waypoints, scales)

        self.scenario.make(self.bng)
        self.bng.set_deterministic(30)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(1.0)

        for slot in self.slots:
            slot.lidar = Lidar(
                f"lidar_{slot.name}",
                self.bng,
                slot.vehicle,
                pos=self.LIDAR_MOUNT_POS,
                dir=self.LIDAR_MOUNT_DIR,
                up=self.LIDAR_MOUNT_UP,
                requested_update_time=0.05,
                frequency=30,
                vertical_resolution=self.LIDAR_VERT_RES,
                vertical_angle=self.LIDAR_VERT_ANGLE,
                horizontal_angle=self.LIDAR_FOV_DEG,
                max_distance=self.LIDAR_MAX_DIST,
                is_360_mode=False,
                is_rotate_mode=False,
                is_using_shared_memory=False,
                is_visualised=LIDAR_VISUALISE,
            )
            self._cache_ego_local_bbox(slot)

    def _cache_ego_local_bbox(self, slot: VehicleSlot):
        try:
            slot.vehicle.poll_sensors()
            bbox = slot.vehicle.get_bbox()
        except Exception:
            slot.ego_local_extents = None
            return
        state = slot.vehicle.state or {}
        slot.ego_local_extents = ego_local_extents_from_bbox(bbox, state, self.LIDAR_SELF_MARGIN)

    def reset_all(self):
        """Teleport every vehicle to spawn, zero episode state, prime last_obs."""
        if self.bng is None:
            self.launch()
        for slot in self.slots:
            slot.reset_episode()
            slot.vehicle.teleport(
                self.trajectory.spawn_pos, rot_quat=self.trajectory.spawn_rot, reset=True
            )
            slot.vehicle.control(throttle=0.0, steering=0.0, brake=0.0)
        self.bng.step(5)
        for slot in self.slots:
            slot.last_obs = self.observe(slot)
            slot.last_dist = slot.current_dist

    def reset_vehicle(self, slot: VehicleSlot):
        """Teleport one finished vehicle back to spawn for its next episode."""
        slot.vehicle.teleport(
            self.trajectory.spawn_pos, rot_quat=self.trajectory.spawn_rot, reset=True
        )
        slot.reset_episode()
        slot.last_obs = self.observe(slot)
        slot.last_dist = slot.current_dist

    def step_physics(self):
        """Advance every vehicle by one env step (10 physics ticks)."""
        self.bng.step(10)

    def close(self):
        if self.bng is None:
            return
        import threading

        for slot in self.slots:
            if slot.lidar is not None:
                t = threading.Thread(target=slot.lidar.remove, daemon=True)
                t.start()
                t.join(timeout=3.0)
                slot.lidar = None
        t = threading.Thread(target=self.bng.close, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.bng = None
```

Note on `reset_all`/`reset_vehicle`: in `TestLifecycle.test_reset_vehicle_teleports_and_resets_state` the slot's sensors are not wired, but `reset_vehicle` calls `observe`. To keep that test focused on teleport+state, the test wires only `slot.vehicle`; therefore split the observe call out: the test asserts teleport + state reset only. Adjust `reset_vehicle` so the observe-refresh is guarded:

```python
        if slot.lidar is not None or slot.electrics is not None:
            slot.last_obs = self.observe(slot)
            slot.last_dist = slot.current_dist
```

This guard lets the lifecycle test (no sensors wired) verify teleport/reset without needing full sensor mocks, while real runs (sensors present) still refresh `last_obs`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng_multi.py -v`
Expected: PASS (all subtask 4 tests).

- [ ] **Step 5: Lint + commit**

```bash
ruff check environments/beamng_multi.py
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "feat: BeamNGMultiEnv scene lifecycle (launch/reset/step/close)"
```

---

## Task 5: MultiAgentRunner parallel training loop

**Files:**
- Create: `core/multi_runner.py`
- Test: `tests/test_multi_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_multi_runner.py`. A `FakeEnv` and `FakeAgent` let the loop run with zero BeamNG dependency.

```python
"""Tests for core.multi_runner — parallel loop with a fake env + agents."""

import numpy as np

from core.multi_runner import MultiAgentRunner
from environments.beamng_multi import VehicleSlot


class FakeAgent:
    def __init__(self, action=0):
        self._action = action
        self.epsilon = 1.0
        self.updates = 0
        self.decays = 0
        self.saved = 0

    def select_action(self, state):
        return self._action

    def update(self, s, a, r, ns, done):
        self.updates += 1
        return 0.1

    def decay_epsilon(self):
        self.decays += 1

    def save(self, path):
        self.saved += 1


class FakeEnv:
    """Each slot finishes its episode after `episode_len` ticks."""

    def __init__(self, n_slots=2, episode_len=3, n_states=4):
        self.n_states = n_states
        self.slots = []
        for i in range(n_slots):
            agent = FakeAgent(action=i)
            self.slots.append(
                VehicleSlot(
                    name=f"ego_{i}", color="Red", vehicle_id="taxi", agent=agent,
                    reward_mode="default", action_space="discrete",
                    save_path=f"outputs/a{i}.pth",
                )
            )
            self.slots[-1].last_obs = np.zeros(n_states, dtype=np.float32)
        self.episode_len = episode_len
        self.reset_all_calls = 0
        self.step_calls = 0
        self.reset_vehicle_calls = 0

    def reset_all(self):
        self.reset_all_calls += 1
        for s in self.slots:
            s.last_obs = np.zeros(self.n_states, dtype=np.float32)

    def observe(self, slot):
        return np.zeros(self.n_states, dtype=np.float32)

    def apply_action(self, slot, action):
        pass

    def step_physics(self):
        self.step_calls += 1

    def compute_reward(self, slot, obs):
        slot.steps += 1
        done = slot.steps >= self.episode_len
        return 1.0, done

    def reset_vehicle(self, slot):
        self.reset_vehicle_calls += 1
        slot.last_obs = np.zeros(self.n_states, dtype=np.float32)

    def close(self):
        pass


class TestMultiAgentRunner:
    def test_runs_until_each_agent_completes_n_episodes(self):
        env = FakeEnv(n_slots=2, episode_len=3)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=2, time_limit=None, save_every=999)
        # 2 episodes x 2 slots completed
        for s in env.slots:
            assert s.episode == 2
            assert len(s.reward_history) == 2

    def test_steps_physics_once_per_tick(self):
        env = FakeEnv(n_slots=2, episode_len=3)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=1, time_limit=None, save_every=999)
        # 1 episode of 3 ticks -> 3 physics steps
        assert env.step_calls == 3

    def test_updates_every_agent_each_tick(self):
        env = FakeEnv(n_slots=2, episode_len=3)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=1, time_limit=None, save_every=999)
        for s in env.slots:
            assert s.agent.updates == 3
            assert s.agent.decays == 1  # one episode finished

    def test_finished_vehicle_is_reset(self):
        env = FakeEnv(n_slots=1, episode_len=2)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=2, time_limit=None, save_every=999)
        # episode boundaries trigger a reset_vehicle each (except possibly the last)
        assert env.reset_vehicle_calls >= 1

    def test_time_limit_zero_stops_immediately_after_reset(self):
        env = FakeEnv(n_slots=1, episode_len=3)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=100, time_limit=0.0, save_every=999)
        # No full episode should complete under a zero time budget
        assert env.slots[0].episode == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multi_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'core.multi_runner'`.

- [ ] **Step 3: Write the implementation**

Create `core/multi_runner.py`:

```python
"""Parallel training loop for BeamNGMultiEnv: many agents, one shared step."""

import time

import numpy as np


class MultiAgentRunner:
    """Drives N agents on one shared BeamNG scenario.

    Each tick: collect every active agent's action, step physics once, then
    update every agent. A vehicle whose episode ends is teleported to spawn and
    continues immediately while the others keep driving. The session stops when
    every agent has completed ``n_episodes`` or the wall-clock ``time_limit``
    (seconds) is reached — whichever comes first.
    """

    def train(self, env, n_episodes, time_limit=None, save_every=50):
        start = time.time()
        env.reset_all()

        def time_up():
            return time_limit is not None and (time.time() - start) >= time_limit

        def all_done():
            return all(s.episode >= n_episodes for s in env.slots)

        try:
            while not all_done() and not time_up():
                pending = []
                for slot in env.slots:
                    if slot.episode >= n_episodes:
                        continue
                    state = slot.last_obs
                    action = slot.agent.select_action(state)
                    env.apply_action(slot, action)
                    pending.append((slot, state, action))

                if not pending:
                    break

                env.step_physics()

                for slot, state, action in pending:
                    next_obs = env.observe(slot)
                    reward, done = env.compute_reward(slot, next_obs)
                    loss = slot.agent.update(state, action, reward, next_obs, done)
                    if loss is not None:
                        slot.ep_losses.append(loss)
                    slot.ep_reward += reward
                    slot.steps += 1
                    slot.last_obs = next_obs

                    if done:
                        self._finish_episode(env, slot, save_every)
        except KeyboardInterrupt:
            print("Multi-agent training interrupted by user.")
        finally:
            for slot in env.slots:
                slot.agent.save(slot.save_path)
                self._save_slot_plot(slot)

        return {
            slot.name: {
                "episodes": slot.episode,
                "rewards": slot.reward_history,
                "steps": slot.steps_history,
            }
            for slot in env.slots
        }

    def _finish_episode(self, env, slot, save_every):
        slot.reward_history.append(slot.ep_reward)
        slot.steps_history.append(slot.steps)
        slot.agent.decay_epsilon()
        if hasattr(slot.agent, "episode"):
            slot.agent.episode = slot.episode + 1
        slot.episode += 1

        avg = np.mean(slot.reward_history[-20:])
        print(
            f"[{slot.name}] ep {slot.episode} reward={slot.ep_reward:.1f} "
            f"avg20={avg:.1f} eps={getattr(slot.agent, 'epsilon', 0.0):.3f}"
        )

        if save_every and slot.episode % save_every == 0:
            slot.agent.save(slot.save_path)

        env.reset_vehicle(slot)

    def _save_slot_plot(self, slot):
        """Write a per-agent reward/steps plot, reusing PipelineRunner's plotter."""
        if not slot.reward_history:
            return
        from core.runner import PipelineRunner

        PipelineRunner._save_plot(
            slot.reward_history,
            slot.steps_history,
            slot.name,
            f"outputs/{slot.name}_multi_training.png",
            slot.episode,
        )
```

Note on the loop's per-tick `slot.steps += 1`: `compute_reward` in the real `BeamNGMultiEnv` does **not** increment `steps` (unlike the FakeEnv used in tests). The runner owns step counting. Update the `FakeEnv.compute_reward` in the test to NOT increment steps and instead let the runner do it — but the test above increments in the fake to drive its own episode length. To avoid double counting, the real env's `compute_reward` reads `slot.steps` for the MAX_STEPS check **before** the runner increments. Keep the runner as the single place that increments `slot.steps`, and in the FakeEnv test, base `done` on a separate counter.

Reconcile by changing `FakeEnv.compute_reward` to use its own tick counter rather than `slot.steps`:

```python
    def compute_reward(self, slot, obs):
        # FakeEnv tracks ticks separately so the runner remains the only writer
        # of slot.steps.
        slot._fake_ticks = getattr(slot, "_fake_ticks", 0) + 1
        done = slot._fake_ticks >= self.episode_len
        if done:
            slot._fake_ticks = 0
        return 1.0, done
```

(Use this version of `FakeEnv.compute_reward` in Step 1's test file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_multi_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
ruff check core/multi_runner.py
git add core/multi_runner.py tests/test_multi_runner.py
git commit -m "feat: MultiAgentRunner parallel training loop"
```

---

## Task 6: CLI integration

**Files:**
- Modify: `core/cli.py`
- Test: `tests/test_cli_multi.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_multi.py`. The interactive prompts aren't tested; the pure builder that turns vehicle specs into agents + env is.

```python
"""Tests for the multi-agent CLI builder helper."""

from unittest.mock import MagicMock, patch

from core.cli import build_multi_session


def test_build_multi_session_builds_one_agent_per_spec():
    specs = [
        {"algo": "dqn", "vehicle_id": "taxi", "color": "Yellow", "save_path": "outputs/dqn.pth"},
        {"algo": "ddpg", "vehicle_id": "taxi", "color": "Red", "save_path": "outputs/ddpg.pth"},
    ]
    # Patch the env so no BeamNG launch happens; capture the slots passed in.
    with patch("core.cli.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock(n_states=14)
        env, slots = build_multi_session(specs, map_name="gridmap_v2", trajectory_hints=0)

    assert len(slots) == 2
    assert slots[0].action_space == "discrete"
    assert slots[1].action_space == "continuous"
    # every slot has a concrete agent instance attached
    assert all(s.agent is not None for s in slots)


def test_build_multi_session_passes_map_and_hints_to_env():
    specs = [
        {"algo": "dqn", "vehicle_id": "taxi", "color": "Yellow", "save_path": "outputs/dqn.pth"},
    ]
    with patch("core.cli.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock(n_states=16)
        build_multi_session(specs, map_name="italy", trajectory_hints=1)
        _, kwargs = EnvCls.call_args
        assert kwargs["map_name"] == "italy"
        assert kwargs["trajectory_hints"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_multi.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_multi_session'`.

- [ ] **Step 3: Write the implementation**

In `core/cli.py`, add imports near the top:

```python
from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
from core.multi_runner import MultiAgentRunner
from environments.beamng_multi import BeamNGMultiEnv, build_slots
```

Algorithms allowed in BeamNG racing (exclude tabular `q_learning`):

```python
_MULTI_ALGOS = ["dqn", "ddpg", "td3"]
```

Add the builder (pure enough to unit-test):

```python
def build_multi_session(specs: list[dict], map_name: str, trajectory_hints: int):
    """Create the BeamNGMultiEnv and an agent per spec.

    Each agent is built against the shared observation size (env.n_states) and
    its own action dimensionality from the algorithm's registered defaults.
    Returns (env, slots).
    """
    # Construct the env first (without agents) to learn the shared n_states.
    env = BeamNGMultiEnv(
        slots=[],
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        map_name=map_name,
        trajectory_hints=trajectory_hints,
    )
    n_states = env.n_states

    enriched = []
    for spec in specs:
        algo_info = registry.get_algorithm(spec["algo"])
        cls = algo_info["class"]
        cfg = dict(algo_info["default_config"])
        cfg["n_states"] = n_states
        # Discrete (DQN) uses the 7-action table; continuous algos keep their
        # configured action dimensionality (n_actions from defaults, else 3).
        if spec["algo"] == "dqn":
            cfg["n_actions"] = BeamNGMultiEnv.N_ACTIONS_DISCRETE
        else:
            cfg.setdefault("n_actions", 3)
        cfg.pop("state_type", None)
        agent = cls(**cfg)
        enriched.append({**spec, "agent": agent})

    slots = build_slots(enriched)
    env.slots = slots
    return env, slots
```

Add the discrete action count constant to `BeamNGMultiEnv` (in `environments/beamng_multi.py`), next to the other reused constants:

```python
    N_ACTIONS_DISCRETE = len(BeamNGDrivingEnv.ACTIONS)  # 7
```

Add the interactive menu function:

```python
def _multi_train_menu():
    print("\n--- Multi-Agent Training (BeamNG) ---")
    print("\nAvailable maps:")
    map_name = _pick(_BEAMNG_MAPS, "Map")

    hints = input("\nTrajectory hints per vehicle [0]: ").strip()
    trajectory_hints = int(hints) if hints.isdigit() else 0

    vehicle_keys = list(_BEAMNG_VEHICLES.keys())
    vehicle_labels = list(_BEAMNG_VEHICLES.values())
    colors = ["Yellow", "Red", "Blue", "Green", "Orange", "White", "Black"]

    specs = []
    while True:
        print(f"\n--- Vehicle {len(specs)} ---")
        print("Algorithm:")
        algo = _pick(_MULTI_ALGOS, "Algorithm")
        print("Vehicle model:")
        vlabel = _pick(vehicle_labels, "Vehicle")
        vehicle_id = vehicle_keys[vehicle_labels.index(vlabel)]
        color = colors[len(specs) % len(colors)]
        default_path = f"outputs/{algo}_multi_{len(specs)}.pth"
        save_path = input(f"  Model save path [{default_path}]: ").strip() or default_path
        specs.append(
            {"algo": algo, "vehicle_id": vehicle_id, "color": color, "save_path": save_path}
        )
        more = input("\nAdd another vehicle? [y/N]: ").strip().lower()
        if more != "y":
            break

    if not specs:
        print("No vehicles configured.")
        return

    n_episodes = _ask_int("\nEpisodes per agent", 500)
    minutes = _ask_float("Time limit (minutes, 0 = none)", 0.0)
    time_limit = minutes * 60.0 if minutes > 0 else None

    env, slots = build_multi_session(specs, map_name, trajectory_hints)
    for slot in slots:
        if os.path.exists(slot.save_path):
            choice = (
                input(f"  [{slot.name}] '{slot.save_path}' exists. [C]ontinue / [R]eset? [C/R]: ")
                .strip()
                .lower()
            )
            if choice == "r":
                os.remove(slot.save_path)
            else:
                slot.agent.load(slot.save_path)
                slot.episode = getattr(slot.agent, "episode", 0)

    os.makedirs("outputs", exist_ok=True)
    runner = MultiAgentRunner()
    print(f"\n--- Training {len(slots)} agents on {map_name} ---\n")
    try:
        runner.train(env, n_episodes=n_episodes, time_limit=time_limit)
    finally:
        env.close()
```

Wire it into `main_menu`: add a menu line and dispatch. Update the printed menu and the choice handling:

```python
        print("6. Multi-agent training (BeamNG)")
        print("7. Quit")
```

```python
        elif choice == "6":
            _multi_train_menu()
        elif choice == "7":
            print("Bye!")
            break
```

(Renumber the existing Quit from 6 to 7.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_multi.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite + lint**

Run: `python -m pytest -q`
Expected: PASS.
Run: `ruff check core/cli.py environments/beamng_multi.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add core/cli.py environments/beamng_multi.py tests/test_cli_multi.py
git commit -m "feat: CLI multi-agent training menu + session builder"
```

---

## Task 7: Manual end-to-end verification (requires BeamNG)

This cannot be automated (needs a live BeamNG.drive). Perform it once after Tasks 1-6.

- [ ] **Step 1: Pre-warm a trajectory** (if not cached)

Run `python main.py` → option 5 (Generate trajectories) → `gridmap_v2`. Confirm `outputs/trajectories/gridmap_v2.json` exists.

- [ ] **Step 2: Launch a multi-agent session**

Run `python main.py` → option 6 → map `gridmap_v2` → hints `0` → add three vehicles:
- `dqn` / Burnside (Taxi)
- `ddpg` / Ibishu Pigeon
- `td3` / Gavril T-Series

Episodes per agent: `5`. Time limit: `2` minutes.

- [ ] **Step 3: Observe**

Confirm in the BeamNG window:
- three differently-coloured vehicles spawn at the same point and pass through each other (no collisions);
- all three drive simultaneously and each respawns to spawn when it crashes / goes off-track / hits max steps, without disturbing the others;
- per-vehicle progress lines print (`[ego_0] ep N reward=...`).

- [ ] **Step 4: Confirm artifacts**

After the session ends (time limit or 5 episodes each), confirm `outputs/dqn_multi_0.pth`, `outputs/ddpg_multi_1.pth`, `outputs/td3_multi_2.pth` exist.

- [ ] **Step 5: Record the result**

Note any issues (perf, LiDAR errors) in the PR description. No commit needed unless fixes were required.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- New env file + new runner file → Tasks 3-5 (`beamng_multi.py`) + Task 5 (`multi_runner.py`). ✓
- Shared LiDAR geometry helpers (approach A) → Tasks 1-2. ✓
- Delete NPC code → Task 2 Step 3a. ✓
- VehicleSlot with all per-vehicle state → Task 3. ✓
- Same spawn point for all vehicles → Task 4e `_load_scenario` (all `add_vehicle` use `trajectory.spawn_pos`). ✓
- One LiDAR per vehicle, created after scenario start → Task 4e. ✓
- Teleport-to-continue per-vehicle reset → Task 4e `reset_vehicle` + Task 5 `_finish_episode`. ✓
- Mixed algos on one observation (DQN/DDPG/TD3), q_learning excluded → Task 4a `build_slots`, Task 6 `_MULTI_ALGOS`. ✓
- Reward mode per slot (default vs ddpg) → Task 4c. ✓
- Session end = episodes OR time limit, whichever first → Task 5 `time_up`/`all_done`. ✓
- Per-agent checkpoints + plots → Task 5 (`save` + `_save_slot_plot`, reusing `PipelineRunner._save_plot`). ✓
- Interactive CLI selection → Task 6. ✓
- Same `trajectory_hints` for all → Task 4a (`n_states`) + Task 6 (single prompt). ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `VehicleSlot` fields, `build_slots` spec keys (`algo/agent/vehicle_id/color/save_path`), `compute_reward(slot, obs)`, `observe(slot)`, `apply_action(slot, action)`, `step_physics()`, `reset_vehicle(slot)`, `reset_all()`, `N_ACTIONS_DISCRETE` are used identically across Tasks 4-6. ✓

**No outstanding gaps.** Every spec requirement maps to a task above.
