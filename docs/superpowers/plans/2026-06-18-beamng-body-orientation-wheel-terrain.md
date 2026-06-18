# BeamNG body-orientation & wheel-terrain options — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pitch+roll (`body_orientation`) and per-wheel road position (`wheel_terrain`) two independent opt-in observation flags available to every BeamNG environment (single-agent and multi-agent), plumbed like `trajectory_hints`.

**Architecture:** The feature math lives once, as pure stateless functions in `environments/beamng_geometry.py`. Both the base `BeamNGDrivingEnv` and the separate `BeamNGMultiEnv` call those functions through thin wrappers, gated by two boolean flags. Extra blocks are appended at the *end* of the observation vector (after waypoint hints) so flag-off observations are byte-identical to today. CLI prompts and registry factories thread the flags through exactly like `trajectory_hints`.

**Tech Stack:** Python, NumPy, beamngpy (`RoadsSensor`), pytest.

## Global Constraints

- Flags default to `False` everywhere; flag-off behaviour and observation length must be unchanged for every env (regression guard).
- `body_orientation` adds exactly **2** dims `[pitch, roll]`; `wheel_terrain` adds exactly **2** dims `[left_terrain, right_terrain]`.
- Observation order is fixed: `kinematic(6) | perception(P) | hints(2·H) | [pitch,roll]? | [left,right]?`.
- `n_states = N_STATES + 2·trajectory_hints + 2·body_orientation + 2·wheel_terrain` (booleans as 0/1).
- `HALF_TRACK_WIDTH = 0.7` m (half vehicle track), reused from the deleted subclass.
- The feature math exists in exactly one place (`beamng_geometry.py`); both envs call it — no duplicated logic block.
- Observation only — no reward-function changes.
- `BeamNGContinuousRollEnv` and its `beamng_continuous_roll` registry entry are deleted; no references may remain.

---

### Task 1: Shared orientation/terrain geometry helpers

**Files:**
- Modify: `environments/beamng_geometry.py` (add two module-level functions)
- Test: `tests/test_beamng_geometry.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `body_orientation_features(dir_vec, up_vec) -> np.ndarray` → shape `(2,)` `[pitch, roll]`, each clipped to `[-1, 1]`
  - `wheel_terrain_features(roads_payload, half_track_width) -> np.ndarray` → shape `(2,)` `[left, right]`, each clipped to `[-1, 1]`; accepts a dict, a list, or `None` and falls back to neutral `(0, 0)`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_beamng_geometry.py`:

```python
class TestBodyOrientationFeatures:
    def test_flat_vehicle_reads_zero(self):
        from environments.beamng_geometry import body_orientation_features

        out = body_orientation_features((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)

    def test_nose_up_is_positive_pitch_zero_roll(self):
        from environments.beamng_geometry import body_orientation_features

        # facing +Y, body tilted nose-up: up vector leans backward (-Y)
        pitch, roll = body_orientation_features((0.0, 1.0, 0.0), (0.0, -0.3, 0.95))
        assert pitch > 0.0
        assert abs(roll) < 1e-6

    def test_lean_right_is_positive_roll(self):
        from environments.beamng_geometry import body_orientation_features

        # facing +Y (lateral axis = +X), up vector leans right (+X) -> roll > 0
        pitch, roll = body_orientation_features((0.0, 1.0, 0.0), (0.3, 0.0, 0.95))
        assert roll > 0.0
        assert abs(pitch) < 1e-6

    def test_saturates_at_one(self):
        from environments.beamng_geometry import body_orientation_features

        pitch, _ = body_orientation_features((0.0, 1.0, 0.0), (0.0, -5.0, 0.1))
        assert pitch == pytest.approx(1.0)


class TestWheelTerrainFeatures:
    def test_none_payload_is_neutral(self):
        from environments.beamng_geometry import wheel_terrain_features

        out = wheel_terrain_features(None, 0.7)
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)

    def test_dict_payload_normalizes_and_clamps(self):
        from environments.beamng_geometry import wheel_terrain_features

        left, right = wheel_terrain_features(
            {"halfWidth": 3.0, "dist2Left": 3.7, "dist2Right": 0.7}, 0.7
        )
        assert left == pytest.approx(1.0, abs=1e-6)   # (3.7-0.7)/3.0 = 1.0
        assert right == pytest.approx(0.0, abs=1e-6)  # (0.7-0.7)/3.0 = 0.0

    def test_list_payload_uses_first_element(self):
        from environments.beamng_geometry import wheel_terrain_features

        out = wheel_terrain_features(
            [{"halfWidth": 3.0, "dist2Left": 0.7, "dist2Right": 0.7}], 0.7
        )
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)

    def test_empty_list_is_neutral(self):
        from environments.beamng_geometry import wheel_terrain_features

        np.testing.assert_allclose(wheel_terrain_features([], 0.7), [0.0, 0.0], atol=1e-6)
```

Confirm `tests/test_beamng_geometry.py` imports `numpy as np` and `pytest` at the top; add them if missing.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beamng_geometry.py::TestBodyOrientationFeatures tests/test_beamng_geometry.py::TestWheelTerrainFeatures -v`
Expected: FAIL — `ImportError: cannot import name 'body_orientation_features'`.

- [ ] **Step 3: Add the two functions**

Append to `environments/beamng_geometry.py`:

```python
def body_orientation_features(dir_vec, up_vec) -> np.ndarray:
    """Return [pitch, roll] in [-1, 1] from a vehicle's forward and up vectors.

    pitch: + = nose up (uphill), - = nose down.
    roll:  + = leaning right, - = leaning left.
    A flat vehicle reads (0, 0); a 90 deg tilt saturates at +/-1. Derived by
    projecting the world-space up vector onto the vehicle's forward and lateral
    axes (the lateral axis is the forward axis rotated +90 deg in the XY plane).
    """
    fwd_len = float(np.hypot(dir_vec[0], dir_vec[1])) or 1.0
    fwd_x = dir_vec[0] / fwd_len
    fwd_y = dir_vec[1] / fwd_len
    pitch = -(float(up_vec[0]) * fwd_x + float(up_vec[1]) * fwd_y)
    roll = float(up_vec[0]) * (-fwd_y) + float(up_vec[1]) * fwd_x
    return np.array([np.clip(pitch, -1.0, 1.0), np.clip(roll, -1.0, 1.0)], dtype=np.float32)


def wheel_terrain_features(roads_payload, half_track_width) -> np.ndarray:
    """Return [left, right] road-edge position in [-1, 1] from a RoadsSensor poll.

    +1 = well on road, 0 = at the edge, -1 = off road. Measured at the
    front-axle midpoint, so this is the honest left/right road position (no
    per-wheel duplication). Accepts the raw poll payload (dict, list, or None)
    and falls back to neutral (0, 0) when data is missing.
    """
    roads = roads_payload
    if isinstance(roads, list):
        roads = roads[0] if roads else {}
    if not isinstance(roads, dict):
        roads = {}
    half_w = max(float(roads.get("halfWidth", 3.0)), 0.5)
    d_left = float(roads.get("dist2Left", half_track_width))
    d_right = float(roads.get("dist2Right", half_track_width))
    left = float(np.clip((d_left - half_track_width) / half_w, -1.0, 1.0))
    right = float(np.clip((d_right - half_track_width) / half_w, -1.0, 1.0))
    return np.array([left, right], dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng_geometry.py -v`
Expected: PASS (new tests plus the pre-existing LiDAR-geometry tests).

- [ ] **Step 5: Commit**

```bash
git add environments/beamng_geometry.py tests/test_beamng_geometry.py
git commit -m "feat: add shared body_orientation/wheel_terrain geometry helpers"
```

---

### Task 2: Flags + wrappers on `BeamNGDrivingEnv`

**Files:**
- Modify: `environments/beamng.py` (import ~24; `BeamNGDrivingEnv.__init__` ~116-172; add helper methods)
- Test: `tests/test_beamng.py`

**Interfaces:**
- Consumes: `body_orientation_features`, `wheel_terrain_features` (Task 1).
- Produces:
  - `BeamNGDrivingEnv(__init__ ..., body_orientation: bool = False, wheel_terrain: bool = False)`
  - `self.body_orientation: bool`, `self.wheel_terrain: bool`, `self.roads_sensor` (default `None`)
  - `BeamNGDrivingEnv.HALF_TRACK_WIDTH = 0.7` (class attr)
  - `_body_orientation_features(state: dict) -> np.ndarray` → `(2,)` (wrapper)
  - `_wheel_terrain_features() -> np.ndarray` → `(2,)` (wrapper, polls `self.roads_sensor`)
  - `_extra_features(state: dict) -> np.ndarray` → `(0,)`, `(2,)`, or `(4,)` depending on flags
  - `self.n_states` includes `+2` per enabled flag

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_beamng.py` (and add `from unittest.mock import MagicMock` to its imports):

```python
class TestExtraFeatures:
    def test_flags_default_off_no_extra(self):
        env = BeamNGDrivingEnv(beamng_home="unused")
        assert env.body_orientation is False
        assert env.wheel_terrain is False
        assert env._extra_features({}).shape == (0,)
        assert env.n_states == BeamNGDrivingEnv.N_STATES  # 14

    def test_n_states_accounts_for_flags(self):
        base = BeamNGDrivingEnv.N_STATES
        assert BeamNGDrivingEnv(beamng_home="x", body_orientation=True).n_states == base + 2
        assert BeamNGDrivingEnv(beamng_home="x", wheel_terrain=True).n_states == base + 2
        both = BeamNGDrivingEnv(beamng_home="x", body_orientation=True, wheel_terrain=True)
        assert both.n_states == base + 4

    def test_n_states_combines_flags_and_hints(self):
        env = BeamNGDrivingEnv(
            beamng_home="x", trajectory_hints=2, body_orientation=True, wheel_terrain=True
        )
        assert env.n_states == BeamNGDrivingEnv.N_STATES + 4 + 2 + 2

    def test_extra_features_order_is_orientation_then_terrain(self):
        env = BeamNGDrivingEnv(beamng_home="x", body_orientation=True, wheel_terrain=True)
        env.roads_sensor = None
        state = {"dir": (0.0, 1.0, 0.0), "up": (0.0, -0.3, 0.95)}
        out = env._extra_features(state)
        assert out.shape == (4,)
        assert out[0] > 0.0          # pitch (nose up) first
        assert out[2] == pytest.approx(0.0, abs=1e-6)  # left terrain (neutral) after

    def test_wheel_terrain_wrapper_reads_sensor(self):
        env = BeamNGDrivingEnv(beamng_home="x", wheel_terrain=True)
        env.roads_sensor = MagicMock()
        env.roads_sensor.poll.return_value = {"halfWidth": 3.0, "dist2Left": 3.7, "dist2Right": 0.7}
        left, right = env._wheel_terrain_features()
        assert left == pytest.approx(1.0, abs=1e-6)
        assert right == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beamng.py::TestExtraFeatures -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'body_orientation'`.

- [ ] **Step 3: Import the shared helpers**

In `environments/beamng.py`, extend the geometry import (~line 24):

```python
from environments.beamng_geometry import (
    LidarConfig,
    body_orientation_features,
    wheel_terrain_features,
)
```

- [ ] **Step 4: Add the class attribute and constructor params**

Add the class attribute near the other tunables (after `MAX_DAMAGE`, ~line 85):

```python
    HALF_TRACK_WIDTH = 0.7  # metres — half vehicle track, for per-wheel road-edge projection
```

Change the `__init__` signature (~line 116) to add the two params after `trajectory_hints`:

```python
        map_name: str = "gridmap_v2",
        trajectory_hints: int = 0,
        body_orientation: bool = False,
        wheel_terrain: bool = False,
    ):
```

Inside `__init__`, replace the `self.n_states = ...` line (~160) and add state:

```python
        self.trajectory_hints = trajectory_hints
        self.body_orientation = body_orientation
        self.wheel_terrain = wheel_terrain
        self.n_states = (
            self.N_STATES
            + trajectory_hints * 2
            + (2 if body_orientation else 0)
            + (2 if wheel_terrain else 0)
        )
```

Add `self.roads_sensor = None` alongside the other sensor attributes (~line 146):

```python
        self.lidar: Lidar = None
        self.roads_sensor: RoadsSensor = None
```

- [ ] **Step 5: Add the wrapper helpers**

Add these methods to `BeamNGDrivingEnv` (place them just after `_get_waypoint_hints`, ~line 738):

```python
    def _body_orientation_features(self, state) -> np.ndarray:
        """[pitch, roll] from the vehicle's forward/up vectors (see geometry helper)."""
        return body_orientation_features(
            state.get("dir", (0.0, 1.0, 0.0)), state.get("up", (0.0, 0.0, 1.0))
        )

    def _wheel_terrain_features(self) -> np.ndarray:
        """[left, right] road-edge position from the RoadsSensor (neutral without one)."""
        payload = self.roads_sensor.poll() if self.roads_sensor is not None else None
        return wheel_terrain_features(payload, self.HALF_TRACK_WIDTH)

    def _extra_features(self, state) -> np.ndarray:
        """Optional observation tail: body orientation and/or wheel terrain.

        Appended after the waypoint hints. Empty array when both flags are off,
        so flag-off observations are unchanged.
        """
        blocks = []
        if self.body_orientation:
            blocks.append(self._body_orientation_features(state))
        if self.wheel_terrain:
            blocks.append(self._wheel_terrain_features())
        if not blocks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(blocks)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng.py::TestExtraFeatures -v`
Expected: PASS (all 5 tests).

- [ ] **Step 7: Commit**

```bash
git add environments/beamng.py tests/test_beamng.py
git commit -m "feat: add body_orientation/wheel_terrain flags + wrappers to BeamNGDrivingEnv"
```

---

### Task 3: Wire extras into observation + RoadsSensor lifecycle; delete `BeamNGContinuousRollEnv`

**Files:**
- Modify: `environments/beamng.py` (`_observe` ~444-494; `_load_scenario` ~394-442; `close` ~329-343; add sensor helpers after `_remove_lidar` ~353; `BeamNGCameraEnv._load_scenario` ~1161; `BeamNGCameraEnv._observe` ~1207; delete `BeamNGContinuousRollEnv` ~1025-1110)
- Test: `tests/test_beamng.py`

**Interfaces:**
- Consumes: `_extra_features`, `body_orientation`, `wheel_terrain`, `roads_sensor` (Task 2).
- Produces:
  - `_attach_roads_sensor()` — creates `self.roads_sensor` when `wheel_terrain` is on (no-op otherwise)
  - `_remove_roads_sensor()` — tears it down
  - Both `_observe` implementations end with `self._extra_features(state)` appended
  - `BeamNGContinuousRollEnv` no longer exists

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_beamng.py`:

```python
class TestRoadsSensorLifecycle:
    def test_attach_roads_sensor_noop_when_flag_off(self):
        env = BeamNGDrivingEnv(beamng_home="x", wheel_terrain=False)
        env.bng = MagicMock()
        env.vehicle = MagicMock()
        env._attach_roads_sensor()
        assert env.roads_sensor is None

    def test_remove_roads_sensor_clears_handle(self):
        env = BeamNGDrivingEnv(beamng_home="x", wheel_terrain=True)
        env.roads_sensor = MagicMock()
        env._remove_roads_sensor()
        assert env.roads_sensor is None


class TestContinuousRollDeleted:
    def test_class_is_gone(self):
        import environments.beamng as m
        assert not hasattr(m, "BeamNGContinuousRollEnv")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beamng.py::TestRoadsSensorLifecycle tests/test_beamng.py::TestContinuousRollDeleted -v`
Expected: FAIL — `AttributeError: ... _attach_roads_sensor`; deletion test fails (class still present).

- [ ] **Step 3: Add the sensor lifecycle helpers**

Add to `BeamNGDrivingEnv`, right after `_remove_lidar` (~line 353):

```python
    def _attach_roads_sensor(self):
        """Attach a RoadsSensor when wheel_terrain is on; replace any prior one."""
        if not self.wheel_terrain:
            return
        self._remove_roads_sensor()
        self.roads_sensor = RoadsSensor("roads", self.bng, self.vehicle)

    def _remove_roads_sensor(self):
        if self.roads_sensor is None:
            return
        t = threading.Thread(target=self.roads_sensor.remove, daemon=True)
        t.start()
        t.join(timeout=3.0)
        self.roads_sensor = None
```

- [ ] **Step 4: Wire into base `_load_scenario`, `close`, and `_observe`**

In `BeamNGDrivingEnv._load_scenario`, where `self._remove_lidar()` is called at the top (~line 399), add the roads removal right after it:

```python
        self._remove_lidar()
        self._remove_roads_sensor()
```

At the end of `_load_scenario`, after `self._update_active_marker(0)` (~line 442), add:

```python
        self._attach_roads_sensor()
```

In `BeamNGDrivingEnv.close`, after `self._remove_lidar()` (~line 337), add:

```python
            self._remove_lidar()
            self._remove_roads_sensor()
```

In `BeamNGDrivingEnv._observe`, the `state` dict is already fetched (~line 455). Append the extras as a fourth block of the final `np.concatenate` (~line 476-492):

```python
                lidar_bins,
                waypoint_hints,
                self._extra_features(state),
            ]
        )

        return obs
```

- [ ] **Step 5: Wire into `BeamNGCameraEnv`**

In `BeamNGCameraEnv._load_scenario`, after `self._update_active_marker(0)` (~line 1205), add:

```python
        self._attach_roads_sensor()
```

(The camera `_load_scenario` does not call `super()`, so it needs its own attach. Its `close()` already calls `super().close()`, which now removes the roads sensor — no change needed there.)

In `BeamNGCameraEnv._observe`, `state` is already fetched (~line 1217). Append the extras to its final `np.concatenate` (~line 1232-1248):

```python
                cam_pixels,
                waypoint_hints,
                self._extra_features(state),
            ]
        )
```

- [ ] **Step 6: Delete `BeamNGContinuousRollEnv`**

Remove the entire `class BeamNGContinuousRollEnv(BeamNGContinuousEnv):` block (~lines 1025-1110, from the class line up to but not including `class BeamNGCameraEnv`). The `RoadsSensor` import at the top of the file stays (now used by the base class).

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng.py -v`
Expected: PASS (new lifecycle + deletion tests pass; pre-existing tests unchanged).

- [ ] **Step 8: Commit**

```bash
git add environments/beamng.py tests/test_beamng.py
git commit -m "feat: append optional extras to BeamNG observations; drop BeamNGContinuousRollEnv"
```

---

### Task 4: Registry — forward flags, delete `beamng_continuous_roll`

**Files:**
- Modify: `environments/__init__.py` (all `_make_beamng*` factories; delete `_make_beamng_continuous_roll` ~102-119)
- Test: `tests/test_beamng.py` (registry assertion)

**Interfaces:**
- Consumes: `BeamNGDrivingEnv(..., body_orientation, wheel_terrain)` (Task 2); deleted class (Task 3).
- Produces: every `_make_beamng*` factory accepts and forwards `body_orientation=False, wheel_terrain=False`; no `beamng_continuous_roll` registration.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_beamng.py`:

```python
class TestRegistry:
    def test_continuous_roll_not_registered(self):
        from core.registry import registry
        import environments  # noqa: F401  (triggers registration)
        assert "beamng_continuous_roll" not in registry.list_environments()

    def test_beamng_factory_forwards_flags(self):
        from environments import _make_beamng
        env = _make_beamng(body_orientation=True, wheel_terrain=True)
        assert env.body_orientation is True
        assert env.wheel_terrain is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_beamng.py::TestRegistry -v`
Expected: FAIL — `beamng_continuous_roll` still registered / `_make_beamng` rejects the kwargs.

- [ ] **Step 3: Add flag params to every factory**

In `environments/__init__.py`, update each factory to accept and forward the flags.

`_make_beamng` (~line 13):

```python
def _make_beamng(
    reward_mode="default",
    vehicle_id="taxi",
    map_name="gridmap_v2",
    trajectory_hints=0,
    body_orientation=False,
    wheel_terrain=False,
):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGDrivingEnv

    return BeamNGDrivingEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        reward_mode=reward_mode,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=trajectory_hints,
        body_orientation=body_orientation,
        wheel_terrain=wheel_terrain,
    )
```

`_make_beamng_lidar`, `_make_beamng_continuous`, `_make_beamng_camera` (~37, ~58, ~81): add `body_orientation=False, wheel_terrain=False` to each signature (before `**_kwargs`) and pass both to the constructor. Example for `_make_beamng_lidar`:

```python
def _make_beamng_lidar(
    vehicle_id="taxi",
    map_name="gridmap_v2",
    trajectory_hints=0,
    body_orientation=False,
    wheel_terrain=False,
    **_kwargs,
):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGLidarEnv

    return BeamNGLidarEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=trajectory_hints,
        body_orientation=body_orientation,
        wheel_terrain=wheel_terrain,
    )
```

Apply the identical pattern to `_make_beamng_continuous` and `_make_beamng_camera`.

- [ ] **Step 4: Delete `_make_beamng_continuous_roll` and its registration**

Remove the `_make_beamng_continuous_roll` function and the `registry.register_environment("beamng_continuous_roll", ...)` block (~lines 102-119) entirely.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng.py::TestRegistry -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add environments/__init__.py tests/test_beamng.py
git commit -m "feat: forward orientation/terrain flags through registry; drop beamng_continuous_roll"
```

---

### Task 5: Multi-agent env — slot flags, sizing, per-slot sensor, observation

**Files:**
- Modify: `environments/beamng_multi.py` (imports ~15, ~26-30; `VehicleSlot` ~33-91; `slot_n_states` ~138-141; `build_slots` ~163-194; `_create_slot_sensor` ~561-604; `observe` ~420-463; `close` ~701-717; add helper + `HALF_TRACK_WIDTH`)
- Test: `tests/test_beamng_multi.py`

**Interfaces:**
- Consumes: `body_orientation_features`, `wheel_terrain_features` (Task 1).
- Produces:
  - `VehicleSlot(..., body_orientation: bool = False, wheel_terrain: bool = False, roads_sensor: Any = None)`
  - `slot_n_states(env_name, trajectory_hints=0, body_orientation=False, wheel_terrain=False) -> int`
  - `build_slots` reads `body_orientation`/`wheel_terrain` from each spec dict
  - `BeamNGMultiEnv._slot_extra_features(slot, state) -> np.ndarray`
  - `BeamNGMultiEnv.HALF_TRACK_WIDTH = 0.7`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_beamng_multi.py`:

```python
class TestSlotExtraFeatures:
    def test_slot_n_states_with_flags(self):
        assert slot_n_states("beamng", body_orientation=True) == 16        # 14 + 2
        assert slot_n_states("beamng", wheel_terrain=True) == 16           # 14 + 2
        assert slot_n_states("beamng", body_orientation=True, wheel_terrain=True) == 18
        assert (
            slot_n_states("beamng", trajectory_hints=1, body_orientation=True, wheel_terrain=True)
            == 14 + 2 + 2 + 2
        )

    def test_build_slots_reads_flags(self):
        specs = [
            {
                "algo": "dqn",
                "env": "beamng",
                "agent": _FakeAgent(),
                "vehicle_id": "taxi",
                "color": "Yellow",
                "save_path": "outputs/x.pth",
                "body_orientation": True,
                "wheel_terrain": True,
            }
        ]
        slot = build_slots(specs)[0]
        assert slot.body_orientation is True
        assert slot.wheel_terrain is True
        assert slot.n_states == 18

    def test_build_slots_flags_default_off(self):
        slot = build_slots(SPECS)[0]
        assert slot.body_orientation is False
        assert slot.wheel_terrain is False
        assert slot.n_states == 14
```

Extend the `TestObserve` class with a flagged-observation test:

```python
    def test_observe_appends_extras_when_flags_on(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]
        slot.body_orientation = True
        slot.wheel_terrain = True
        slot.n_states = 18
        self._wire_slot_sensors(
            slot, speed=10.0, steering=0.0, damage=0.0,
            pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), lidar_points=None,
        )
        slot.vehicle.state = {"pos": (0.0, 0.0, 0.0), "vel": (1.0, 0.0, 0.0),
                              "dir": (1.0, 0.0, 0.0), "up": (0.0, 0.0, 1.0)}
        slot.roads_sensor = MagicMock()
        slot.roads_sensor.poll.return_value = {"halfWidth": 3.0, "dist2Left": 0.7, "dist2Right": 0.7}
        obs = env.observe(slot)
        assert obs.shape == (18,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beamng_multi.py::TestSlotExtraFeatures tests/test_beamng_multi.py::TestObserve -v`
Expected: FAIL — `slot_n_states() got an unexpected keyword argument 'body_orientation'` / `VehicleSlot` has no such field.

- [ ] **Step 3: Import shared helpers + RoadsSensor and add slot fields + class constant**

In `environments/beamng_multi.py`, add `RoadsSensor` to the beamngpy import (~line 15) and the `except ImportError` fallback (~line 17):

```python
    from beamngpy.sensors import Camera, Damage, Electrics, Lidar, RoadsSensor
except ImportError:
    BeamNGpy = Scenario = Vehicle = Camera = Damage = Electrics = Lidar = RoadsSensor = None
```

Extend the geometry import (~line 26):

```python
from environments.beamng_geometry import (
    LidarConfig,
    body_orientation_features,
    ego_local_extents_from_bbox,
    process_lidar,
    wheel_terrain_features,
)
```

Add fields to `VehicleSlot`. Under the "Environment profile" group (~line 50-54):

```python
    perception: str = "lidar"  # "lidar" | "lidar_grid" | "camera"
    trajectory_hints: int = 0
    body_orientation: bool = False
    wheel_terrain: bool = False
    n_states: int = 14  # observation length for this vehicle's env
```

Under the "Sensors" group (~line 58-62):

```python
    lidar: Any = None
    camera: Any = None
    roads_sensor: Any = None
```

Add the class constant to `BeamNGMultiEnv` near `GRID_LANE_OFFSET` (~line 233):

```python
    HALF_TRACK_WIDTH = 0.7  # metres — half vehicle track, for per-wheel road-edge projection
```

- [ ] **Step 4: Update `slot_n_states` and `build_slots`**

Replace `slot_n_states` (~line 138):

```python
def slot_n_states(
    env_name: str,
    trajectory_hints: int = 0,
    body_orientation: bool = False,
    wheel_terrain: bool = False,
) -> int:
    """Observation length for a vehicle running the given env with the given options."""
    perception = env_profile(env_name)
    return (
        _KINEMATIC_FEATURES
        + _PERCEPTION_FEATURES[perception]
        + 2 * trajectory_hints
        + (2 if body_orientation else 0)
        + (2 if wheel_terrain else 0)
    )
```

In `build_slots` (~line 172), read the flags and pass them through:

```python
        algo = spec["algo"]
        env_name = spec.get("env", "beamng")
        trajectory_hints = spec.get("trajectory_hints", 0)
        body_orientation = spec.get("body_orientation", False)
        wheel_terrain = spec.get("wheel_terrain", False)
        perception = env_profile(env_name)
        continuous = algo in _CONTINUOUS_ALGOS
        ddpg_reward = continuous and perception in ("lidar", "lidar_grid")
        slots.append(
            VehicleSlot(
                name=f"ego_{i}",
                color=spec["color"],
                vehicle_id=spec["vehicle_id"],
                agent=spec["agent"],
                reward_mode="ddpg" if ddpg_reward else "default",
                action_space="continuous" if continuous else "discrete",
                save_path=spec["save_path"],
                env_name=env_name,
                perception=perception,
                trajectory_hints=trajectory_hints,
                body_orientation=body_orientation,
                wheel_terrain=wheel_terrain,
                n_states=slot_n_states(
                    env_name, trajectory_hints, body_orientation, wheel_terrain
                ),
            )
        )
```

- [ ] **Step 5: Add the slot-extras helper and wire into `observe`**

Add the helper to `BeamNGMultiEnv` (place after `_waypoint_hints`, ~line 493):

```python
    def _slot_extra_features(self, slot, state) -> np.ndarray:
        """Optional observation tail for a slot (body orientation / wheel terrain).

        Calls the shared geometry helpers; empty when both flags are off.
        """
        blocks = []
        if slot.body_orientation:
            blocks.append(
                body_orientation_features(
                    state.get("dir", (0.0, 1.0, 0.0)), state.get("up", (0.0, 0.0, 1.0))
                )
            )
        if slot.wheel_terrain:
            payload = slot.roads_sensor.poll() if slot.roads_sensor is not None else None
            blocks.append(wheel_terrain_features(payload, self.HALF_TRACK_WIDTH))
        if not blocks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(blocks)
```

In `observe` (~line 447), append the extras as the final block of the `np.concatenate`. The `state` dict is already in scope (`state = slot.vehicle.state or {}`, ~line 430):

```python
                perception,
                waypoint_hints,
                self._slot_extra_features(slot, state),
            ]
        )
```

- [ ] **Step 6: Attach the per-slot RoadsSensor and tear it down**

In `_create_slot_sensor`, attach the roads sensor at the very top of the method (~line 568), before the `if slot.perception == "camera":` early-return branch:

```python
        if slot.wheel_terrain:
            slot.roads_sensor = RoadsSensor(f"roads_{slot.name}", self.bng, slot.vehicle)

        if slot.perception == "camera":
            slot.camera = Camera(
```

In `close` (~line 707), add `"roads_sensor"` to the removal loop:

```python
        for slot in self.slots:
            for sensor_attr in ("lidar", "camera", "roads_sensor"):
                sensor = getattr(slot, sensor_attr)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng_multi.py -v`
Expected: PASS (new flag/observe tests pass; all pre-existing multi tests unchanged).

- [ ] **Step 8: Commit**

```bash
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "feat: per-slot body_orientation/wheel_terrain options in BeamNGMultiEnv"
```

---

### Task 6: CLI — prompts thread flags into single & multi sessions

**Files:**
- Modify: `core/cli.py` (add `_ask_bool` after `_ask_int` ~line 41; single-train ~147-164; single-eval/play ~248-265; `build_multi_session` ~415-432; `_multi_train_menu` spec builder ~459-471)
- Test: `tests/test_cli_multi.py`

**Interfaces:**
- Consumes: `slot_n_states(env_name, trajectory_hints, body_orientation, wheel_terrain)` (Task 5); registry factories forwarding flags (Task 4).
- Produces: `_ask_bool(prompt: str, default: bool = False) -> bool`; spec dicts and `beamng_kwargs` carrying `body_orientation`/`wheel_terrain`; agent `n_states` sized with the flags.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli_multi.py`:

```python
def test_build_multi_session_sizes_agent_with_flags():
    # Body orientation + wheel terrain on a lidar env -> 14 + 2 + 2 = 18 states.
    specs = [
        {
            "algo": "dqn",
            "env": "beamng",
            "vehicle_id": "taxi",
            "color": "Yellow",
            "save_path": "outputs/multi-agents/dqn.pth",
            "body_orientation": True,
            "wheel_terrain": True,
        },
    ]
    with patch("core.cli.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        _, slots = build_multi_session(specs, map_name="gridmap_v2")
    assert slots[0].n_states == 18
    assert slots[0].body_orientation is True
    assert slots[0].wheel_terrain is True


def test_ask_bool_parses_yes_no():
    from core.cli import _ask_bool
    with patch("builtins.input", return_value="y"):
        assert _ask_bool("?") is True
    with patch("builtins.input", return_value=""):
        assert _ask_bool("?", default=False) is False
    with patch("builtins.input", return_value="yes"):
        assert _ask_bool("?") is True
    with patch("builtins.input", return_value="n"):
        assert _ask_bool("?", default=True) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_multi.py::test_build_multi_session_sizes_agent_with_flags tests/test_cli_multi.py::test_ask_bool_parses_yes_no -v`
Expected: FAIL — `cannot import name '_ask_bool'`; `slot_n_states` ignores flags so n_states would be 14.

- [ ] **Step 3: Add the `_ask_bool` helper**

In `core/cli.py`, after `_ask_int` (~line 41):

```python
def _ask_bool(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")
```

- [ ] **Step 4: Thread flags through `build_multi_session`**

In `build_multi_session` (~line 420), read the flags and pass to `slot_n_states`:

```python
        trajectory_hints = spec.get("trajectory_hints", 0)
        body_orientation = spec.get("body_orientation", False)
        wheel_terrain = spec.get("wheel_terrain", False)
        cfg["n_states"] = slot_n_states(
            spec.get("env", "beamng"), trajectory_hints, body_orientation, wheel_terrain
        )
```

- [ ] **Step 5: Add prompts to the multi spec builder**

In `_multi_train_menu`, after the `hints = _ask_int(...)` line (~line 459), add the prompts and include them in the spec dict (~line 462):

```python
        hints = _ask_int("Checkpoint hints (waypoints ahead in obs, 0 = none)", 0, min_val=0)
        body_orientation = _ask_bool("Include body orientation (pitch + roll) in obs?")
        wheel_terrain = _ask_bool("Include per-wheel road position in obs?")
        default_path = os.path.join(_MULTI_OUTPUT_DIR, f"{algo}_{env_name}_{len(specs)}.pth")
        save_path = input(f"  Model save path [{default_path}]: ").strip() or default_path
        specs.append(
            {
                "algo": algo,
                "env": env_name,
                "vehicle_id": vehicle_id,
                "color": color,
                "save_path": save_path,
                "trajectory_hints": hints,
                "body_orientation": body_orientation,
                "wheel_terrain": wheel_terrain,
            }
        )
```

- [ ] **Step 6: Add prompts to the single-agent train path**

In the single-agent train menu, replace the `beamng_kwargs` block (~line 147-164) so it asks and forwards the flags and sizes `n_states`:

```python
    beamng_kwargs = {}
    trajectory_hints = 0
    body_orientation = False
    wheel_terrain = False
    if env_name.startswith("beamng"):
        map_name, vehicle_id = _pick_beamng_options()
        trajectory_hints = _ask_int(
            "\nCheckpoint hints (waypoints ahead in obs, 0 = none)", 0, min_val=0
        )
        body_orientation = _ask_bool("Include body orientation (pitch + roll) in obs?")
        wheel_terrain = _ask_bool("Include per-wheel road position in obs?")
        beamng_kwargs = {
            "map_name": map_name,
            "vehicle_id": vehicle_id,
            "trajectory_hints": trajectory_hints,
            "body_orientation": body_orientation,
            "wheel_terrain": wheel_terrain,
        }

    # Adjust n_states for the chosen options before building the agent
    extra_states = trajectory_hints * 2 + (2 if body_orientation else 0) + (2 if wheel_terrain else 0)
    env_meta = {
        **env_info["metadata"],
        "n_states": env_info["metadata"]["n_states"] + extra_states,
    }
```

- [ ] **Step 7: Add prompts to the single-agent eval/play path**

In the eval menu, replace the `beamng_kwargs` block (~line 248-264) with the same pattern, keeping the "must match the trained model" wording:

```python
    beamng_kwargs = {}
    trajectory_hints = 0
    body_orientation = False
    wheel_terrain = False
    if env_name.startswith("beamng"):
        map_name, vehicle_id = _pick_beamng_options()
        trajectory_hints = _ask_int(
            "\nCheckpoint hints (must match the trained model)", 0, min_val=0
        )
        body_orientation = _ask_bool("Body orientation in obs? (must match the trained model)")
        wheel_terrain = _ask_bool("Per-wheel road position in obs? (must match the trained model)")
        beamng_kwargs = {
            "map_name": map_name,
            "vehicle_id": vehicle_id,
            "trajectory_hints": trajectory_hints,
            "body_orientation": body_orientation,
            "wheel_terrain": wheel_terrain,
        }

    extra_states = trajectory_hints * 2 + (2 if body_orientation else 0) + (2 if wheel_terrain else 0)
    env_meta = {
        **env_info["metadata"],
        "n_states": env_info["metadata"]["n_states"] + extra_states,
    }
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli_multi.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add core/cli.py tests/test_cli_multi.py
git commit -m "feat: CLI prompts for body_orientation/wheel_terrain in single and multi sessions"
```

---

### Task 7: Regression sweep & docs

**Files:**
- Modify: `docs/romain.md` (note the fifth-issue fix is now an option); any stray references.
- Test: full suite.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: a clean tree with no `beamng_continuous_roll` / `BeamNGContinuousRollEnv` references and a green test suite.

- [ ] **Step 1: Confirm no stale references remain**

Run: `git grep -n "continuous_roll\|BeamNGContinuousRollEnv"`
Expected: only matches inside `docs/superpowers/` historical specs/plans (acceptable). No matches in `environments/`, `core/`, or `tests/`. If any appear in code/tests, remove them.

- [ ] **Step 2: Note the fix in `docs/romain.md`**

Under the "Fifth issue" paragraph (~line 24), append:

```markdown
This is now an opt-in observation: `body_orientation` (pitch + roll) and
`wheel_terrain` (per-wheel road position) can be toggled per environment and
per vehicle, instead of being hardcoded into a single env.
```

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: PASS (no failures, no errors). Confirm flag-off `n_states` values still read 14 / 38 / 262 in `test_beamng_multi.py`.

- [ ] **Step 4: Lint/format**

Run: `ruff format environments/ core/ tests/ && ruff check environments/ core/ tests/`
Expected: no changes needed / no lint errors. Fix anything reported.

- [ ] **Step 5: Commit**

```bash
git add docs/romain.md
git commit -m "docs: note body_orientation/wheel_terrain are now opt-in options"
```

---

## Self-Review

**Spec coverage:**
- Feature math in exactly one place → Task 1 (`beamng_geometry.py`); both envs call it (Tasks 2, 5). ✓
- Two independent flags → Tasks 2, 4, 5, 6. ✓
- Available to every BeamNG env (lidar/lidar_grid/camera/continuous, single + multi) → base class (Tasks 2/3), camera override (Task 3), registry (Task 4), multi (Task 5). ✓
- Honest 2-value wheel terrain → `wheel_terrain_features` returns `(2,)` (Task 1). ✓
- Append at end / order fixed → Tasks 3, 5. ✓
- `n_states` formula → Tasks 2, 5, 6. ✓
- RoadsSensor lifecycle + safe fallback → Tasks 1, 3, 5. ✓
- Delete `BeamNGContinuousRollEnv` + registration → Tasks 3, 4, 7. ✓
- CLI prompts (train, play, multi) → Task 6. ✓
- Tests for flag combinations + regression → Tasks 1, 2, 5, 6, 7. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** shared `body_orientation_features(dir_vec, up_vec)` / `wheel_terrain_features(roads_payload, half_track_width)` (Task 1); single-env wrappers `_body_orientation_features(state)` / `_wheel_terrain_features()` / `_extra_features(state)` (Task 2); `_attach_roads_sensor()` / `_remove_roads_sensor()` (Task 3); `_slot_extra_features(slot, state)` (Task 5); `slot_n_states(env_name, trajectory_hints, body_orientation, wheel_terrain)` (Task 5); `_ask_bool(prompt, default)` (Task 6) — names used consistently across tasks. ✓
