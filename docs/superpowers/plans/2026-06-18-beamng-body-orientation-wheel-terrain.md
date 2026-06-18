# BeamNG body-orientation & wheel-terrain options — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make pitch+roll (`body_orientation`) and per-wheel road position (`wheel_terrain`) two independent opt-in observation flags available to every BeamNG environment (single-agent and multi-agent), plumbed like `trajectory_hints`.

**Architecture:** Lift the feature math out of the one-off `BeamNGContinuousRollEnv` subclass into helper methods on the base `BeamNGDrivingEnv`, gated by two boolean flags. Extra blocks are appended at the *end* of the observation vector (after waypoint hints) so flag-off observations are byte-identical to today. The multi-agent env (`BeamNGMultiEnv`, a separate class) mirrors the same helpers per slot. CLI prompts and registry factories thread the flags through exactly like `trajectory_hints`.

**Tech Stack:** Python, NumPy, beamngpy (`RoadsSensor`), pytest.

## Global Constraints

- Flags default to `False` everywhere; flag-off behaviour and observation length must be unchanged for every env (regression guard).
- `body_orientation` adds exactly **2** dims `[pitch, roll]`; `wheel_terrain` adds exactly **2** dims `[left_terrain, right_terrain]`.
- Observation order is fixed: `kinematic(6) | perception(P) | hints(2·H) | [pitch,roll]? | [left,right]?`.
- `n_states = N_STATES + 2·trajectory_hints + 2·body_orientation + 2·wheel_terrain` (booleans as 0/1).
- `HALF_TRACK_WIDTH = 0.7` m (half vehicle track), reused from the deleted subclass.
- Observation only — no reward-function changes.
- `BeamNGContinuousRollEnv` and its `beamng_continuous_roll` registry entry are deleted; no references may remain.

---

### Task 1: Feature helpers + flags on `BeamNGDrivingEnv`

**Files:**
- Modify: `environments/beamng.py` (`BeamNGDrivingEnv.__init__` ~116-172; add helper methods)
- Test: `tests/test_beamng.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `BeamNGDrivingEnv(__init__ ..., body_orientation: bool = False, wheel_terrain: bool = False)`
  - `self.body_orientation: bool`, `self.wheel_terrain: bool`, `self.roads_sensor` (default `None`)
  - `BeamNGDrivingEnv.HALF_TRACK_WIDTH = 0.7` (class attr)
  - `_body_orientation_features(state: dict) -> np.ndarray` → shape `(2,)` `[pitch, roll]`
  - `_wheel_terrain_features() -> np.ndarray` → shape `(2,)` `[left, right]`
  - `_extra_features(state: dict) -> np.ndarray` → shape `(0,)`, `(2,)`, or `(4,)` depending on flags
  - `self.n_states` includes `+2` per enabled flag

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_beamng.py`:

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

    def test_body_orientation_flat_vehicle_reads_zero(self):
        env = BeamNGDrivingEnv(beamng_home="x", body_orientation=True)
        state = {"dir": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0)}
        out = env._body_orientation_features(state)
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)

    def test_body_orientation_nose_up_is_positive_pitch(self):
        env = BeamNGDrivingEnv(beamng_home="x", body_orientation=True)
        # facing +Y, body tilted nose-up: up vector leans backward (-Y)
        state = {"dir": (0.0, 1.0, 0.0), "up": (0.0, -0.3, 0.95)}
        pitch, roll = env._body_orientation_features(state)
        assert pitch > 0.0
        assert abs(roll) < 1e-6

    def test_wheel_terrain_defaults_to_neutral_without_sensor(self):
        env = BeamNGDrivingEnv(beamng_home="x", wheel_terrain=True)
        env.roads_sensor = None
        out = env._wheel_terrain_features()
        assert out.shape == (2,)
        # dist2Left/Right default to HALF_TRACK_WIDTH -> (0 - 0)/half_w == 0
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)

    def test_wheel_terrain_reads_sensor_and_clamps(self):
        env = BeamNGDrivingEnv(beamng_home="x", wheel_terrain=True)
        env.roads_sensor = MagicMock()
        env.roads_sensor.poll.return_value = {
            "halfWidth": 3.0, "dist2Left": 3.7, "dist2Right": 0.7
        }
        left, right = env._wheel_terrain_features()
        assert left == pytest.approx(1.0, abs=1e-6)   # (3.7-0.7)/3.0 = 1.0
        assert right == pytest.approx(0.0, abs=1e-6)  # (0.7-0.7)/3.0 = 0.0

    def test_wheel_terrain_handles_list_payload(self):
        env = BeamNGDrivingEnv(beamng_home="x", wheel_terrain=True)
        env.roads_sensor = MagicMock()
        env.roads_sensor.poll.return_value = [{"halfWidth": 3.0, "dist2Left": 0.7, "dist2Right": 0.7}]
        out = env._wheel_terrain_features()
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)
```

Add `from unittest.mock import MagicMock` to the test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_beamng.py::TestExtraFeatures -v`
Expected: FAIL — `__init__() got an unexpected keyword argument 'body_orientation'` / `AttributeError: ... _extra_features`.

- [ ] **Step 3: Add the class attribute and constructor params**

In `environments/beamng.py`, add the class attribute near the other tunables (after `MAX_DAMAGE`, ~line 85):

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

Add `self.roads_sensor = None` alongside the other sensor attributes (~line 146, near `self.lidar: Lidar = None`):

```python
        self.lidar: Lidar = None
        self.roads_sensor: RoadsSensor = None
```

- [ ] **Step 4: Add the helper methods**

Add these methods to `BeamNGDrivingEnv` (place them just after `_get_waypoint_hints`, ~line 738):

```python
    def _body_orientation_features(self, state) -> np.ndarray:
        """Return [pitch, roll] in [-1, 1] from the vehicle's up/forward vectors.

        pitch: + = nose up (uphill), - = nose down.
        roll:  + = leaning right, - = leaning left.
        """
        dir_vec = state.get("dir", (0.0, 1.0, 0.0))
        up_vec = state.get("up", (0.0, 0.0, 1.0))
        fwd_len = float(np.hypot(dir_vec[0], dir_vec[1])) or 1.0
        fwd_x = dir_vec[0] / fwd_len
        fwd_y = dir_vec[1] / fwd_len
        pitch = -(float(up_vec[0]) * fwd_x + float(up_vec[1]) * fwd_y)
        lat_x = -fwd_y
        lat_y = fwd_x
        roll = float(up_vec[0]) * lat_x + float(up_vec[1]) * lat_y
        return np.array(
            [np.clip(pitch, -1.0, 1.0), np.clip(roll, -1.0, 1.0)], dtype=np.float32
        )

    def _wheel_terrain_features(self) -> np.ndarray:
        """Return [left_terrain, right_terrain] in [-1, 1] from the RoadsSensor.

        +1 = well on road, 0 = at the edge, -1 = off road. Measured at the
        front-axle midpoint, so it is the honest left/right road-edge position
        (no per-wheel duplication). Falls back to neutral (0, 0) without a sensor.
        """
        roads = self.roads_sensor.poll() if self.roads_sensor is not None else {}
        if isinstance(roads, list):
            roads = roads[0] if roads else {}
        if not isinstance(roads, dict):
            roads = {}
        half_w = max(float(roads.get("halfWidth", 3.0)), 0.5)
        d_left = float(roads.get("dist2Left", self.HALF_TRACK_WIDTH))
        d_right = float(roads.get("dist2Right", self.HALF_TRACK_WIDTH))
        left = float(np.clip((d_left - self.HALF_TRACK_WIDTH) / half_w, -1.0, 1.0))
        right = float(np.clip((d_right - self.HALF_TRACK_WIDTH) / half_w, -1.0, 1.0))
        return np.array([left, right], dtype=np.float32)

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

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_beamng.py::TestExtraFeatures -v`
Expected: PASS (all 8 tests).

- [ ] **Step 6: Commit**

```bash
git add environments/beamng.py tests/test_beamng.py
git commit -m "feat: add body_orientation/wheel_terrain feature helpers to BeamNGDrivingEnv"
```

---

### Task 2: Wire extras into observation + RoadsSensor lifecycle; delete `BeamNGContinuousRollEnv`

**Files:**
- Modify: `environments/beamng.py` (`_observe` ~444-494; `_load_scenario` ~394-442; `close` ~329-343; `BeamNGCameraEnv._load_scenario` ~1161; `BeamNGCameraEnv._observe` ~1207; delete `BeamNGContinuousRollEnv` ~1025-1110)
- Test: `tests/test_beamng.py`

**Interfaces:**
- Consumes: `_extra_features`, `body_orientation`, `wheel_terrain`, `roads_sensor` (Task 1).
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
Expected: FAIL — `AttributeError: ... _attach_roads_sensor` and the deletion test fails (class still present).

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

In `BeamNGDrivingEnv._load_scenario`, at the top where `self._remove_lidar()` is called (~line 399), add the roads removal:

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

In `BeamNGDrivingEnv._observe`, the `state` dict is already fetched (~line 455). Change the final `np.concatenate([...])` (~line 476-492) to append the extras as a fourth block:

```python
        obs = np.concatenate(
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

### Task 3: Registry — forward flags, delete `beamng_continuous_roll`

**Files:**
- Modify: `environments/__init__.py` (all `_make_beamng*` factories; delete `_make_beamng_continuous_roll` ~102-119)
- Test: `tests/test_beamng.py` (registry assertion)

**Interfaces:**
- Consumes: `BeamNGDrivingEnv(..., body_orientation, wheel_terrain)` (Task 1); deleted class (Task 2).
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

### Task 4: Multi-agent env — slot flags, sizing, per-slot sensor, observation

**Files:**
- Modify: `environments/beamng_multi.py` (imports ~15; `VehicleSlot` ~33-91; `slot_n_states` ~138-141; `build_slots` ~163-194; `_create_slot_sensor` ~561-604; `observe` ~420-463; `close` ~701-717; add helpers + `HALF_TRACK_WIDTH`)
- Test: `tests/test_beamng_multi.py`

**Interfaces:**
- Consumes: same feature math as `BeamNGDrivingEnv` (Task 1), re-implemented per slot.
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

- [ ] **Step 3: Import RoadsSensor and add slot fields + class constant**

In `environments/beamng_multi.py`, add `RoadsSensor` to the import (~line 15) and the `except ImportError` fallback (~line 17):

```python
    from beamngpy.sensors import Camera, Damage, Electrics, Lidar, RoadsSensor
except ImportError:
    BeamNGpy = Scenario = Vehicle = Camera = Damage = Electrics = Lidar = RoadsSensor = None
```

Add fields to `VehicleSlot`. Under the "Environment profile" group (~line 50-54), add:

```python
    perception: str = "lidar"  # "lidar" | "lidar_grid" | "camera"
    trajectory_hints: int = 0
    body_orientation: bool = False
    wheel_terrain: bool = False
    n_states: int = 14  # observation length for this vehicle's env
```

Under the "Sensors" group (~line 58-62), add the roads sensor handle:

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

- [ ] **Step 5: Add the feature helpers and wire into `observe`**

Add helpers to `BeamNGMultiEnv` (place after `_waypoint_hints`, ~line 493):

```python
    def _body_orientation_features(self, state) -> np.ndarray:
        """[pitch, roll] in [-1, 1] from the vehicle up/forward vectors."""
        dir_vec = state.get("dir", (0.0, 1.0, 0.0))
        up_vec = state.get("up", (0.0, 0.0, 1.0))
        fwd_len = float(np.hypot(dir_vec[0], dir_vec[1])) or 1.0
        fwd_x = dir_vec[0] / fwd_len
        fwd_y = dir_vec[1] / fwd_len
        pitch = -(float(up_vec[0]) * fwd_x + float(up_vec[1]) * fwd_y)
        roll = float(up_vec[0]) * (-fwd_y) + float(up_vec[1]) * fwd_x
        return np.array(
            [np.clip(pitch, -1.0, 1.0), np.clip(roll, -1.0, 1.0)], dtype=np.float32
        )

    def _wheel_terrain_features(self, slot) -> np.ndarray:
        """[left, right] road-edge position in [-1, 1] from the slot's RoadsSensor."""
        roads = slot.roads_sensor.poll() if slot.roads_sensor is not None else {}
        if isinstance(roads, list):
            roads = roads[0] if roads else {}
        if not isinstance(roads, dict):
            roads = {}
        half_w = max(float(roads.get("halfWidth", 3.0)), 0.5)
        d_left = float(roads.get("dist2Left", self.HALF_TRACK_WIDTH))
        d_right = float(roads.get("dist2Right", self.HALF_TRACK_WIDTH))
        left = float(np.clip((d_left - self.HALF_TRACK_WIDTH) / half_w, -1.0, 1.0))
        right = float(np.clip((d_right - self.HALF_TRACK_WIDTH) / half_w, -1.0, 1.0))
        return np.array([left, right], dtype=np.float32)

    def _slot_extra_features(self, slot, state) -> np.ndarray:
        """Optional observation tail for a slot (body orientation / wheel terrain)."""
        blocks = []
        if slot.body_orientation:
            blocks.append(self._body_orientation_features(state))
        if slot.wheel_terrain:
            blocks.append(self._wheel_terrain_features(slot))
        if not blocks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(blocks)
```

In `observe` (~line 447), append the extras as the final block of the `np.concatenate`:

```python
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
                perception,
                waypoint_hints,
                self._slot_extra_features(slot, state),
            ]
        )
```

- [ ] **Step 6: Attach the per-slot RoadsSensor and tear it down**

In `_create_slot_sensor`, at the very end of the method (~line 604, after `self._cache_ego_local_bbox(slot)` for the LiDAR branch), attach the roads sensor for any slot that needs it. Because the camera branch returns early, add the attach for both paths — put it just before the `if slot.perception == "camera":` early-return branch by restructuring: attach roads first, then create the perception sensor. Insert at the top of `_create_slot_sensor` (~line 568):

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

### Task 5: CLI — prompts thread flags into single & multi sessions

**Files:**
- Modify: `core/cli.py` (add `_ask_bool` ~after `_ask_int` line 41; single-train ~147-164; single-eval/play ~248-265; `build_multi_session` ~415-432; `_multi_train_menu` spec builder ~459-471)
- Test: `tests/test_cli_multi.py`

**Interfaces:**
- Consumes: `slot_n_states(env_name, trajectory_hints, body_orientation, wheel_terrain)` (Task 4); registry factories forwarding flags (Task 3).
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
```

Add to `tests/test_cli_multi.py` a unit test for the helper:

```python
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
Expected: FAIL — `cannot import name '_ask_bool'`; and `slot_n_states` ignores the flags so n_states would be 14.

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

### Task 6: Regression sweep & docs

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
- Two independent flags → Tasks 1, 3, 4, 5. ✓
- Available to every BeamNG env (lidar/lidar_grid/camera/continuous, single + multi) → base class (Task 1/2), camera override (Task 2), registry (Task 3), multi (Task 4). ✓
- Honest 2-value wheel terrain → `_wheel_terrain_features` returns `(2,)` (Tasks 1, 4). ✓
- Append at end / order fixed → Tasks 2, 4. ✓
- `n_states` formula → Tasks 1, 4, 5. ✓
- RoadsSensor lifecycle + safe fallback → Tasks 1, 2, 4. ✓
- Delete `BeamNGContinuousRollEnv` + registration → Tasks 2, 3, 6. ✓
- CLI prompts (train, play, multi) → Task 5. ✓
- Tests for flag combinations + regression → Tasks 1, 4, 5, 6. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code. ✓

**Type consistency:** `_body_orientation_features(state)`, `_wheel_terrain_features()` (single) / `_wheel_terrain_features(slot)` (multi), `_extra_features(state)` (single) / `_slot_extra_features(slot, state)` (multi), `_attach_roads_sensor()`, `_remove_roads_sensor()`, `slot_n_states(env_name, trajectory_hints, body_orientation, wheel_terrain)`, `_ask_bool(prompt, default)` — names used consistently across tasks. ✓
