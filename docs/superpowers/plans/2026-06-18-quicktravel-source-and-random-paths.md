# Quick-Travel Source + Per-Episode Random Paths Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Source paths from a map's named quick-travel waypoints (with name logging) and add an opt-in to randomize the path each training episode, for both single-agent and multi-agent training.

**Architecture:** `core/trajectory.py` queries `find_waypoints()` first (falling back to `SpawnSphere`) and logs discovered names. The single-agent env keeps all paths and re-picks one per `reset()` when `random_path` is on. The multi-agent env deals a random *distinct* path per vehicle each episode, preserving the no-collision guarantee. CLI prompts thread a `random_path` flag.

**Tech Stack:** Python 3, `beamngpy`, `numpy`, `pytest`. Pure logic unit-tested with mocked BeamNG and patched `random`.

## Global Constraints

- Lint with ruff; run tests with `python -m pytest` from repo root (`c:\Epitech\T-AIA-902`).
- BeamNG-free tests: mock `bng` with MagicMock; patch `random.choice`/`random.shuffle`/`random.randrange` where randomness is exercised. Never launch the simulator.
- `random_path` defaults to **False** everywhere; when False, behavior must be byte-for-byte the current behavior. No RNG seed parameter.
- `_teleport_points` returns `list[tuple[Vec3, Quat, str]]` (pos, rot, name).
- Multi-agent must keep distinct paths per vehicle at all times (vehicles ≤ paths still hard-errors).

---

### Task 1: Quick-travel source — `find_waypoints()` first, names, logging

**Files:**
- Modify: `core/trajectory.py`
- Test: `tests/test_trajectory.py`

**Interfaces:**
- Produces: `_teleport_points(bng) -> list[tuple[Vec3, Quat, str]]` — queries `find_waypoints()` first, then `find_objects_class("SpawnSphere")`, then `[]`; each entry is `(pos, rot_quat, name)`.
- `generate()` consumes the 3-tuples (uses pos+rot to build paths, names for a log line).

- [ ] **Step 1: Write/replace the failing tests**

In `tests/test_trajectory.py`, replace `test_teleport_points_reads_spawnspheres` and `test_teleport_points_empty_on_error` and add the ordering/name tests. Update `_spawn_obj` to set `.name`:

```python
def _spawn_obj(pos, rot=(0.0, 0.0, 0.0, 1.0), name="wp"):
    obj = MagicMock()
    obj.pos = pos
    obj.rot_quat = rot
    obj.name = name
    return obj


def test_teleport_points_prefers_waypoints():
    bng = MagicMock()
    bng.scenario.find_waypoints.return_value = [
        _spawn_obj((1.0, 2.0, 3.0), name="garage"),
        _spawn_obj((4.0, 5.0, 6.0), name="quarry"),
    ]
    pts = _teleport_points(bng)
    bng.scenario.find_waypoints.assert_called_once_with()
    bng.scenario.find_objects_class.assert_not_called()
    assert pts[0][0] == (1.0, 2.0, 3.0)
    assert pts[0][2] == "garage"
    assert len(pts) == 2


def test_teleport_points_falls_back_to_spawnspheres():
    bng = MagicMock()
    bng.scenario.find_waypoints.return_value = []
    bng.scenario.find_objects_class.return_value = [_spawn_obj((7.0, 8.0, 9.0), name="ss0")]
    pts = _teleport_points(bng)
    bng.scenario.find_objects_class.assert_called_once_with("SpawnSphere")
    assert pts[0][0] == (7.0, 8.0, 9.0)


def test_teleport_points_empty_on_error():
    bng = MagicMock()
    bng.scenario.find_waypoints.side_effect = RuntimeError("boom")
    bng.scenario.find_objects_class.side_effect = RuntimeError("boom")
    assert _teleport_points(bng) == []


def test_generate_logs_quicktravel_names(capsys):
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_waypoints.return_value = [
        _spawn_obj((0.0, 0.0, 0.0), name="north_wp"),
        _spawn_obj((201.0, 0.0, 0.0), name="east_wp"),
    ]
    generate(bng, map_name="italy")
    out = capsys.readouterr().out
    assert "north_wp" in out and "east_wp" in out
```

Existing `test_generate_builds_one_path_per_teleport` and `test_generate_dedupes_nearby_teleports` use `bng.scenario.find_objects_class.return_value = [...]`. Update those two to set `bng.scenario.find_waypoints.return_value = [...]` instead (with `_spawn_obj(...)` now carrying names), since waypoints are queried first. Keep their assertions otherwise unchanged.

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest tests/test_trajectory.py -k "teleport_points or quicktravel_names or builds_one_path or dedupes_nearby" -v`
Expected: FAIL (find_waypoints not queried first; entries are 2-tuples; no name logging).

- [ ] **Step 3: Implement**

In `core/trajectory.py`, replace `_teleport_points` with:

```python
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
```

In `generate()`, the teleport loop currently unpacks `for pos, rot in teleports:`. Change it to capture names and log them:

```python
    teleports = _teleport_points(bng)
    if teleports:
        print(
            f"[trajectory] {map_name}: {len(teleports)} quick-travel points: "
            + ", ".join(t[2] for t in teleports)
        )

    if roads and teleports:
        scored: list[tuple[TrajectoryData, float]] = []
        accepted_spawns: list[Vec3] = []
        for pos, rot, _name in teleports:
            built = _path_from_teleport(pos, rot, roads, map_name)
            ...
```
(Only the unpacking and the added log line change; the dedup/sort body is unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_trajectory.py -v`
Expected: PASS (all trajectory tests).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check core/trajectory.py tests/test_trajectory.py
git add core/trajectory.py tests/test_trajectory.py
git commit -m "feat: source paths from named quick-travel waypoints with logging"
```

---

### Task 2: Single-agent per-episode random path

**Files:**
- Modify: `environments/beamng.py`
- Test: `tests/test_trajectory.py` (env-level test, no simulator)

**Interfaces:**
- `BeamNGDrivingEnv.__init__(..., random_path: bool = False)`; subclasses `BeamNGContinuousEnv` and `BeamNGCameraEnv` thread it through.
- `self._paths: list[TrajectoryData]` holds all paths; `self.trajectory = self._paths[0]` by default.
- Produces: `_pick_episode_path()` — when `random_path`, sets `self.trajectory = random.choice(self._paths)` and refreshes `self.waypoints`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trajectory.py`:

```python
def test_single_env_random_path_picks_from_all_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    p0 = _sample_traj(source="teleport:first")
    p1 = TrajectoryData(
        spawn_pos=(99.0, 99.0, 1.0),
        spawn_rot=(0.0, 0.0, 0.0, 1.0),
        sparse_waypoints=[(99.0, 100.0, 0.0), (99.0, 110.0, 0.0)],
        dense_waypoints=[(99.0, 100.0, 0.0)],
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        source="teleport:second",
    )
    MapTrajectories(
        map_name="italy", generated_at="2026-06-18T12:00:00+00:00", paths=[p0, p1]
    )  # construct to validate shape
    (tmp_path / "italy.json").write_text(
        MapTrajectories(
            map_name="italy", generated_at="2026-06-18T12:00:00+00:00", paths=[p0, p1]
        ).to_json()
    )

    from environments.beamng import BeamNGDrivingEnv

    env = BeamNGDrivingEnv(beamng_home="unused", map_name="italy", random_path=True)
    env._resolve_trajectory()  # populates env._paths and default env.trajectory
    env._paths = [p0, p1]

    monkeypatch.setattr("environments.beamng.random.choice", lambda seq: seq[1])
    env._pick_episode_path()
    assert env.trajectory.source == "teleport:second"
    assert env.waypoints == list(p1.sparse_waypoints)


def test_single_env_resolve_trajectory_default_first_path_when_not_random(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    p0 = _sample_traj(source="teleport:first")
    (tmp_path / "italy.json").write_text(
        MapTrajectories(
            map_name="italy", generated_at="2026-06-18T12:00:00+00:00", paths=[p0]
        ).to_json()
    )
    from environments.beamng import BeamNGDrivingEnv

    env = BeamNGDrivingEnv(beamng_home="unused", map_name="italy")  # random_path defaults False
    traj = env._resolve_trajectory()
    assert traj.source == "teleport:first"
    assert env._paths[0].source == "teleport:first"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_trajectory.py -k "single_env_random_path or default_first_path_when_not_random" -v`
Expected: FAIL — `random_path` kwarg unknown / `_paths` / `_pick_episode_path` missing.

- [ ] **Step 3: Implement**

In `environments/beamng.py`:

Add `import random` at the top if not already imported (check the existing import block; `random` is already used by `_randomize_waypoints` per a commented call — confirm and only add if missing).

Add `random_path: bool = False` to `BeamNGDrivingEnv.__init__` signature (after `wheel_terrain`) and store it:

```python
        self.wheel_terrain = wheel_terrain
        self.random_path = random_path
```
Add the paths holder next to the trajectory attribute:

```python
        self.trajectory: TrajectoryData | None = None
        self._paths: list[TrajectoryData] = []
        self.waypoints: list[tuple[float, float, float]] = []
```

Change `_resolve_trajectory` to populate `self._paths` and default to `paths[0]`. Both return sites currently end `.paths[0]`; rewrite the method body so it stores all paths:

```python
    def _resolve_trajectory(self) -> TrajectoryData:
        """Load cached trajectories (all paths); default to the longest road."""
        from core.trajectory import CACHE_DIR

        cache_path = CACHE_DIR / f"{self.map_name}.json"
        if cache_path.exists():
            self._paths = load_or_generate(self.map_name, bng=None).paths
        else:
            probe = Scenario(self.map_name, "trajectory_probe", description="Road probe")
            probe_vehicle = Vehicle("probe_vehicle", model="etk800")
            probe.add_vehicle(probe_vehicle, pos=(0.0, 0.0, 100.0), rot_quat=(0.0, 0.0, 0.0, 1.0))
            probe.make(self.bng)
            self.bng.load_scenario(probe)
            self.bng.start_scenario()
            time.sleep(0.5)
            self._paths = load_or_generate(self.map_name, self.bng).paths
        self.trajectory = self._paths[0]
        return self.trajectory
```
(Preserve the existing probe lines exactly as they already are; only the assignment to `self._paths` + `self.trajectory` changes. If `time`/`Scenario`/`Vehicle` were imported locally before, keep that.)

Add the per-episode picker:

```python
    def _pick_episode_path(self) -> None:
        """When random_path is on, choose a random path for the next episode."""
        if not self.random_path or not self._paths:
            return
        self.trajectory = random.choice(self._paths)
        self.waypoints = self._select_waypoints()
```

Wire it into `reset()`. In the `else` branch (bng already up), when `random_path` is on, pick a new path and teleport to its spawn instead of relying on `restart()`'s baked spawn:

```python
        if self.bng is None:
            self._launch()
        else:
            if self.random_path:
                self._pick_episode_path()
                self.bng.scenario.restart()
                self.vehicle.teleport(
                    self.trajectory.spawn_pos,
                    rot_quat=self.trajectory.spawn_rot,
                    reset=True,
                )
            else:
                self.bng.scenario.restart()
            self._update_active_marker(0)
            ... (keep the existing LiDAR test block unchanged)
```

Make `_load_scenario` add the union of all paths' checkpoints when randomizing. Find the checkpoint block in the base `_load_scenario`:

```python
        checkpoint_wps = (
            [wp for p in self._paths for wp in p.sparse_waypoints]
            if self.random_path
            else self.waypoints
        )
        scales = [(5.0, 5.0, 1.0)] * len(checkpoint_wps)
        self.scenario.add_checkpoints(checkpoint_wps, scales)
```

Thread `random_path` through the two subclasses. In `BeamNGContinuousEnv.__init__`, add `random_path: bool = False` to the signature and pass `random_path=random_path` in its `super().__init__(...)`. Do the same in `BeamNGCameraEnv.__init__`. In `BeamNGCameraEnv._load_scenario`, apply the same `checkpoint_wps` union block shown above (it has its own copy of the add_checkpoints call).

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_trajectory.py -v`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check environments/beamng.py tests/test_trajectory.py
git add environments/beamng.py tests/test_trajectory.py
git commit -m "feat: single-agent random path per episode"
```

---

### Task 3: Multi-agent per-episode random path (distinct deal)

**Files:**
- Modify: `environments/beamng_multi.py`
- Test: `tests/test_beamng_multi.py`

**Interfaces:**
- `BeamNGMultiEnv.__init__(..., random_path: bool = False)`.
- `VehicleSlot.path_idx: int = 0`.
- `_assign_paths()` — off: `paths[i] -> slot i`; on: distinct random deal.
- `_pick_distinct_path_idx(slot) -> int` — random index not held by other slots.
- `reset_vehicle(slot)` re-picks a distinct random path when `random_path`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_beamng_multi.py` (the `TestPathAssignment` class already builds `_mt(n)`):

```python
    def test_random_assign_gives_distinct_paths(self, monkeypatch):
        env = _env()  # 3 slots
        env.random_path = True
        env.trajectories = self._mt(5)
        monkeypatch.setattr(
            "environments.beamng_multi.random.shuffle",
            lambda seq: seq.reverse(),
        )
        env._assign_paths()
        idxs = [s.path_idx for s in env.slots]
        assert len(set(idxs)) == 3  # distinct
        # reversed [0,1,2,3,4] -> [4,3,2,1,0]; first 3 dealt
        assert idxs == [4, 3, 2]

    def test_pick_distinct_path_idx_avoids_other_slots(self):
        env = _env()  # 3 slots
        env.random_path = True
        env.trajectories = self._mt(3)
        env.slots[1].path_idx = 1
        env.slots[2].path_idx = 2
        # only index 0 is free for slot 0
        assert env._pick_distinct_path_idx(env.slots[0]) == 0

    def test_assign_paths_not_random_is_sequential(self):
        env = _env()
        env.trajectories = self._mt(3)
        env._assign_paths()
        assert [s.path_idx for s in env.slots] == [0, 1, 2]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_beamng_multi.py::TestPathAssignment -v`
Expected: FAIL — `random_path`/`path_idx`/`_pick_distinct_path_idx` missing.

- [ ] **Step 3: Implement**

In `environments/beamng_multi.py`:

Add `import random` at the top (check it is not already imported).

Add `path_idx: int = 0` to `VehicleSlot` (near `waypoints`).

Add `random_path` to the constructor signature (after `map_name`) and store `self.random_path = random_path`.

Replace `_assign_paths` and add the picker:

```python
    def _assign_paths(self):
        """Give each vehicle its own path; error if vehicles outnumber paths."""
        paths = self.trajectories.paths
        if len(self.slots) > len(paths):
            raise ValueError(
                f"{len(self.slots)} vehicles requested but map '{self.map_name}' has "
                f"only {len(paths)} distinct path(s). Reduce the vehicle count to "
                f"<= {len(paths)} or pick a map with more quick-travel points."
            )
        if self.random_path:
            order = list(range(len(paths)))
            random.shuffle(order)
            for slot, idx in zip(self.slots, order, strict=False):
                slot.path_idx = idx
                self._apply_path(slot, paths[idx])
        else:
            for i, slot in enumerate(self.slots):
                slot.path_idx = i
                self._apply_path(slot, paths[i])

    def _apply_path(self, slot, path):
        slot.waypoints = list(path.sparse_waypoints)
        slot.spawn_pos = path.spawn_pos
        slot.spawn_rot = path.spawn_rot

    def _pick_distinct_path_idx(self, slot) -> int:
        """A random path index not currently held by any other slot."""
        taken = {s.path_idx for s in self.slots if s is not slot}
        free = [i for i in range(len(self.trajectories.paths)) if i not in taken]
        return random.choice(free)
```

In `reset_vehicle`, re-pick a distinct path when randomizing, before teleporting:

```python
    def reset_vehicle(self, slot: VehicleSlot):
        """Teleport one finished vehicle to its (possibly new) path for the next episode."""
        if self.random_path and self.trajectories is not None:
            slot.path_idx = self._pick_distinct_path_idx(slot)
            self._apply_path(slot, self.trajectories.paths[slot.path_idx])
        slot.vehicle.teleport(slot.spawn_pos, rot_quat=slot.spawn_rot, reset=True)
        slot.reset_episode()
        if slot.lidar is not None or slot.electrics is not None:
            slot.last_obs = self.observe(slot)
            slot.last_dist = slot.current_dist
        self._update_slot_marker(slot)
```

In `_load_scenario`, when randomizing, add the union of ALL paths' waypoints as checkpoints (not just assigned slots'):

```python
        if self.random_path:
            all_waypoints = [wp for p in self.trajectories.paths for wp in p.sparse_waypoints]
        else:
            all_waypoints = [wp for slot in self.slots for wp in slot.waypoints]
        scales = [(5.0, 5.0, 1.0)] * len(all_waypoints)
        self.scenario.add_checkpoints(all_waypoints, scales)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_beamng_multi.py -v`
Expected: PASS (new TestPathAssignment cases + all existing).

- [ ] **Step 5: Lint + commit**

```bash
python -m ruff check environments/beamng_multi.py tests/test_beamng_multi.py
git add environments/beamng_multi.py tests/test_beamng_multi.py
git commit -m "feat: multi-agent random path per episode with distinct deal"
```

---

### Task 4: CLI — randomize-path prompt threaded into both training modes

**Files:**
- Modify: `core/cli.py`

**Interfaces:**
- Consumes: `BeamNGDrivingEnv(..., random_path=...)`, `BeamNGMultiEnv(..., random_path=...)`.

- [ ] **Step 1: Single-agent prompt + wiring**

In `core/cli.py`, in the single-agent training setup (the block around the `beamng_kwargs` dict), after `wheel_terrain = _ask_bool(...)` add:

```python
        random_path = _ask_bool("Randomize path each episode?")
        beamng_kwargs = {
            "map_name": map_name,
            "vehicle_id": vehicle_id,
            "trajectory_hints": trajectory_hints,
            "body_orientation": body_orientation,
            "wheel_terrain": wheel_terrain,
            "random_path": random_path,
        }
```

- [ ] **Step 2: Multi-agent prompt + wiring**

In `_multi_train_menu()`, after the map is picked (`map_name = _pick(_BEAMNG_MAPS, "Map")`) add:

```python
    random_path = _ask_bool("Randomize path each episode (deals distinct paths per vehicle)?")
```

Change `build_multi_session(specs, map_name)` to accept and forward `random_path`:

```python
def build_multi_session(specs: list[dict], map_name: str, random_path: bool = False):
    ...
    env = BeamNGMultiEnv(
        slots=[],
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        map_name=map_name,
        random_path=random_path,
    )
```

And update its call site in `_multi_train_menu`: `env, slots = build_multi_session(specs, map_name, random_path)`.

- [ ] **Step 3: Verify import + smoke**

Run: `python -c "import core.cli"` (confirms no syntax/import error).
Expected: no output, exit 0. (If it fails on a heavy dependency unrelated to this change, report it; do not stub.)

- [ ] **Step 4: Commit**

```bash
git add core/cli.py
git commit -m "feat: CLI prompt to randomize path each episode (single + multi)"
```

---

### Task 5: Full suite + verification

**Files:** none (verification only).

- [ ] **Step 1: Full suite**

Run: `python -m pytest -q`
Expected: all pass (no regressions). If any fail, report which.

- [ ] **Step 2: Lint touched files**

Run: `python -m ruff check core/trajectory.py environments/beamng.py environments/beamng_multi.py core/cli.py tests/`
Expected: clean.

---

## Self-Review

**Spec coverage:**
- Quick-travel source (`find_waypoints` first) + names + logging → Task 1. ✓
- Single-agent `random_path` (all paths, reset re-pick + teleport, checkpoint union, subclass threading) → Task 2. ✓
- Multi-agent `random_path` (distinct deal, `_pick_distinct_path_idx`, reset re-pick, checkpoint union) → Task 3. ✓
- CLI prompts + threading (single + multi) → Task 4. ✓
- Determinism: default-off, no seed → reflected in every default and the constraints. ✓
- Testing (mocked bng, patched random) → Tasks 1-3. ✓

**Placeholder scan:** none.

**Type consistency:** `_teleport_points` 3-tuple is consumed in `generate`'s loop (Task 1). `random_path` bool flows constructor→reset/_assign_paths consistently. `_apply_path`/`_pick_distinct_path_idx`/`_pick_episode_path` names match across tasks. `VehicleSlot.path_idx` added Task 3, used in `_assign_paths`/`reset_vehicle`/`_pick_distinct_path_idx`.
