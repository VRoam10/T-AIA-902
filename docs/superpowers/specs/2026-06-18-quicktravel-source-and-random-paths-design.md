# Quick-Travel Source + Per-Episode Random Paths — Design

**Date**: 2026-06-18
**Status**: Approved (design phase)
**Scope**: Follow-up increment to the multi-path trajectories feature
(`2026-06-18-multi-path-trajectories-design.md`). Two changes: (1) source paths
from a map's **named quick-travel waypoints** instead of `SpawnSphere` spawn
points, and (2) add an opt-in to **randomize the path each training episode**,
for both single-agent and multi-agent training.

---

## Goal

- Generate one path per **quick-travel point** of a map. In BeamNG these are the
  named markers returned by `scenario.find_waypoints()` (e.g.
  `city_east_bridge1`), which match the in-game quick-travel list more closely
  than `SpawnSphere` spawn points.
- Let a training run optionally pick a **random path each episode** so an agent
  generalizes across the map's start points instead of overfitting one route.

## Non-goals

- Seed control / deterministic RNG for randomization (reproducibility comes from
  leaving the option off, which is the default).
- Changing the road-snapping / dedup / sort / fallback logic (unchanged).
- Pathfinding between quick-travel points.

## 1. Source → named quick-travel waypoints (`core/trajectory.py`)

`_teleport_points(bng)` is reordered and renamed in spirit to "quick-travel
points":

- Query `bng.scenario.find_waypoints()` **first** (named markers; each has
  `.name` and `.pos`, optional `.rot_quat`).
- Fall back to `bng.scenario.find_objects_class("SpawnSphere")`.
- Return `[]` on error/empty (then `generate()` falls back to longest-road, then
  square loop, as today).

Each returned entry carries its **name**: `list[tuple[Vec3, Quat, str]]`
(SpawnSphere objects without a name use their `oid`/`name` attr or `"spawn_<i>"`).
A point with no usable rotation defaults to identity `(0,0,0,1)`; the path
direction is then chosen by the nearest-road tangent (existing
`_road_path_from_teleport` behavior).

`generate()` logs one line after collecting points:
`[trajectory] <map>: <N> quick-travel points: <name1>, <name2>, ...`
so the user can confirm they match the in-game quick-travel list. Consumers
(`_path_from_teleport`, the dedup/sort loop) ignore the name except for logging.

## 2. Single-agent per-episode random path (`environments/beamng.py`)

- New constructor flag `random_path: bool = False`.
- The env stores **all** paths: `self._paths: list[TrajectoryData]`, set in
  `_resolve_trajectory` from `load_or_generate(...).paths`. `self.trajectory`
  defaults to `self._paths[0]` (unchanged single-agent behavior when the flag is
  off).
- On `reset()`, when `random_path` is on:
  - `self.trajectory = random.choice(self._paths)`,
  - `self.waypoints = self._select_waypoints()`,
  - teleport the vehicle to `self.trajectory.spawn_pos` / `spawn_rot`
    (`reset=True`) instead of relying on `scenario.restart()`'s baked spawn,
  - reset the active marker to waypoint 0.
  When off: `reset()` is byte-for-byte the current flow (`restart()` + marker).
- `_load_scenario` adds the **union of all paths'** waypoints as checkpoints when
  `random_path` is on (so a marker exists wherever the agent spawns); otherwise
  it adds only `self.waypoints`, as today.

## 3. Multi-agent per-episode random path (`environments/beamng_multi.py`)

- New `random_path: bool = False` flag on `BeamNGMultiEnv`.
- `VehicleSlot` gains `path_idx: int = 0` (index into `self.trajectories.paths`).
- `_assign_paths()`:
  - off → deal `paths[i]` to slot `i` (current behavior); set `slot.path_idx = i`.
  - on → deal a random **distinct** subset: shuffle path indices, assign the
    first `len(slots)` to the slots. Records `slot.path_idx`.
  - the `len(slots) > len(paths)` hard `ValueError` is unchanged.
- `reset_vehicle(slot)` (one vehicle finishing mid-session): when `random_path`
  is on, pick a new `path_idx` uniformly at random from
  `set(range(len(paths))) - {other active slots' path_idx}` (always non-empty
  since vehicles ≤ paths), reassign the slot's `waypoints`/`spawn_pos`/`spawn_rot`
  from that path, then teleport as today. Distinctness — and therefore the
  no-collision guarantee — holds continuously.
- `_load_scenario` checkpoints = union of **all** `self.trajectories.paths`
  waypoints when `random_path` is on (every possible target visible); union of
  assigned slots' waypoints otherwise (current behavior).

A helper `_pick_distinct_path_idx(slot)` centralizes the "random index not held
by other active slots" logic so `_assign_paths` and `reset_vehicle` share it.

## 4. CLI (`core/cli.py`)

- `_pick_beamng_options()` (single-agent setup) and `_multi_train_menu()`
  (multi-agent setup) prompt: `Randomize path each episode? [y/N]` (default No),
  via the existing `_ask_bool` helper.
- Single: thread `random_path` into the `BeamNGDrivingEnv` constructor.
- Multi: carry `random_path` on the session and pass it to `BeamNGMultiEnv`
  (it is a per-session env flag, not per-vehicle).

## 5. Determinism

Randomization uses the `random` module. Default-off preserves current
reproducible runs. No seed parameter (out of scope).

## Testing

- `core/trajectory.py`:
  - `_teleport_points` queries `find_waypoints()` first; uses `SpawnSphere` only
    when waypoints are empty.
  - returned entries carry names; `generate()` logs the discovered names.
  - a waypoint with no rotation still yields a valid path (identity heading →
    nearest-road tangent picks direction).
- `environments/beamng.py`:
  - `random_path=True` selects via `random.choice` over all paths on reset
    (patch `random.choice`, assert the chosen spawn is used);
  - `random_path=False` leaves `reset()` behavior unchanged (still `paths[0]`).
- `environments/beamng_multi.py`:
  - `_assign_paths(random_path=True)` gives every slot a distinct `path_idx`;
  - `_pick_distinct_path_idx` never returns an index held by another active slot;
  - `random_path=False` keeps the `paths[i] → slot i` assignment.

All tests are BeamNG-free (mock `bng`, patch `random`).

## File touch list

| Action | Path |
|---|---|
| Modify | `core/trajectory.py` (waypoints-first source, names, logging) |
| Modify | `environments/beamng.py` (random_path: all paths, reset re-pick) |
| Modify | `environments/beamng_multi.py` (random_path: distinct deal + reset) |
| Modify | `core/cli.py` (randomize prompt, thread the flag) |
| Modify | `tests/test_trajectory.py`, `tests/test_beamng_multi.py`; add single-env random test |
| Create | this design doc |
