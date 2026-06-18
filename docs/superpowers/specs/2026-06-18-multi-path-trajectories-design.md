# Multi-Path Trajectories from Teleport Points — Design

**Date**: 2026-06-18
**Status**: Approved (design phase)
**Scope**: Extend `core/trajectory.py` to generate **one path per map teleport point** instead of a single per-map path, and rework `environments/beamng_multi.py` so each vehicle trains on its **own** path in its **own** location. Builds on the per-map auto-trajectory system (`2026-05-23-auto-trajectory-generation-design.md`).

---

## Goal

Today `outputs/trajectories/{map}.json` stores **one** trajectory per map (the single longest drivable road). The multi-agent env spawns every vehicle abreast on that one start line, all chasing the same waypoints — causing the vehicle-collision and shared-LiDAR interference noted in `docs/romain.md` issue 6.

New behavior: enumerate a map's **teleport points** (BeamNG `SpawnSphere` objects), build **one path per teleport point** by snapping it to the nearest road, and store all of them in the map file. In multi-agent training each vehicle gets its own path in its own location, so:

- agents never collide (they are physically far apart), and
- a single session trains agents across several different parts of the map at once.

## Non-goals

- Multi-route graph pathfinding between teleport points (still out of scope).
- Random per-episode path shuffling (breaks RL reproducibility).
- Changing reward functions or agent code.
- Changing single-agent training behavior (it must stay as-is).

## Source of the paths

`bng.scenario.find_objects_class("SpawnSphere")` returns the map's predefined teleport/spawn points, each carrying a world `pos` and `rot_quat`. If a map exposes none, fall back to `bng.scenario.find_waypoints()`; if still none, fall back to the longest-road single path (current behavior), and finally to the square loop.

## Path generation (`core/trajectory.py`)

For each teleport point, during the probe phase (after the map is loaded):

1. **Nearest road** — from `get_road_network(include_edges=True, drivable_only=True)`, compute each road's centerline (reusing `_edge_center`) and pick the road whose closest centerline point is nearest the teleport `pos` (XY distance).
2. **Orient to heading** — the road centerline is an ordered polyline; reverse it if following it from the snap point would run *against* the teleport heading, so the car always drives forward along its path. Heading is taken from the teleport's `rot_quat`.
3. **Resample** — sparse (25 m) and dense (8 m), via the existing `resample()`.
4. **Spawn pose** — use the teleport point's own `pos` (z + `SPAWN_Z_OFFSET_M`) and `rot_quat`.

**Dedup**: skip a teleport point whose spawn is within `MIN_PATH_SEPARATION_M` (default 30 m) of an already-accepted path's spawn, so two near-identical paths don't both make the list.

**Ordering**: paths are sorted by road length descending, so `paths[0]` is the longest road. This preserves single-agent behavior, which takes `paths[0]`.

**Fallbacks** (always yield ≥1 path):
- No teleport points found → one path from the longest road (current `generate()` logic).
- No usable roads → one square-loop path (current `_square_loop_fallback`).

## Data structure & cache

`TrajectoryData` is unchanged — it is now the per-path unit. A new wrapper holds all paths for a map:

```python
@dataclass(frozen=True)
class MapTrajectories:
    map_name: str
    generated_at: str           # ISO 8601
    paths: list[TrajectoryData] # one per teleport point, longest road first

    def to_json(self) -> str: ...
    @classmethod
    def from_json(cls, payload: str) -> "MapTrajectories": ...
```

Serialized to `outputs/trajectories/{map}.json` as
`{"map_name": ..., "generated_at": ..., "paths": [ <TrajectoryData>, ... ]}`.

`from_json` also accepts the **old single-object format** (a dict with top-level
`spawn_pos`): it wraps that as a 1-element `paths` list, so existing caches keep
loading without regeneration. The cache stays gitignored/regenerable.

`load_or_generate(map_name, bng)` now returns `MapTrajectories`. `generate(bng, map_name)` returns `MapTrajectories`.

## Multi-agent env integration (`environments/beamng_multi.py`)

- `_resolve_trajectory()` returns `MapTrajectories`.
- **Cap = hard error**: if `len(slots) > len(paths)`, raise a `ValueError` with a clear message (how many vehicles requested, how many paths the map has, suggestion to add vehicles back / pick a richer map). No silent capping.
- Assign path `i` to slot `i`: set `slot.waypoints` (sparse) and `slot.spawn_pos`/`slot.spawn_rot` from `paths[i]`.
- Replace the shared `self.waypoints` with **per-slot `slot.waypoints`**. Touches: `_path_errors`, `observe` (checkpoint_dist target), `_waypoint_hints`, `_update_slot_marker`, and both reward fns (`len(self.waypoints)` → `len(slot.waypoints)`).
- Remove the abreast starting-grid logic: `_grid_pose`, `_spawn_axes`, and `GRID_LANE_OFFSET` go away; each slot spawns at its own path's pose.
- `add_checkpoints` adds the **union** of every slot's waypoints (for in-game visualization). Each slot's colored marker still tracks only its own current target.

### VehicleSlot

Add `waypoints: list[tuple[float, float, float]] = field(default_factory=list)`.
`spawn_pos`/`spawn_rot` already exist on the slot; they are now sourced from the
slot's path rather than computed by `_grid_pose`.

## Single-agent env integration (`environments/beamng.py`)

`_resolve_trajectory()` calls the multi-path loader and takes `paths[0]`
(longest road). Because paths are sorted length-descending, single-agent spawn
pose and waypoints are unchanged from today.

## CLI (`core/cli.py`)

- "Generate trajectories" produces all paths per map (no flow change beyond what
  `generate()` now returns).
- Multi-agent session setup surfaces how many paths the chosen map has, so the
  user can size the vehicle count before the hard-error cap triggers.

## Constants

```python
MIN_PATH_SEPARATION_M = 30.0   # dedup radius between accepted path spawns
# Reuse existing: SPARSE_SPACING_M, DENSE_SPACING_M, SPAWN_Z_OFFSET_M, fallbacks
```

## Error handling

- Vehicles > paths → `ValueError` (hard error, see above).
- `find_objects_class` unavailable / raises → fall back to longest-road single path.
- Malformed road network → existing fallback chain.
- Corrupt cache → existing log-delete-regenerate path.

## Testing

Pure-function unit tests (no BeamNG; probe data mocked, consistent with existing
tests):

- `tests/test_trajectory.py`:
  - nearest-road snapping picks the closest road to a given point;
  - heading-based orientation reverses the polyline when it runs backward;
  - min-separation dedup drops a near-duplicate teleport point;
  - `MapTrajectories` round-trips through JSON;
  - `from_json` accepts the old single-object format (1-element `paths`);
  - paths come out sorted longest-road-first.
- `tests/test_beamng_multi.py`:
  - each slot gets its own path's waypoints + spawn pose;
  - `len(slots) > len(paths)` raises `ValueError`.

## File touch list

| Action | Path |
|---|---|
| Modify | `core/trajectory.py` (MapTrajectories, teleport enum, nearest-road snap, orient, dedup) |
| Modify | `environments/beamng_multi.py` (per-slot paths, hard-error cap, marker union, drop grid) |
| Modify | `environments/beamng.py` (paths[0]) |
| Modify | `core/cli.py` (path-count messaging) |
| Modify | `tests/test_trajectory.py`, `tests/test_beamng_multi.py` |
| Create | `docs/superpowers/specs/2026-06-18-multi-path-trajectories-design.md` (this file) |
