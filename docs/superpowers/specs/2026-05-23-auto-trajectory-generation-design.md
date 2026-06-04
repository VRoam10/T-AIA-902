# Auto-Trajectory Generation per Map — Design

**Date**: 2026-05-23
**Status**: Approved (design phase)
**Scope**: Replace hardcoded `DEFAULT_WAYPOINTS` / `DDPG_WAYPOINTS` / `SPAWN_POS` / `SPAWN_ROT` in `environments/beamng.py` with a per-map automatic generation system backed by BeamNGpy's road-network API.

---

## Goal

Given any BeamNG map name in `AVAILABLE_MAPS` (`gridmap_v2`, `italy`, `west_coast_usa`, `smallgrid`), produce a deterministic trajectory (spawn pose + sparse waypoints + dense waypoints) without manual `F8`-console intervention, and reuse it across training runs for reproducibility.

## Non-goals

- Multi-route pathfinding through a road graph (Phase 2 if needed).
- Random / per-episode trajectory shuffling (rejected — breaks RL reproducibility).
- Human-driven trajectory recording (already covered by the existing `scenario_creator.md` workflow; this spec automates it away).
- Changing the reward functions or RL agent code.

## Source of the trajectory

`beamngpy.Scenario.get_road_network(include_edges=True, drivable_only=True)` returns a `StrDict` mapping each `DecalRoad` ID to its metadata and edge points. We pick the **longest drivable road** in the map and resample its centerline at two densities.

For maps where the API returns nothing (`smallgrid` — empty grid level), a **geometric fallback** generates an 80 m square loop centered on the world origin.

## Data structure

```python
# core/trajectory.py
@dataclass
class TrajectoryData:
    spawn_pos: tuple[float, float, float]            # (x, y, z) — world coords
    spawn_rot: tuple[float, float, float, float]     # quaternion (x, y, z, w)
    sparse_waypoints: list[tuple[float, float, float]]  # ~25 m spacing — discrete algos
    dense_waypoints:  list[tuple[float, float, float]]  # ~8 m spacing  — continuous algos
    map_name: str
    generated_at: str    # ISO 8601 timestamp
    source: str          # "road_network:<road_id>" or "fallback:square_loop"
```

Serialized to `outputs/trajectories/{map_name}.json`.

## Algorithm

### 1. Road selection

```
roads = bng.scenario.get_road_network(include_edges=True, drivable_only=True)
for road_id, road in roads.items():
    edges = road.get("edges", [])
    if len(edges) < 2: skip
    centerline = [edge_center(e) for e in edges]      # list of (x, y, z)
    length = sum(euclid(centerline[i], centerline[i+1]) for i in range(len(centerline)-1))
pick road with max length
```

`edge_center(e)` extracts the centerline point from a single edge dict. The exact key is version-dependent in BeamNGpy — typical names seen are `"middle"`, or the midpoint of `"left"` and `"right"`. The implementation must probe the actual keys at runtime (one `print(next(iter(roads.values()))["edges"][0])` is enough) and pick whichever is present, with `midpoint(left, right)` as a safe fallback.

If `roads` is empty or no road has ≥ 2 edges with non-zero length → geometric fallback.

### 2. Resampling

Standard arc-length resampling at fixed spacing:

```
def resample(path, spacing):
    # Build cumulative distances along the polyline.
    # For each target distance k * spacing, linearly interpolate between the
    # two surrounding original points.
    # Always include the first and last original points.
```

Two passes: `spacing = 25.0` → sparse, `spacing = 8.0` → dense.

### 3. Spawn pose

```
spawn_pos = (wp[0].x, wp[0].y, wp[0].z + 1.0)        # 1 m above road surface
heading   = atan2(wp[1].y - wp[0].y, wp[1].x - wp[0].x)
spawn_rot = (0.0, 0.0, sin(heading/2), cos(heading/2))  # rotation around Z
```

### 4. Fallback (empty maps)

```
center = (0.0, 0.0, 1.0)
side   = 80.0
corners = [(±side/2, ±side/2, 1.0)]   # CCW from (+x, -y)
# Resample the closed polyline at the two densities.
```

## Cache layer

`core/trajectory.py` exposes:

```python
def load_or_generate(map_name: str, bng: BeamNGpy | None) -> TrajectoryData:
    cache_path = Path("outputs/trajectories") / f"{map_name}.json"
    if cache_path.exists():
        return TrajectoryData.from_json(cache_path.read_text())
    if bng is None:
        raise RuntimeError(f"No cached trajectory for {map_name} and no BeamNG instance to generate one")
    data = generate(bng, map_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(data.to_json())
    return data
```

The cache is **gitignored** (it's an output, regenerable). Already covered by the existing `outputs/` ignore.

## Integration into `environments/beamng.py`

Removed:
- Class constants `DEFAULT_WAYPOINTS`, `DDPG_WAYPOINTS`, `SPAWN_POS`, `SPAWN_ROT`.

Added in `__init__`:
- `self.trajectory: TrajectoryData | None = None` (loaded lazily once the scenario is first launched).

Modified `_launch` / `_load_scenario`:
- Two-phase launch when the cache is missing:
  1. **Probe phase**: open BeamNG, load a minimal `Scenario` with the chosen map and a placeholder vehicle at `(0, 0, 100)`. Call `bng.scenario.get_road_network(...)`, build and save the `TrajectoryData`.
  2. **Real scenario**: close the probe scenario, build the actual `Scenario` with `spawn_pos = trajectory.spawn_pos` and `spawn_rot = trajectory.spawn_rot`, set `self.waypoints` from sparse or dense based on `reward_mode`.
- When the cache exists: skip the probe phase entirely.

Waypoint selection:
- `reward_mode == "ddpg"` → `dense_waypoints`
- otherwise → `sparse_waypoints`

## Main menu

`main.py` gets one new option:

```
5. Generate trajectories for maps
```

Selecting it asks which map (or "all") and runs `load_or_generate(...)` in a one-shot BeamNG session, writing the JSONs. Useful for pre-warming the cache before a training campaign.

## File touch list

| Action | Path |
|---|---|
| Create | `core/trajectory.py` |
| Create | `docs/superpowers/specs/2026-05-23-auto-trajectory-generation-design.md` (this file) |
| Modify | `environments/beamng.py` |
| Modify | `main.py` (+ `core/cli.py` for the new menu entry) |
| Modify | `scenario_creator.md` (point to the auto flow, mark manual flow as legacy) |
| Modify | `README.md` (mention the trajectory cache) |
| Create dir | `outputs/trajectories/` (gitignored by parent rule) |

## Error handling

- BeamNG not running when generation is needed → raise `RuntimeError` with a clear message telling the user to run "Generate trajectories" from the main menu first.
- `get_road_network()` returns malformed data → log a warning and use the fallback.
- Cache file present but unreadable / wrong schema → log a warning, delete it, regenerate.

## Testing

- Unit-test `resample()` against a hand-computed polyline (no BeamNG needed).
- Unit-test `heading_to_quat()` for the four cardinal directions (matches `scenario_creator.md` table).
- Unit-test the fallback square loop (4 corners, closed).
- Manual smoke test: run "Generate trajectories" on `gridmap_v2`, then run a short DQN training and confirm the agent spawns on a road and waypoints are visible.

## Open questions resolved

- **Trajectory variation**: fixed per map (chosen by user).
- **Spacing**: adaptive per algorithm class — sparse for discrete, dense for continuous.
- **Source**: road network extraction (chosen over AI-recording, human-recording, and pathfinding).
- **Multi-route graph**: deferred to Phase 2 — phase 1 takes the longest single road.
