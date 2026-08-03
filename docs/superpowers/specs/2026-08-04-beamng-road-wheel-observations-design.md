# Road position, wheel performance, and path-relative pace

**Date:** 2026-08-04
**Branch:** `feat/beamng-racing`
**Status:** approved design, not yet implemented

Two `beamng` observation options — `road_info` (where the car is on the road) and
`wheel_info` (how the tyres are coping) — plus the reward change that makes long
checkpoint gaps trainable. Both flags feed the derived checkpoint path, so a run's
file name states its observation width.

---

## Problem

The policy cannot find the next checkpoint when checkpoints are far apart, which is
the normal case on the game's own tracks: 30 of the 44 shipped sprint/lap tracks have
a gap over 300 m, and italy's `highway1` averages 1064 m. Adding checkpoints is not
the fix — the authored chain *is* the race, and densifying it changes what is being
scored.

Five separate things misfire once a segment is long — three in the observation, two in
the reward:

| Where | What |
|---|---|
| [`beamng.py:859-891`](../../../environments/beamng.py#L859) | `heading_err` is the bearing to the next checkpoint **in a straight line**. Over 300 m of bend, "point at the checkpoint" means "point into the scenery". |
| [`beamng.py:890`](../../../environments/beamng.py#L890) | `lateral_err = dist * sin(heading_err)` is offset from that line of sight — and is algebraically derivable from the two features beside it, so the slot carries no information at all. |
| [`beamng_reward.py:178-183`](../../../environments/beamng_reward.py#L178) | The pace terms are straight-line closure to the checkpoint plus speed projected onto that same bearing. Following the road round a bend *increases* straight-line distance, so correct driving is penalised. |
| [`beamng_reward.py:69`](../../../environments/beamng_reward.py#L69) | The segment-time bonus par comes from `SPARSE_SPACING_M` (25 m), so on a game track the bonus is permanently zero. |
| [`beamng.py:818`](../../../environments/beamng.py#L818) | `trajectory_hints` normalize over 100 m, so on a game track every hint saturates at ±1. |

Separately, the observation says nothing about the road itself (where its edges are,
which way it bends) or about grip (wheelspin, lockup, sliding), both of which a fast
lap depends on.

## Decisions

Locked by explicit choice; do not re-litigate.

1. **Scope**: new observation features **and** the two long-segment reward terms. An
   honest observation alone would still be contradicted by a reward that pays for
   straight-line closure to a distant checkpoint.
2. **Road source**: the live `RoadsSensor`, with the known reset freeze fixed. It is
   the only run-time source of actual road shape — a cached polyline between two
   authored checkpoints is a straight chord and adds nothing over today's bearing.
3. **Road block**: 6 features — two edges, road-relative heading, signed curvature,
   and one look-ahead centerline point.
4. **Wheel block**: 4 features — longitudinal slip, slip angle, ABS active, lateral g
   (via a `GForces` sensor attached alongside Electrics/Damage).
5. **Options**: two independent booleans, `road_info` and `wheel_info`, default false,
   each with its own save-path token so they can be benchmarked separately.
   `wheel_terrain` is **deleted**: its `edgeL`/`edgeR` pair is a strict subset of
   `road_info`, and it was never offered in any menu.
6. **Freeze fix**: guarantee at least one physics step between a teleport and the
   first road poll; neutral features until a reading exists.

## Observation layout

Both blocks append at the tail, so every existing prefix keeps its meaning. The
`wheel_terrain` pair is absorbed by the road block rather than sitting beside it.

```
kin(6) | perception(P) | hints(2H) | [pitch, roll]? | road(6)? | wheel(4)?
```

`beamng_spec.obs_size` becomes:

```python
obs_size(sensor, trajectory_hints=0, body_orientation=False,
         road_info=False, wheel_info=False) -> int
```

with `ROAD_FEATURES = 6` and `WHEEL_FEATURES = 4` beside the existing sizes. The
`wheel_terrain` parameter is removed outright rather than deprecated: a dead kwarg on
this function is exactly how a wrong `n_states` gets computed silently.

### `road_info` — one `RoadsSensor` poll per step

The reading is selected by the existing `_latest_road_reading` normalizer, which
already collapses the index-map / flat-dict / list poll shapes.

| Feature | Source field | Normalization |
|---|---|---|
| `edge_left` | `dist2Left` | `(d - HALF_TRACK_WIDTH) / halfWidth`, clip ±1 — today's `wheel_terrain` formula, unchanged. +1 = well on road, 0 = at the edge, −1 = off road |
| `edge_right` | `dist2Right` | same |
| `road_heading` | `headingAngle` | `/ (pi/2)`, clip ±1 — the car's yaw relative to the road reference line |
| `curvature` | `roadRadius` | `clip(CURV_NORM_M / R, 0, 1)`, signed. `CURV_NORM_M = 50.0`, so a 50 m hairpin reads 1.0 and a 500 m sweeper 0.1. `NaN` (the sensor's "road is straight") → 0.0 |
| `ahead_fwd` | `P0..P3` | the four centerline points de-rotated into vehicle-local coords and sorted by forward distance; the farthest one's forward component `/ ROAD_AHEAD_NORM_M` (50.0), clip ±1 |
| `ahead_left` | `P0..P3` | the same point's lateral component, same normalization |

Two details that must not be guessed at implementation time:

* **Curvature sign** comes from the z-component of the cross product of successive
  centerline-point deltas, *not* from the lateral offset of a point — the latter
  inherits the car's own displacement from the centerline and would read a straight
  road as curved whenever the car runs wide.
* **Point ordering** — the sensor documents `P0..P3` as the four *closest* centerline
  points, not four points ahead. They are sorted by their vehicle-local forward
  component, and a reading with no usable point yields zeros.

Every field falls back the way `wheel_terrain_features` already does: a missing
payload, a missing key or a non-finite value yields the neutral value for that
feature rather than raising. A sensor hiccup must never end an episode.

The `RoadsSensor` is created only when `road_info` is on, and torn down through the
existing bounded `remove_sensor` helper — a run with the flag off makes no road poll
and carries no freeze exposure at all.

### `wheel_info` — Electrics, vehicle state, and `GForces`

`GForces` is a classic sensor, attached next to Electrics and Damage, so it is read
inside the `poll_sensors()` round-trip the env already makes. No extra sim traffic. It
is attached only when `wheel_info` is on.

| Feature | Source | Normalization |
|---|---|---|
| `long_slip` | `electrics.wheelspeed − ‖state.vel‖` | `/ max(‖vel‖, SLIP_REF_MS=5.0)`, clip ±1. Positive = wheelspin, negative = lockup |
| `slip_angle` | `state.vel` vs `state.dir` | signed angle in the vehicle frame `/ (pi/2)`, clip ±1; forced to 0.0 below 1 m/s, where velocity direction is noise |
| `abs_active` | `electrics.abs_active` | 0.0 / 1.0 |
| `lat_g` | `GForces` | first present of `gx2`, `gx`, else 0.0; `/ LAT_G_NORM (1.5 g)`, clip ±1 |

Ground speed comes from the state vector rather than `electrics.airspeed`: we already
poll it and its meaning is unambiguous.

`abs_active` is the only driver aid worth reading on this car — the Vivace Hillclimb
config deletes ESC and TC (`vivace_DSE_ESC: ""`, `vivace_DSE_TC: ""`) and keeps ABS
(`vivace_DSE_ABS`), so `esc_active` / `tcs_active` would be constant zero.

**`lat_g` axis is unverified.** This project's vehicle frame has forward = `−Y`
(see `LIDAR_MOUNT_DIR` in [`beamng_sensors.py:44`](../../../environments/beamng_sensors.py#L44)),
which makes **x** the lateral axis — but `GForces` is a raw passthrough with no
documented keys in beamngpy, so the axis choice is a hypothesis to confirm in-sim
(see Verification).

## The reset freeze

Polling the `RoadsSensor` right after a teleport with no intervening physics step
hard-freezes the simulator on road-dense maps (measured: gridmap fine, west_coast
froze). That is why `wheel_terrain` was never offered in a menu, and it has to be
fixed before road data can be a real option.

The fix is local to `reset()`: advance at least one physics step between the teleport
and the first road poll, and return neutral zeros from the road block while no
reading exists. No sensor recreation, no watchdog thread — the trigger is known, and
one extra step per episode is free next to the teleport itself.

## Path projection

New module `environments/beamng_path.py`, one pure function:

```python
project_onto_path(polyline, pos) -> PathPosition(
    progress_m,      # arc length from the polyline start to the projection point
    cross_track_m,   # signed perpendicular distance from the polyline (+ = left)
    tangent_rad,     # heading of the segment the car projects onto
    segment_index,
    segment_len_m,   # length of that segment
)
```

The polyline is `[spawn_pos, *waypoints]`. Prepending the spawn is what makes progress
work on the first segment; `waypoints` starts *after* the spawn, so projecting onto it
alone would clamp to zero until checkpoint 0 fell behind the car.

This **replaces** `track_progress_m`
([`beamng_geometry.py:196`](../../../environments/beamng_geometry.py#L196)), which is
deleted with its tests moving to the new module. Keeping both would invite drift, and
the old one cannot do this job: it is `arc_to_target − straight_line_remaining`, so its
per-step delta *is* the straight-line closure the pace terms already use. It made the
signal continuous across checkpoints; it cannot stop a bend from reading as backward
progress. Projection can, because it is a function of position alone.

Multi-lap runs add `laps_completed * total_length` to `progress_m`, derived from
`waypoint_idx // len(waypoints)`. Laps are locked at 1 today, but the offset is three
lines and its absence would silently flatten the gap term on the first lap boundary.

`reset()` sets `last_progress_m` from the spawn position rather than 0.0, so the first
step of an episode reports the metres it actually covered instead of the whole distance
from the polyline start. The dense/sparse curriculum swaps `waypoints` only at reset,
so the guide polyline is fixed for the length of an episode.

## Reward changes

In [`beamng_reward.py`](../../../environments/beamng_reward.py):

1. **Progress** becomes `PROGRESS_COEF * (progress_m - last_progress_m)` — metres
   gained *along the path*. The `checkpoint_hit` zeroing disappears with it:
   projection progress does not jump when the target index advances, so the case that
   hack existed for no longer occurs.
2. **Speed alignment** becomes `speed * cos(vehicle_heading - tangent_rad)` — speed
   projected onto the path tangent instead of onto the bearing to a checkpoint up to a
   kilometre away. Passed in as a new `path_alignment` argument; when it is absent the
   reward falls back to today's `cos(heading_err * pi)`, so no caller is forced to
   change in the same step.
3. **Segment-time bonus** becomes scale-free:
   `SEGMENT_TIME_BONUS * clip(1 - steps_since_checkpoint / par, 0, 1)`, with
   `par = steps_for_distance(segment_len_m, SEGMENT_PAR_SPEED_MS)` taken from the
   segment actually being driven, and `SEGMENT_TIME_BONUS = 25.0` — half a checkpoint
   bonus, which is what the current constants were tuned to pay. The module-level
   `SEGMENT_TARGET_STEPS` and `SEGMENT_TIME_COEF` are deleted; `SEGMENT_PAR_SPEED_MS`
   stays, now applied per segment.

   Recomputing par under the existing `(par - steps) * SEGMENT_TIME_COEF` shape is not
   enough: on italy `highway1` (1064 m average) par is ~266 steps and a 40 m/s run
   takes ~80, so the bonus would pay 744 — more than the 50 checkpoint bonus and the
   300 finish bonus combined, and it would grow with track length.

The race gap term reads the same `progress_m`, so "how far along the track am I" has
exactly one definition, shared by pace and gap. The single-vehicle env starts
computing it (today only the multi/race envs do).

### One observation slot changes meaning

`lateral_err` (kin index 3) is filled with `cross_track_m / 5.0` — true signed
distance from the path — instead of `dist * sin(heading_err)`. Same width, same
normalization; existing checkpoints still load. This is not new information being
added, it is a slot that currently carries none: `dist * sin(heading_err)` is a
function of the two features either side of it.

`heading_err` and `dist` are unchanged. The agent does need to know where the scored
checkpoint is; what it lacked was where the *road* goes.

## Plumbing

`road_info` / `wheel_info` replace `wheel_terrain` at every layer:

* envs — `BeamNGDrivingEnv` kwargs, `VehicleSlot`
  ([`beamng_multi.py`](../../../environments/beamng_multi.py)), race specs
  ([`beamng_race.py`](../../../environments/beamng_race.py))
* factory — `_make_beamng` ([`environments/__init__.py:19`](../../../environments/__init__.py#L19)),
  where every option must be named *and* forwarded or the `**kwargs` sink swallows it
* actions — `BeamNGOptions` and `RacerOptions`
  ([`pipeline_actions.py:83`](../../../core/pipeline_actions.py#L83),
  [`:165`](../../../core/pipeline_actions.py#L165)), all four `obs_size` /
  `slot_n_states` call sites, and [`tui_backend.py`](../../../core/tui_backend.py)
  payload decoding. `RacerOptions` carries no `wheel_terrain` today, so racers gain
  both flags for the first time.

New module `environments/beamng_features.py` holds the two pure block builders plus
`_latest_road_reading`, moved out of `beamng_geometry.py` (which stays LiDAR/bbox/grid
geometry). `wheel_terrain_features` becomes `road_info_features`, reusing its edge math
verbatim so those two features keep their exact numerics inside the larger block.

### Save path

`beamngPathSuffix` ([`workflows.ts:155`](../../../tui/src/workflows.ts#L155)) gains two
tokens, in fixed order `_h{n} _ori _road _whl`:

```
outputs/dqn_lidar_road_whl.pth
outputs/td3_camera_h3_ori_road_whl.pth
outputs/multi-agents/dqn_lidar_road_0.pth
```

The multi-slot path builder ([`forms.ts:228`](../../../tui/src/forms.ts#L228)) and the
racer chip labels ([`forms.ts:250`](../../../tui/src/forms.ts#L250)) use the same
tokens. Both new keys join the in-place refresh allow-lists in
[`controller.ts:85`](../../../tui/src/controller.ts#L85) and
[`:90`](../../../tui/src/controller.ts#L90), so the displayed path tracks the toggles
instead of going stale.

### Forms

Two `false`/`true` choices wherever `body_orientation` already appears: train,
multi-train, course racers, human play. Default `false`, so every existing checkpoint
keeps loading at its current width.

Human play gains obs-log labels for the new blocks — `edgeL edgeR rdhead curv aheadF
aheadL` and `slip slipang abs latg` — in `_format_observation_lines`. Those lines are
how the `lat_g` axis and the curvature sign actually get verified.

## Testing

* pure-math tables for both feature builders: `NaN` `roadRadius`, missing keys,
  off-road negatives, centerline points arriving out of order, sub-1 m/s slip angle
* `project_onto_path`: straight line, bend, hairpin, position past the end, single
  point, empty polyline, and the lap offset
* reward: the segment bonus pays comparably on a 25 m and a 1000 m segment; progress
  is positive round a bend where straight-line closure is negative; `path_alignment`
  absent reproduces today's number exactly
* env wiring through the fake-`bng` doubles: reset steps before the first road poll,
  and the first observation of an episode is neutral in the road block
* option plumbing: `test_pipeline_actions*`, `test_tui_backend.py`, `test_env_factories.py`
* TypeScript: suffix tokens, payload builders, racer chips

## Verification (in-sim, not unit tests)

1. Human play on `west_coast_usa` — the map that froze — with `road_info` on: resets
   repeatedly without a freeze.
2. A short training run on `west_coast_usa` with both flags on, to the same end.
3. From the human-play obs log: `lat_g` swings sign with steering direction in a
   steady corner, and `curvature` sign matches the direction the road bends.
4. A sprint on italy (`highway1`, ~1064 m segments): the reward's progress term stays
   positive while the car follows the road round a bend that increases straight-line
   distance to the checkpoint.

## Out of scope

Noted, not fixed here:

* **Shift mode.** Nothing in the repo calls `vehicle.set_shift_mode`, and both the old
  `trackday_M` and the current `hillclimb_SQ` are manual/sequential gearboxes. If the
  car never upshifts, that dwarfs every effect in this design. Worth its own look.
* **Speed normalization.** `speed / 50.0` ([`beamng.py:707`](../../../environments/beamng.py#L707))
  saturates well below this car's 81 m/s top speed.
* **The `ACTIONS` retune.** Sharp turns at 0.15 throttle was chosen for the
  spin-prone RWD Scintilla; the car has been AWD for two changes now.
* **Road-snapping the checkpoint chain.** Projecting authored checkpoints onto the road
  network at generation time would give road-following geometry with no per-step sim
  cost, but needs a new generate step and a cache regeneration.
