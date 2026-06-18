# BeamNG body-orientation & wheel-terrain observation options

**Date:** 2026-06-18
**Status:** Approved (design)

## Problem

The LiDAR is mounted directly on the car body. Because the suspension pitches
and rolls, the sensor's readings are skewed relative to the road — the agent
sees obstacle distances tilted by the chassis attitude without knowing the
chassis is tilted. Giving the agent the car's **pitch** and **roll**, plus where
each side of the car sits relative to the **road edges**, lets it interpret the
LiDAR (and drive) with that context.

Today this exists only as a one-off subclass, `BeamNGContinuousRollEnv`, which
hardcodes pitch + roll + 4 wheel-terrain values onto the continuous env. No
other env (lidar, lidar_grid, camera, or any multi-agent slot) can use it, and
the features can't be enabled or disabled independently.

## Goal

Turn these features into **two independent, opt-in flags** available to every
BeamNG environment (single-agent and multi-agent), threaded through the same
plumbing as the existing `trajectory_hints` per-session option.

- `body_orientation` — appends `[pitch, roll]` (**+2** observation dims)
- `wheel_terrain` — appends `[left_terrain, right_terrain]` (**+2** observation dims)

Both default to `False`, so existing behaviour is unchanged unless opted in.

## Feature definitions

### Body orientation (`+2`)
Derived from the vehicle's world-space `up` vector projected onto its forward
and lateral axes (same math as today's `BeamNGContinuousRollEnv`):

- `pitch` — forward/backward tilt, `+` = nose up (uphill), `−` = nose down.
- `roll` — lateral tilt, `+` = leaning right, `−` = leaning left.

A flat vehicle reads `(0, 0)`; 90° tilt saturates at `±1`. No extra sensor
needed — `pitch`/`roll` come from `vehicle.state` (`dir`, `up`).

### Wheel terrain (`+2`)
From a `RoadsSensor`, measured at the front-axle midpoint:

- `left_terrain` — `+1` = well inside the road on the left, `0` = at the left
  edge, `−1` = off-road on the left. Computed as
  `clip((dist2Left − HALF_TRACK_WIDTH) / halfWidth, −1, 1)`.
- `right_terrain` — same for the right edge.

**Honest 2-value form:** the current subclass emits 4 values (FL/FR/RL/RR) but
`RoadsSensor` only samples the front-axle midpoint, so RL/RR merely duplicate
FL/FR. We drop the duplication and emit the two real measurements only.

## Observation layout

Extra blocks are appended at the **end**, after the waypoint hints, so the
perception and hint blocks stay contiguous and existing models without the flags
are byte-for-byte unchanged:

```
kinematic(6) | perception(P) | hints(2·H) | [pitch, roll]? | [left, right]?
```

`n_states = N_STATES + 2·trajectory_hints + 2·body_orientation + 2·wheel_terrain`
(booleans counted as 0/1).

## Components

### 1. `environments/beamng.py` — `BeamNGDrivingEnv`
- Add `body_orientation: bool = False` and `wheel_terrain: bool = False` to
  `__init__`; store on self; fold into the `n_states` formula above.
- Add a `roads_sensor` attribute (created only when `wheel_terrain` is on).
- New helpers:
  - `_body_orientation_features(state) -> np.ndarray` — returns `[pitch, roll]`
    (lifted from `BeamNGContinuousRollEnv._observe`).
  - `_wheel_terrain_features() -> np.ndarray` — returns `[left, right]` (polls
    `roads_sensor`; falls back to neutral values if unavailable).
  - `_extra_features(state) -> np.ndarray` — concatenates whichever of the two
    blocks are enabled (empty array when neither is).
- `_observe` ends with `np.concatenate([..., self._extra_features(state)])`.
- `_load_scenario` attaches a `RoadsSensor` after `start_scenario()` when
  `wheel_terrain` is on (and removes any prior one); `close()` tears it down.

### 2. `environments/beamng.py` — `BeamNGCameraEnv`
Its `_observe` is a full override (doesn't call `super()`), so it must also end
by concatenating `self._extra_features(state)`. The shared helper keeps the math
in one place.

### 3. Delete `BeamNGContinuousRollEnv`
Remove the subclass entirely and its `beamng_continuous_roll` registry entry —
fully superseded by the flags. This drops the experimental 20-dim env and any
checkpoints trained against it; acceptable as it was a hardcoded stopgap.

### 4. `environments/beamng_multi.py`
- `VehicleSlot` gains `body_orientation: bool = False`, `wheel_terrain: bool =
  False`, and a per-slot `roads_sensor` attribute.
- `build_slots` reads both flags from each spec dict (default `False`).
- `slot_n_states(env_name, trajectory_hints, body_orientation, wheel_terrain)`
  adds `2·body_orientation + 2·wheel_terrain`.
- `_create_slot_sensor` attaches a per-slot `RoadsSensor` when `wheel_terrain`
  is on; `close()` removes it alongside lidar/camera.
- `observe()` appends `_extra_features` for the slot (shared helper or mirrored
  for the per-slot sensor handle).

### 5. `core/cli.py`
For any `beamng*` env, after the existing hints prompt, ask two yes/no
questions:
- "Include body orientation (pitch + roll)? [y/N]"
- "Include per-wheel road position? [y/N]"

Feed the booleans into `beamng_kwargs` (single-agent train & play) and into each
vehicle's spec dict (multi-agent), and add `2·flag` to the `n_states` used to
size the agent. The play path must prompt identically so the loaded model's
input size matches.

### 6. `environments/__init__.py` (registry)
Each `_make_beamng*` factory forwards `body_orientation` / `wheel_terrain` to the
constructor. Registry `metadata["n_states"]` stays at the base (flag-off) value;
the CLI already adjusts `n_states` at agent-build time (as it does for hints).

## Error handling
- `RoadsSensor` poll may return a list or dict, or be missing — normalize and
  fall back to neutral terrain values (`HALF_TRACK_WIDTH` distances → `0` after
  clamping) so a sensor hiccup never crashes `observe()`.
- `body_orientation` reads `state["dir"]`/`state["up"]` with safe defaults
  (`(0,1,0)` / `(0,0,1)`) when state is missing.

## Testing
- `tests/test_beamng_multi.py`: update `slot_n_states` expectations; add cases
  for each flag combination (neither / body only / wheels only / both) and
  assert observation length.
- `tests/test_cli_multi.py`: cover the new prompts feeding the spec dict and the
  `n_states` sizing.
- Verify `beamng_continuous_roll` references are gone (registry, CLI, tests).
- Confirm flag-off observation length equals the pre-change length for every env
  (regression guard).

## Out of scope
- True per-wheel suspension/ground telemetry (kept to the honest 2-value
  front-axle measurement).
- Adding orientation/terrain terms to the reward functions — observation only.
