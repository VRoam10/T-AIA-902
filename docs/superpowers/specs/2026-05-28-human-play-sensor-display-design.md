# Human Play Sensor Display — Design Spec

**Date:** 2026-05-28
**Branch:** feat/remove-radar

---

## Overview

Add a sensor selection prompt to the CLI human play mode. Before launching BeamNG, the user chooses:
1. **None** — drive with no sensor output in the terminal
2. **LiDAR** — print the 8 normalized LiDAR bins each tick
3. **Camera** — use `BeamNGCameraEnv` and render the 16×16 grayscale frame as ASCII art in-place

---

## Architecture

### CLI (`core/cli.py`)

`_human_play_menu()` gains a third prompt after map and vehicle selection:

```
Show sensor during play?
  1. None
  2. LiDAR
  3. Camera
```

Routing logic:
- **None** → `BeamNGDrivingEnv(map_name, vehicle_id)` → `env.human_play()`
- **LiDAR** → `BeamNGDrivingEnv(map_name, vehicle_id)` → `env.human_play_lidar()`
- **Camera** → `BeamNGCameraEnv(map_name, vehicle_id)` → `env.human_play_camera()`

`BeamNGCameraEnv` is imported directly in `core/cli.py` via `from environments.beamng import BeamNGCameraEnv`.

### Environment (`environments/beamng.py`)

#### `BeamNGContinuousEnv.human_play()` (modified)

Remove the existing `[LiDAR bins]` print. This becomes the clean no-sensor play loop. All other logic (polling position, speed, heading, markers, sleep) stays identical.

#### `BeamNGContinuousEnv.human_play_lidar()` (new)

Identical to `human_play()` with one addition per tick:

```python
print(f"[LiDAR bins] {' '.join(f'{v:.2f}' for v in lidar_bins)}")
```

This is exactly the line moved out of `human_play()`.

#### `BeamNGCameraEnv.human_play_camera()` (new)

Defined on `BeamNGCameraEnv`. Per tick:
1. Poll LiDAR (for kinematics) and camera sensor
2. Call `_process_camera()` → 256-element float array, reshape to 16×16
3. Map each value to a character from `' ░▒▓█'` (5-level ramp, index = `int(v * 4)`)
4. Print the 16 rows, then move cursor up 16 lines using `\033[16A` so the next tick overwrites in-place (no scroll)
5. On exit (KeyboardInterrupt / loop end), print 16 blank lines to clear the frame

---

## Data Flow

```
CLI prompt → sensor choice
    │
    ├─ None    → BeamNGDrivingEnv → human_play()          → no output
    ├─ LiDAR   → BeamNGDrivingEnv → human_play_lidar()    → "[LiDAR bins] ..."
    └─ Camera  → BeamNGCameraEnv  → human_play_camera()   → ASCII frame (in-place)
```

---

## Error Handling

- If `BeamNGCameraEnv` fails to initialise the camera sensor, it raises inside `_load_scenario()` — the existing exception propagation in `_human_play_menu()` (the `finally: env.close()` block) already handles cleanup.
- ANSI escape codes for in-place rendering work on Windows 10+ (VT100 enabled by default). No fallback needed.

---

## Testing

Manual verification:
1. Launch human play → select None → confirm no sensor output in terminal
2. Launch human play → select LiDAR → confirm `[LiDAR bins]` line appears each tick
3. Launch human play → select Camera → confirm 16-row ASCII frame updates in-place

No automated tests required — this is a display-only feature with no state mutations.

---

## Files Changed

| File | Change |
|------|--------|
| `core/cli.py` | Add sensor prompt in `_human_play_menu()`, route to correct env + method |
| `environments/beamng.py` | Strip LiDAR print from `human_play()`, add `human_play_lidar()`, add `BeamNGCameraEnv.human_play_camera()` |
