# BeamNG Driving Environment

`environments/beamng.py` — `BeamNGDrivingEnv`

Gymnasium-style RL environment wrapping BeamNG.drive via beamngpy.
Map: `gridmap_v2`. Vehicle: Burnside (taxi config).

---

## Timing

| Level | Detail |
|---|---|
| BeamNG physics | 30 Hz (set via `bng.set_deterministic(30)`) |
| Per `env.step()` | `bng.step(10)` → advances **10 physics ticks** = **~333 ms** of sim time |
| Effective decision rate | **~3 decisions/second** of sim time |
| Max steps per episode | 400 steps → **~133 seconds** of sim time |

> Note: wall-clock time per step depends on machine speed and BeamNG load. In headless mode expect roughly 1–3 wall-clock seconds per step during training.

---

## State Space

`N_STATES = 13` — a flat `float32` array, all values normalized to approximately `[-1, 1]` or `[0, 1]`.

| Index | Name | Raw source | Normalization |
|---|---|---|---|
| 0 | `speed` | `electrics.wheelspeed` (m/s) | `/ 50.0`, clipped to `[-1, 1]` |
| 1 | `steering` | `electrics.steering` | clipped to `[-1, 1]` (already normalized by BeamNG) |
| 2 | `heading_error` | angle between vehicle heading and next waypoint (rad) | `/ π`, clipped to `[-1, 1]` |
| 3 | `lateral_error` | perpendicular distance from the path to next waypoint (m) | `/ 5.0`, clipped to `[-1, 1]` |
| 4 | `damage` | `damage_sensor.damage` (cumulative) | `/ 1000.0`, clipped to `[0, 1]` |
| 5–12 | `lidar[0..7]` | nearest obstacle in each angular bin (m) | `/ 50.0` (LIDAR_MAX_DIST), `0` = obstacle right there, `1` = clear |

---

## Action Space

`N_ACTIONS = 7` — discrete.

| Index | Description | Throttle | Steering | Brake |
|---|---|---|---|---|
| 0 | Idle / coast | 0.0 | 0.0 | 0.0 |
| 1 | Full throttle straight | 1.0 | 0.0 | 0.0 |
| 2 | Throttle + slight left | 1.0 | -0.3 | 0.0 |
| 3 | Throttle + slight right | 1.0 | +0.3 | 0.0 |
| 4 | Brake | 0.0 | 0.0 | 1.0 |
| 5 | Throttle + sharp left | 0.5 | -0.6 | 0.0 |
| 6 | Throttle + sharp right | 0.5 | +0.6 | 0.0 |

Steering convention: negative = left, positive = right.

Attention with brake at more than 0, the car can reverse.

---

## Reward Function

Computed once per `step()` call from the normalized observation.

### Per-step terms

| Condition | Delta | Note |
|---|---|---|
| Forward speed | `+speed × 2.0` | `speed` is already normalized to `[0, 1]`, max `+2.0` per step |
| Lateral error | `-|lateral_err| × 1.5` | penalizes drifting sideways off the path |
| Heading error | `-|heading_err| × 0.5` | penalizes pointing away from the next waypoint |
| Steering wobble | `-|steering| × 0.1` | penalizes unnecessary wheel movement |
| Stationary (`speed < 0.05`) | `-0.5` | encourages the agent to keep moving |
| Damage spike (> 50 units since last step) | `-50.0` | one-time penalty per collision event |
| Too far from the checkpoint | `-distance × 10.0` | Penalty from stepping too far from checkpoint |

### Episode events

| Event | Delta | Termination? |
|---|---|---|
| Checkpoint reached (waypoint N) | `+100 × N` (e.g. wp1 = +100, wp2 = +200) | No |
| Lap completed (all 3 waypoints hit) | `+200` | **Yes** |
| Cumulative damage ≥ 500 | — | **Yes** |
| Step limit reached (400 steps) | — | **Yes** |
| Too far from checkpoint | `-100` | **Yes** |

### Typical reward range per episode

| Outcome | Approximate total reward |
|---|---|
| Stationary / spinning | −200 to −100 |
| Moves but no checkpoints | −100 to +200 |
| Hits 1 checkpoint | ~+200 to +400 |
| Completes the lap | ~+600 to +800 |

---

## Waypoints & Track

Three waypoints trace a short loop on `gridmap_v2`, (shuffled randomly each episode (order only, positions are fixed)(not ready yet)):

| # | World position (x, y, z) |
|---|---|
| A | (61, −755, 100) |
| B | (90, −734, 100) |
| C | (116, −612, 100) |

A waypoint is considered "hit" when the vehicle centre comes within **8 m** of it.

The agent spawns at `(61, −788, 101)` facing north.

---

## LiDAR

| Parameter | Value |
|---|---|
| Rays (angular bins) | 8 |
| Field of view | 120° (forward-facing, ±60°) |
| Max range | 50 m |
| Mount position | (0, 0, 1.7) — roof height |
| Direction | forward (0, −1, 0) in BeamNG coords |
| Vertical angle | 10° |
| Vertical resolution | 16 layers |

Each bin returns the **nearest** point distance in that angular slice, normalized to `[0, 1]`.
Bins are ordered left-to-right across the FOV.

---

## Episode Lifecycle

```
reset()
  └─ (first call) _launch() → opens BeamNG headless, loads scenario
  └─ (subsequent) scenario.restart() + randomize waypoints
  └─ bng.step(5) — settle physics
  └─ returns initial observation

step(action_idx)
  └─ vehicle.control(throttle, steering, brake)
  └─ bng.step(10) — advance 10 physics ticks (~333 ms sim time)
  └─ _observe() — poll electrics, damage, lidar
  └─ _compute_reward() — compute reward + done flag
  └─ returns (obs, reward, done, info)

close()
  └─ lidar.remove()
  └─ bng.close()
```

---

## Launch Options

BeamNG can be started with `headless`:

- `-headless` — no rendering window

Each environment instance binds to a unique port (default `25252`), allowing multiple instances to run in parallel for grid search.
