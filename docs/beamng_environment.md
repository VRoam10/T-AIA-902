# BeamNG Racing Environment

`environments/beamng.py` — `BeamNGDrivingEnv`

Gymnasium-style RL environment wrapping BeamNG.drive via beamngpy. One environment,
parameterized by two independent axes. Vehicle: **Hirochi Sunburst**, race config
(`vehicles/sunburst2/trackday_M.pc`) — the same car for every entrant, so a head-to-head
result reflects the policies and not the machinery.

---

## The two axes

Defined once in `environments/beamng_spec.py`, which is the single source of truth for
every size derived from them.

| Axis | Values | Chosen by |
|---|---|---|
| `sensor` | `lidar`, `adv_lidar`, `camera` | the user |
| `output` | `fixed`, `continuous` | **derived from the algorithm** |

`output` is not a user-facing choice: a DQN head cannot emit continuous controls and
DDPG/TD3 emit nothing else, so `beamng_spec.output_for_algo` derives it. An
unregistered algorithm raises rather than defaulting — a wrong action head only shows
up much later, as nonsense driving.

| `sensor` | Perception block | `n_states` (no options) |
|---|---|---|
| `lidar` | 8 distance bins, one collapsed row | 14 |
| `adv_lidar` | 4 x 8 grid (elevation x azimuth) | 38 |
| `camera` | 16 x 16 grayscale dashcam, flattened | 262 |

| `output` | Action space |
|---|---|
| `fixed` | 7 discrete actions (the `ACTIONS` table below) |
| `continuous` | 3 outputs: throttle, steering, brake |

---

## Timing

| Level | Detail |
|---|---|
| Deterministic sim rate | `PHYSICS_STEPS_PER_SECOND = 30` (via `bng.set_deterministic`) |
| Per `env.step()` | `bng.step(10)` → 10 simulation steps = **~333 ms** of sim time |
| Effective control rate | **~3 decisions/second** of sim time |
| Max steps per episode | `MAX_STEPS = 500` → **~167 s** of sim time |

These live in `beamng_spec` as `PHYSICS_STEPS_PER_SECOND`,
`PHYSICS_STEPS_PER_ENV_STEP` and `SECONDS_PER_ENV_STEP`. Anything reasoning in
*seconds* — the reward's segment-time par, the realtime race tick — derives from them
via `beamng_spec.steps_for_distance` rather than assuming a step duration.

> Wall-clock time per step depends on machine speed and BeamNG load, and is unrelated
> to the sim-time figures above.

---

## Observation

A flat `float32` array, all values normalized to approximately `[-1, 1]` or `[0, 1]`.
Blocks are concatenated in this order:

```
kinematic(6) | perception(P) | hints(2*H) | [pitch, roll]? | [edgeL, edgeR]?
```

| Index | Name | Raw source | Normalization |
|---|---|---|---|
| 0 | `speed` | `electrics.wheelspeed` (m/s) | `/ 50.0`, clipped to `[-1, 1]` |
| 1 | `steering` | `electrics.steering` | clipped to `[-1, 1]` (already normalized) |
| 2 | `heading_error` | angle between heading and next waypoint (rad) | `/ π`, clipped |
| 3 | `lateral_error` | perpendicular distance from the path (m) | `/ 5.0`, clipped |
| 4 | `damage` | `damage_sensor.damage` (cumulative) | `/ 1000.0`, clipped to `[0, 1]` |
| 5 | `dist` | distance to the target checkpoint (m) | `/ CHECKPOINT_WARN_DIST`, clipped to `[0, 2]` |
| 6.. | perception | the sensor's block | see the table above |

Optional tails, both off by default:

- `trajectory_hints=H` — vehicle-local `(forward, left)` of the next `H` waypoints,
  normalized over 100 m. **+2H** dims.
- `body_orientation` — `[pitch, roll]` from the vehicle's forward/up vectors. **+2**.
- `wheel_terrain` — `[left, right]` road-edge position from a `RoadsSensor`. **+2**.
  **Not offered in the menus**: polling the sensor in the unstepped reset path
  hard-freezes training on road-dense maps.

`beamng_spec.obs_size(sensor, hints, body_orientation, wheel_terrain)` is the only
place this arithmetic lives.

---

## Action space (`output: fixed`)

Throttle falls sharply as steering rises. The car is a mid-engine RWD with far more
power than grip, so the previous taxi-era table (0.4 throttle at 0.6 steering) spun it
on most corner entries.

| Index | Description | Throttle | Steering | Brake |
|---|---|---|---|---|
| 0 | Coast | 0.0 | 0.0 | 0.0 |
| 1 | Full throttle straight | 1.0 | 0.0 | 0.0 |
| 2 | Power-on slight left | 0.6 | -0.25 | 0.0 |
| 3 | Power-on slight right | 0.6 | +0.25 | 0.0 |
| 4 | Brake | 0.0 | 0.0 | 1.0 |
| 5 | Lift + sharp left | 0.15 | -0.55 | 0.0 |
| 6 | Lift + sharp right | 0.15 | +0.55 | 0.0 |

Steering convention: negative = left, positive = right.

> With brake above 0 the car can reverse.

For `output: continuous`, `controls_for` accepts a 3-vector `[throttle, steering,
brake]` (negative halves clipped away) or a 2-vector `[accel, steering]` where a
negative `accel` becomes brake. An `int` action is always read as a table index, so a
discrete agent driving a continuous env still produces valid controls.

---

## Reward

`environments/beamng_reward.compute_race_reward` — one function, called by the single,
multi and race envs, so a policy is rewarded for the same behaviour everywhere.

### Pace

| Term | Delta |
|---|---|
| Progress toward the waypoint (telescoping) | `+(last_dist - dist) × PROGRESS_COEF` |
| Speed projected on the target direction | `+speed × cos(heading_err) × SPEED_ALIGN_COEF` |
| **Every step** | `-STEP_PENALTY` |
| Checkpoint reached | `+CHECKPOINT_BONUS` (flat) |
| Checkpoint reached quickly | `+max(0, SEGMENT_TARGET_STEPS - steps_since_cp) × SEGMENT_TIME_COEF` |
| Path completed | `+FINISH_BONUS + (max_steps - steps) × FINISH_TIME_COEF` |

The step penalty and the two time bonuses are what make *fast* beat *clean but slow*.
They matter more than they look: the progress and speed-alignment terms both integrate
to a constant over a fixed distance regardless of pace, so they shape *where* the car
goes, not *how quickly*.

The checkpoint bonus is deliberately **flat**. The reward this replaced paid
`100 × waypoint_idx`, which made late-path reward an order of magnitude larger than
early-path reward for identical driving — enough to destabilise the value function.

### Contact and track limits

| Condition | Delta | Ends episode? |
|---|---|---|
| New damage | `-damage_delta × DAMAGE_DELTA_COEF` | No |
| Hard hit (> `HARD_HIT_DAMAGE` in one step) | `-HARD_HIT_PENALTY` | **No** — rubbing wheels mid-overtake must not end a race |
| Cumulative damage ≥ `MAX_DAMAGE` | `-CRASH_PENALTY` | **Yes** |
| Nearest LiDAR bin < 0.2 / < 0.4 | graded proximity penalty | No |
| Beyond `CHECKPOINT_WARN_DIST` from the target | graded, up to `OFF_TRACK_WARN_PENALTY` | No |
| Beyond `CHECKPOINT_RESET_DIST` | `-OFF_TRACK_RESET_PENALTY` | **Yes** |
| Step limit (`MAX_STEPS`) | — | **Yes** |

Reaching a checkpoint grants `INVULN_GRACE_STEPS` damage-immune steps, so brushing or
settling right at the checkpoint is not punished.

### Racing (course mode only)

When a rival's progress is supplied:

```
+GAP_COEF × ((my_progress - rival_progress) - (my_last - rival_last))
```

Telescoping, so summing it over an episode gives `GAP_COEF × the final gap in metres`.
One signal that rewards overtaking and defending equally, with no per-step position
bookkeeping, and impossible to farm by oscillating alongside the rival. Plus
`WIN_BONUS` for finishing first and `LOSE_PENALTY` (and termination) when the rival
does. Passing none of the progress arguments — the solo default — contributes exactly
zero, so single-car training is unaffected.

Progress is measured by `beamng_geometry.track_progress_m`: arc length along the
waypoint polyline to the current target, minus the straight-line distance still to
cover. The subtraction is what keeps it continuous across a checkpoint transition,
which a telescoping term requires.

---

## Waypoints and track

Paths are generated per map from the road network and cached in
`outputs/trajectories/<map>.json` (`core/trajectory.py`), created on first launch. Each
path carries a spawn pose plus sparse (`SPARSE_SPACING_M` = 25 m) and dense (8 m)
waypoint lists. A waypoint is "hit" when the vehicle centre comes within
`WAYPOINT_RADIUS` (8 m).

- `random_path` deals a new path each episode (training and human play).
- `dense_episodes=N` is a curriculum warm-up: dense waypoints for the first N
  episodes, then sparse.

---

## LiDAR

| Parameter | Value |
|---|---|
| Rays (azimuth bins) | 8 |
| Elevation bins | 1 (`lidar`) / 4 (`adv_lidar`) |
| Channels per ray | 1 (distance); extensible via `LIDAR_CHANNELS_PER_RAY` |
| Field of view | 360° full ring |
| Max range | 50 m |
| Vertical angle | 26.9° (`lidar`) / 20.0° (`adv_lidar`) |
| Vertical resolution | 32 layers (`lidar`) / 16 (`adv_lidar`) |
| Mount | Centred above the ego bbox roof from `vehicle.get_bbox()`; snapping disabled (`is_snapping_desired=False`, `is_force_inside_triangle=False`); falls back to `(0, 0, 2.4)` if bbox sampling fails |
| Direction | forward `(0, -1, 0)` in BeamNG coords |
| Self-filter | ego OBB + `LIDAR_SELF_MARGIN` (0.3 m) |
| Ground filter | `local_z > bbox_floor + LIDAR_GROUND_CLEARANCE` (0.3 m) |

`adv_lidar` trades vertical resolution for a narrower FOV so its 4 rows span useful
elevations instead of mostly sky and mostly asphalt. That lets the policy tell a wall
(fills every row) from a low object (bottom row only), which a single row cannot
represent.

Each cell holds the **nearest** point distance in its slice, normalized to `[0, 1]`
(0 = obstacle right there, 1 = clear). Binning spans `[-π, π]` from
`arctan2(local_y, local_x)`, where `+local_y` is the vehicle's left; rear points lie at
the wrap boundary and land in the first or last bin. Self-hits and ground returns are
filtered before binning.

> The mount is derived from the captured bbox, which matters for this car: it is much
> lower than the previous vehicle, so a hardcoded 2.4 m mount would float well above
> the roof.

---

## Episode lifecycle

```
reset()
  └─ (first call) _launch() → opens BeamNG, resolves/generates paths, loads scenario
  └─ (subsequent) teleport to a new random path, or scenario.restart()
  └─ bng.step(5) — settle physics
  └─ returns the initial observation

step(action)
  └─ controls_for(action) → vehicle.control(throttle, steering, brake)
  └─ bng.step(PHYSICS_STEPS_PER_ENV_STEP) — ~333 ms of sim time
  └─ _observe() — poll electrics, damage, and the perception sensor
  └─ _compute_reward() — racing reward + done flag
  └─ returns (obs, reward, done, info)

close(kill_sim=True)
  └─ remove lidar / camera / roads sensor (each bounded, so a wedged sim cannot hang)
  └─ bng.close() when kill_sim=True, bng.disconnect() when kill_sim=False
  └─ with kill_sim, wait until the port stops accepting connections
```

---

## The three environments

| Class | File | Shape |
|---|---|---|
| `BeamNGDrivingEnv` | `beamng.py` | One car. Training, human play. |
| `BeamNGMultiEnv` | `beamng_multi.py` | N cars, **one path each** (≥ 30 m apart), one shared physics step. The throughput trainer — no contact. |
| `BeamNGRaceEnv` | `beamng_race.py` | N cars, **one shared path**, starting grid, collisions, gap-aware reward. |

`BeamNGRaceEnv` extends `BeamNGMultiEnv`, which already owns the hard parts (one
scenario with N vehicles, per-slot sensors/observations/reward, bounded shutdown).
Racing overrides four things: shared path + `starting_grid` spawns, the gap term,
whole-field resets instead of per-vehicle ones, and a `realtime` mode for a human
entrant. Collisions need no work — BeamNG vehicles in one scenario always collide;
training only avoids contact because its paths are far apart.

A human entrant is a slot with `human=True` and no agent: it receives no control input,
gets no perception sensor (nothing would read it), but *is* still polled every tick,
because that is what advances its position and checkpoints for the rival's gap term.
`bng.switch_vehicle` gives it keyboard focus, best-effort across beamngpy versions.

---

## Launch options

BeamNG can be started headless (`HEADLESS=true` in `.env`) — no rendering window. Each
environment instance binds to a port (default `25252`), so several can run in parallel.
