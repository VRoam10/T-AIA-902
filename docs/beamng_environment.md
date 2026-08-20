# BeamNG Racing Environment

`environments/beamng.py` — `BeamNGDrivingEnv`

Gymnasium-style RL environment wrapping BeamNG.drive via beamngpy. One environment,
parameterized by two independent axes. Vehicle: **Cherrier Vivace Hillclimb**
(`vehicles/vivace/hillclimb_SQ.pc`) — 682 hp, 1420 kg, AWD, sequential gearbox; the same
car for every entrant, so a head-to-head result reflects the policies and not the
machinery.

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
kinematic(6) | perception(P) | hints(2*H) | [pitch, roll]? | road(6)? | wheel(4)?
```

| Index | Name | Raw source | Normalization |
|---|---|---|---|
| 0 | `speed` | `electrics.wheelspeed` (m/s) | `/ 50.0`, clipped to `[-1, 1]` |
| 1 | `steering` | `electrics.steering` | clipped to `[-1, 1]` (already normalized) |
| 2 | `heading_error` | angle between heading and next waypoint (rad) | `/ π`, clipped |
| 3 | `lateral_error` | signed cross-track distance from the guide polyline (m), + = left of travel | `/ 5.0`, clipped |
| 4 | `damage` | `damage_sensor.damage` (cumulative) | `/ 1000.0`, clipped to `[0, 1]` |
| 5 | `dist` | distance to the target checkpoint (m) | `/ CHECKPOINT_DIST_NORM_M`, clipped to `[0, 2]` |
| 6.. | perception | the sensor's block | see the table above |

The guide polyline is the spawn point followed by every checkpoint. Its projection
(`environments/beamng_path.project_onto_path`) is a function of the car's position
alone, so index 3 does not jump when the target checkpoint advances. The value this
replaced, `dist * sin(heading_err)`, was itself a function of indices 2 and 5 and so
carried no information; width and normalization are unchanged, so existing checkpoints
still load.

Optional tails, all off by default:

- `trajectory_hints=H` — vehicle-local `(forward, left)` of the next `H` waypoints,
  normalized over 100 m. **+2H** dims. Saturates on game tracks, whose checkpoints are
  far further apart than the 100 m norm.
- `body_orientation` — `[pitch, roll]` from the vehicle's forward/up vectors. **+2**.
- `road_info` — `[edge_left, edge_right, road_heading, curvature, ahead_fwd, ahead_left]`
  from a `RoadsSensor`. **+6**. Road-relative, so it does not care how far away the next
  checkpoint is.
- `wheel_info` — `[long_slip, slip_angle, abs_active, lat_g]` from Electrics, the vehicle
  state and a `GForces` sensor. **+4**. `lat_g`'s lateral axis is read as `gx2` then
  `gx`, on the reasoning that this project's vehicle frame has forward = -Y — a
  hypothesis not yet confirmed in-sim.

`beamng_spec.obs_size(sensor, hints, body_orientation, road_info, wheel_info)` is the
only place this arithmetic lives.

---

## Action space (`output: fixed`)

Throttle falls sharply as steering rises. The car — an AWD Cherrier Vivace Hillclimb —
still has far more power than grip, so the previous taxi-era table (0.4 throttle at 0.6
steering) spun it on most corner entries.

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
| Progress along the path (telescoping) | `+(progress_m - last_progress_m) × PROGRESS_COEF`, from a polyline projection that depends on position alone; falls back to straight-line closure to the checkpoint when no projection is supplied |
| Speed projected onto the path tangent | `+speed × cos(velocity_heading - tangent) × SPEED_ALIGN_COEF`; falls back to bearing-to-checkpoint alignment without a tangent |
| **Every step** | `-STEP_PENALTY` |
| Checkpoint reached | `+CHECKPOINT_BONUS` (flat) |
| Checkpoint reached, relative to the segment's own par | `+SEGMENT_TIME_BONUS × clip(1 - steps_since_checkpoint / par, 0, 1)`, where `par = steps_for_distance(segment_len_m, SEGMENT_PAR_SPEED_MS)` is derived from the segment actually driven |
| Path completed | `+FINISH_BONUS` (flat) |
| Path completed, relative to the path's own par | `+FINISH_TIME_BONUS × clip(1 - steps / par, 0, 1)`, where `par = steps_for_distance(path_length_m × laps, SEGMENT_PAR_SPEED_MS)` is derived from the distance actually covered |

The step penalty and the two time bonuses are what make *fast* beat *clean but slow*.
They matter more than they look: the progress and speed-alignment terms both integrate
to a constant over a fixed distance regardless of pace, so they shape *where* the car
goes, not *how quickly*.

The checkpoint bonus is deliberately **flat**. The reward this replaced paid
`100 × waypoint_idx`, which made late-path reward an order of magnitude larger than
early-path reward for identical driving — enough to destabilise the value function.

The segment-time bonus is **scale-free**: because `par` scales with the segment's own
length, `steps_since_checkpoint / par` is a pace ratio, so a near-perfect 25 m segment
and a near-perfect 1000 m segment pay the same instead of one dwarfing the other. This
replaced a *fixed* par (`SPARSE_SPACING_M`, 25 m), which is unmeetable on a game
track — 30 of the 44 shipped sprint/lap tracks have a gap over 300 m, and italy's
`highway1` averages 1064 m. `SEGMENT_TARGET_STEPS`/`SEGMENT_TIME_COEF` no longer exist.

**The finish bonus is scale-free for the same reason**, and was left absolute when the
segment term was rewritten. It paid `1.0 × (max_steps - steps)`, and `MAX_STEPS` is a
constant 5000 while the generated paths run from 65 m to 10.7 km — so "unused steps"
was a flat ~5000 for completing *anything*. Replaying the shipped caches through
`compute_race_reward` measured it at **5285 of a 6104-point episode** on
east_coast_usa's 75.5 m path (87% of the total, 81 reward per metre) against **1.3 per
metre** for driving gridmap_v2's 1767 m path well. The shortest path in the pool was
therefore the whole game, and the per-step driving signal (7–90) was invisible beside
one terminal spike — which is all a critic has to learn from. After the fix the same
episode pays 1113, of which 405 is the finish; per-metre rates across three path
lengths sit at 15.2 / 13.5 / 10.8, and the residual slope is just the flat
`FINISH_BONUS` amortised over a shorter path. `FINISH_TIME_COEF` no longer exists.

### Contact and track limits

| Condition | Delta | Ends episode? |
|---|---|---|
| New damage | `-damage_delta × DAMAGE_DELTA_COEF` | No |
| Hard hit (> `HARD_HIT_DAMAGE` in one step) | `-HARD_HIT_PENALTY` | **No** — rubbing wheels mid-overtake must not end a race |
| Cumulative damage ≥ `MAX_DAMAGE` | `-CRASH_PENALTY` | **Yes** |
| Nearest LiDAR bin < 0.2 / < 0.4 | graded proximity penalty | No |
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

Progress is measured by `beamng_path.project_onto_path`: arc length along the guide
polyline (the spawn point followed by every checkpoint) to the perpendicular
projection of the car's current position. Being a function of position alone rather
than the target checkpoint, it is already continuous across a checkpoint
transition — which a telescoping term requires — and following the road round a bend
can never read as backward progress. There is no lap offset: a term that added
`laps_done * path_length(polyline)` was tried and removed, because `waypoint_idx`
wraps at the end of every run (not only a second lap), so it added a full path length
to progress at the finish even at `laps=1`. A real lap counter needs a lap-crossing
event, not an index; `laps != 1` still raises.

The projection is **seeded** with the previous step's arc length (`near_m`), which
restricts the search to `SEARCH_WINDOW_M` (60 m) either side of it. Without a seed it
takes the globally nearest segment, so a path that passes close to itself projects onto
whichever pass happens to be nearer — and on a path that closes on its own start, a car
sitting at the line reads either arc 0 or a full lap. Since the term pays
`PROGRESS_COEF ×` the *change*, that ambiguity was worth 3 × the path length in one
step. Measured across the cached paths, the worst single-step jump a car circling near
its spawn could produce fell from **1767 m to 0.8 m** on gridmap_v2 path 0 (the default
training path), 1363.7 → 0.8 on path 3 and 185.7 → 1.9 on west_coast_usa path 10 —
removing 5299, 4089 and 551 points of free reward. Every other cached path projects
identically seeded or not, so the window does not perturb ordinary geometry. The window
is wider than one step at the car's top speed (81 m/s × 0.333 s = 27 m) so a fast car
cannot outrun its own seed; a seed that leaves the car further from the windowed stretch
than the window is wide is treated as stale and the search falls back to global.

---

## Waypoints and track

Paths are generated per map from the road network and cached in
`outputs/trajectories/<map>.json` (`core/trajectory.py`), created on first launch. Each
path carries a spawn pose plus sparse (`SPARSE_SPACING_M` = 25 m) and dense
(`DENSE_SPACING_M` = 8 m) waypoint lists.

A waypoint is "reached" when the car's **arc length along the guide line** passes the
arc length that waypoint sits at (`beamng_path.waypoint_arcs`). Proximity used to be
the rule — `dist < WAYPOINT_RADIUS`, 8 m — and three constants made it free rather than
earned: `SPAWN_CLEARANCE_M` (2 m) puts checkpoint 0 inside the ring before the car
moves, `DENSE_SPACING_M` equals the radius so the next checkpoint on the dense chain is
already inside the current one's, and the radius is wide enough that a settling car
drifts through one. Measured: a car **parked** at east_coast_usa's spawn banked +56 over
12 steps of nothing (it now loses 6, the step penalty), and 8 of the 44 cached paths
have *every* dense gap under the radius. The same rule also failed in the opposite
direction at speed — above ~24 m/s a car flew past 8 m markers between control steps
without ever sampling inside one, stalling the chain permanently: west_coast_usa path 4
could not be completed in 90 steps and now finishes in 25. Several checkpoints can fall
inside one step, and all of them count (the flat bonus is still paid once per step;
the distance is already paid for by the progress term).

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
