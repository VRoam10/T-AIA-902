# On Beamng Environment

First Issue:
The algorithm don't see what's around them.
I added lidar to be able to scan what's around them.
Not confident on the calcul of the closer point with this, i let claude generate this code.

Second Issue:
The code that was issued by claude didn't seem to work. So i've the lidar to the beamng human play and look at the value sent to the algorithm.
And my prediction revealed to be corrected. Without moving, the value of the closer point keep changing, which indicated that there were some issues.

Third Issue:
The taxi light were blocking the lidar radar system and the fov where to wide which made the ai see less.
![alt text](image.png)

Fourth Issue:
the ia goes in reverse even with all this fix for no apparent cause.
```
self.vehicle.control(throttle=0.0, steering=0.0, brake=1.0)
```
Cause the ia to go in reverse, since beamng Automatic shifting makes it that if you are standing still and tries to brake, the reverse engages.

Fifth issue:
The lidar is directly attached to the car, with how bad the suspension of it, it will skew the result of the lidar. The fix is to give the pitch and roll of the car to the algo to better understand the position of the car.
This is now an opt-in observation: `body_orientation` (pitch + roll) and
`wheel_terrain` (per-wheel road position) can be toggled per environment and
per vehicle, instead of being hardcoded into a single env.

Sixth issue:
Last follow-up i was asked how we can see if a new changement actually improved the algorithm. For this, i've added a multi agent and multiple environment as to test different combination at the same time. It's not a perfect solution as the vehicule as collision as well as lidar rays (which doesn't have a fix).

Follow-up to the sixth issue: trajectory generation now emits one road-snapped
path per map teleport point (`SpawnSphere`). Multi-agent training assigns one
path per vehicle in a different part of the map, so vehicles no longer share a
start line or collide. If more vehicles than paths are requested, the session
errors out rather than doubling up.

Seventh issue:
The `wheel_terrain` option (per-wheel road position, fed by a beamngpy
`RoadsSensor`) hard-freezes BeamNG during training on large maps (e.g.
`west_coast_usa`), while `gridmap` is fine. Symptom: the whole sim hangs on an
episode reset, BeamNG keeps eating CPU + memory while Python sits idle, and
nothing is written to `beamng.log`.

Root cause, pinned down with a `faulthandler` stack dump (`Timeout (0:00:30)!`):
the main thread is blocked in `roads_sensor.poll()`, called from `observe()`
inside `reset_vehicle()` right after the vehicle teleports to a new random path.
`reset_vehicle` polls the sensor with **no physics step** after the teleport
(unlike `reset_all`, which steps first). On a road-dense map the RoadsSensor's
game-engine side grinds trying to map the road network at a freshly-teleported,
not-yet-settled position and never answers, so Python blocks forever in the
socket `recv`. The LiDAR poll in the same `observe()` runs first and completes —
it is specifically the RoadsSensor. gridmap has a trivial road network, so it
answers instantly and never hangs.

It is NOT the teleport itself (human quick-travel play works), NOT a GPU TDR (no
D3D11 error in the log), and NOT CUDA (reproduced with the agents on CPU).

Fix: the sensor is now only ever polled through a gate (`_road_pollable`) that is
closed the instant a teleport or scenario load repositions the vehicle, and reopened
only once the simulation has actually advanced since — `reset()`/`step()`'s
`_advance()` in training, and (since nothing there ever calls `_advance()` again)
an explicit open right after `resume()` in the two realtime paths, human play and a
realtime race. A `RoadsSensor` poll while the gate is shut returns the neutral
reading instead of touching beamngpy. The option is back in the menus, renamed
`road_info`, and it now carries six features instead of two:
`[edge_left, edge_right, road_heading, curvature, ahead_fwd, ahead_left]` — the extra
four (heading, curvature, look-ahead) came along for free once the freeze was fixed
properly instead of worked around.

Eighth issue:
Multi-path generation could emit paths with too few checkpoints. Each teleport
point snaps to its nearest road and the resulting sub-polyline is resampled at a
fixed 25 m spacing, so a teleport that lands on a short road produced only 1-2
checkpoints after the spawn-clearance drop — not enough for the agent to learn a
trajectory.

First attempt (rejected): repack short paths with a tighter spacing to hit
`MIN_CHECKPOINTS`. This crammed checkpoints a few metres apart on short roads —
bad for training.

Fix: extend the path itself instead of shrinking the spacing. The snapped road
is grown forward through connected roads (junctions within `ROAD_CONNECT_M`,
following the current heading) up to `MIN_PATH_LENGTH_M`, then resampled at the
normal 25 m spacing. So a teleport on a short road still yields a long path with
well-spaced checkpoints. Genuinely isolated dead-end roads keep the default
spacing and simply carry fewer checkpoints rather than crammed ones.

Ninth issue:
Game tracks (quickrace sprint/lap) space their checkpoints far more sparsely than the
generated paths' 25 m — some gaps run past a kilometre (italy's `highway1` averages
1064 m). Two of the reward's pace terms had baked in the 25 m assumption and broke on
those tracks:

- progress was straight-line distance closed to the next checkpoint, which punishes
  following the road: rounding a bend can increase the straight-line distance to a
  checkpoint that far away, so a car driving the racing line correctly reads as going
  backward.
- the checkpoint-time bonus compared `steps_since_checkpoint` against a par fixed at
  `SPARSE_SPACING_M` (25 m) worth of steps. On a 1000 m segment that par is unmeetable,
  so the bonus floors at zero for the whole segment no matter how well it is driven.

Fix: progress and the speed-alignment term now come from projecting the car's position
onto a guide polyline (spawn point + every checkpoint) — `beamng_path.project_onto_path`,
a function of position alone, so it cannot read a bend as backward progress and does
not care how far away the next checkpoint is. The checkpoint-time bonus's par is now
derived from the segment actually being driven
(`steps_for_distance(segment_len_m, SEGMENT_PAR_SPEED_MS)`), so
`steps_since_checkpoint / par` is a pace ratio: a 25 m segment and a 1000 m segment
driven at the same multiple of par speed pay the same, instead of one being trivial
and the other impossible.

Also dropped, discovered while wiring this up: the plan had progress add
`laps_completed * path_length` for closed circuits. `waypoint_idx` wraps at the end of
every run, not only a second lap, so that term added a full path length to progress at
the finish of every episode — even at `laps=1`. A real lap counter needs a
lap-crossing event, not an index division, so it stayed out; `laps != 1` still raises.

Tenth issue: training reported ~6000 reward per episode while the car only span
and crashed. Diagnosed offline, by replaying the cached paths through
`compute_race_reward` with no simulator, and reproduced exactly: 14 steps,
+6122, against the reported 14 steps and 6072-6080.

The reward was 87% participation trophy. `FINISH_BONUS + (MAX_STEPS - steps) x
FINISH_TIME_COEF` with `MAX_STEPS` a constant 5000 pays ~5000 for completing
*anything*, and the generated paths run from 65 m to 10.7 km — so finishing
east_coast_usa's 75.5 m path paid 5285 of a 6104-point episode (81 reward per
metre) against 1.3 per metre for driving gridmap_v2's 1767 m path well. A critic
learning from one terminal spike 700x the size of the per-step signal has nothing
left to distinguish a clean lap from donuts, which is what the training showed.
The finish's time bonus is now relative to par over the distance actually
covered, exactly like the segment bonus it was rewritten alongside — the same
class of error, missed in the same commit.

Three more, all measured on the shipped caches rather than argued:

- a car **parked** at the spawn banked +56 over 12 steps. `SPAWN_CLEARANCE_M`
  (2 m) puts checkpoint 0 inside the old 8 m arrival radius before the car moves,
  and `DENSE_SPACING_M` (8 m) equals that radius, so every next checkpoint on the
  dense chain was already inside the current one's — 100% of dense gaps on 8 of
  44 paths. It now loses 6 (the step penalty).
- the same proximity rule failed the other way at racing speed: above ~24 m/s the
  car passes 8 m markers between control steps without ever sampling inside one,
  and the chain stalls for good. west_coast_usa path 4 could not be finished in
  90 steps; it now finishes in 25. So the old gate handed out free checkpoints
  when slow and refused earned ones when fast.
- a path that closes on its own start made the progress term pay 3 x the path
  length for standing still, because the projection took the globally nearest
  segment: at the start/finish line that is either arc 0 or a full lap. Worst
  single-step jump for a car circling near its spawn, over all cached paths:
  1767 m -> 0.8 m on gridmap_v2 path 0 (the default training path), 1363.7 -> 0.8
  on path 3, 185.7 -> 1.9 on west_coast_usa path 10, removing 5299, 4089 and 551
  points of free reward. The projection is now seeded with the previous step's
  arc length. Every other cached path projects identically seeded or not.

Why none of this was visible: the finish sets `waypoint_idx = 0`, and every
caller builds its metrics *after* the reward — so the checkpoint count read 0 on
exactly the episodes that completed a path. That is why the training plot's
checkpoint panel was blank while the car was finishing a 75 m path every 14
steps. The count is now carried separately (`checkpoints_reached`).

Residual, not chased: italy path 2 has a genuine 42 m arc discontinuity near its
spawn (worth ~127 points), which is real geometry rather than a self-approach
ambiguity — the seeded search returns the same answer.
