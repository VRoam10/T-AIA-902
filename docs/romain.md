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
