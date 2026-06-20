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

Fix for now: removed the "per-wheel road position" (`wheel_terrain`) option from
the training menus so it can't be enabled and trigger the freeze. The feature
code is still present (eval/human-play and tests), just not offered for training.
