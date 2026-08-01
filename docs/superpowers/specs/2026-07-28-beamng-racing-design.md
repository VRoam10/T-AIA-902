# BeamNG racing side-quest

**Date:** 2026-07-28
**Status:** Approved (design)

## Problem

The T-AIA-902 deliverable is finished — the Taxi/DQN study is written up in
`docs/reports/2026-07-03-taxi-qlearning-vs-dqn/` and its outputs are synced to `web/`.
What remains is a pipeline built to be *general*, and that generality is now pure cost
against a new goal: **drive fast, and beat another car to the line.**

Four things get in the way:

1. **Seven menu modes.** Evaluate, benchmark and generate-trajectories exist to serve
   the finished study, as does Taxi in every algorithm's `compatible_envs`.
2. **Four BeamNG env classes crossing two independent concerns.** `beamng`,
   `beamng_lidar`, `beamng_continuous` and `beamng_camera` mix *which sensor feeds the
   observation* with *whether the action head is discrete or continuous*. `beamng` and
   `beamng_continuous` produce identical observations and differ only in `step()`;
   `BeamNGCameraEnv` re-implements `_load_scenario` and `_observe` wholesale rather
   than overriding a perception hook.
3. **The reward measures arrival, not pace.** `compute_merged_reward` pays
   `CHECKPOINT_BONUS_PER_IDX * waypoint_idx` per checkpoint and has no time term at
   all, so a slow clean run outscores a fast one over the same path. Nothing in the
   reward refers to another vehicle.
4. **Four selectable road cars**, none of them a race car.

## Goal

- **Four modes**: training, multi-agent training, human play, course mode.
- **One parameterized BeamNG env** on a visible `sensor` axis, with the output axis
  derived from the algorithm.
- **A racing reward**: time pressure, a finish bonus that scales with how much of the
  step budget is left, and an optional gap term that rewards being the leader.
- **One race car**, so a head-to-head race is symmetric.
- **Course mode**: two cars on the same track, colliding, either agent-vs-agent or
  agent-vs-human, with an exhibition mode and a keep-learning mode.

## Locked decisions

| Decision | Choice |
|---|---|
| Race format | Point-to-point sprint over one generated path. A `laps` param exists but must be 1 until closed-circuit generation lands |
| Output axis | **Derived** from the algorithm (`dqn`/`dqn_per` → fixed, `ddpg`/`td3` → continuous). No Output dropdown |
| Sensor axis | **Visible** dropdown: `lidar` (8 bins) / `adv_lidar` (4×8 grid) / `camera` (16×16) |
| Deletion scope | Delete Taxi + `q_learning`. Keep `benchmarks/`, `run_evaluate`, `run_trajectory` importable but off the menu |
| Race car | `model="sunburst2"`, `part_config="vehicles/sunburst2/trackday_M.pc"` |
| Course mode | A `learning` toggle: off = frozen policies + winner/gap report, on = agents keep updating with the leader reward |

## The two axes

The `sensor` axis chooses the perception block; the `output` axis chooses the action
head. They are genuinely independent, which is why four class names could never
express them cleanly.

```
sensor:  lidar (8)  |  adv_lidar (4x8 = 32)  |  camera (16x16 = 256)
output:  fixed (7 discrete actions)  |  continuous (throttle, steering, brake)
```

`output` is not a user-facing choice because the algorithm already determines it: a DQN
head cannot emit continuous controls, and DDPG/TD3 emit nothing else. Exposing both
would be two fields encoding one decision.

New `environments/beamng_spec.py` is the single source of truth for both axes and every
size derived from them:

```python
SENSORS = ("lidar", "adv_lidar", "camera")
OUTPUTS = ("fixed", "continuous")

PERCEPTION_FEATURES = {"lidar": 8, "adv_lidar": 32, "camera": 256}
KINEMATIC_FEATURES  = 6      # speed, steering, heading_err, lateral_err, damage, dist
LIDAR_GEOMETRY = {           # v_bins / vert_res / vert_angle per lidar sensor
    "lidar":     {"v_bins": 1, "vert_res": 32, "vert_angle": 26.9},
    "adv_lidar": {"v_bins": 4, "vert_res": 16, "vert_angle": 20.0},
}
FIXED_ALGOS      = ("dqn", "dqn_per")
CONTINUOUS_ALGOS = ("ddpg", "td3")

RACE_CAR = {"model": "sunburst2", "licence": "RACE", "color": "White",
            "part_config": "vehicles/sunburst2/trackday_M.pc"}

def output_for_algo(algo) -> str          # "fixed" | "continuous"
def obs_size(sensor, trajectory_hints=0, body_orientation=False,
             wheel_terrain=False) -> int
def action_size(output) -> int            # 7 | 3
```

This replaces **three** copies of the same arithmetic: the `N_STATES` class attributes
in `beamng.py`, `slot_n_states` + `_PERCEPTION_FEATURES` in `beamng_multi.py`, and
`_beamng_extra_dims` in `pipeline_actions.py`.

### Observation layout is unchanged

`obs_size` reproduces today's layouts exactly, so the observation contract does not
move and the reward's `obs[:6]` / `obs[6:6+n]` slicing keeps working:

| sensor | today's env | obs length |
|---|---|---|
| `lidar` | `beamng` / `beamng_continuous` | 14 |
| `adv_lidar` | `beamng_lidar` | 38 |
| `camera` | `beamng_camera` | 262 |

```
kinematic(6) | perception(P) | hints(2·H) | [pitch, roll]? | [left, right]?
```

### Registry

`environments/__init__.py` drops `taxi` and collapses the four BeamNG registrations
into a single `beamng` entry whose metadata is `{"state_type": "continuous"}` only.
`n_states` / `n_actions` leave the metadata entirely — `build_agent` calls `obs_size`
and `action_size`. Since Algorithm + Sensor now determine the env completely, the
Environment dropdown disappears from the TUI.

## Env collapse

`BeamNGLidarEnv`, `BeamNGContinuousEnv` and `BeamNGCameraEnv` are deleted.
`BeamNGDrivingEnv.__init__` gains `sensor="lidar"` and `output="fixed"`:

- `_load_scenario` attaches a LiDAR sized by `LIDAR_GEOMETRY[sensor]` **or** a
  dashcam — one scenario builder, not two copies.
- `_observe` calls one `_perceive()` hook returning the sensor's feature block; the
  kinematic head, waypoint hints and extra features are shared. (They are already
  identical today, duplicated verbatim in `BeamNGCameraEnv._observe`.)
- `step()` branches on `self.output`, replacing `BeamNGContinuousEnv.step`.
- The three near-identical `human_play` / `human_play_lidar` / `human_play_camera`
  loops become one `human_play()`, with the LiDAR filtering diagnostics conditional on
  the sensor being a lidar.

`reward_mode` is deleted from the constructor and from `VehicleSlot`. There was one
reward already (`compute_merged_reward` ignores the flag); now there is one racing
reward, so the `"default"` vs `"ddpg"` distinction has nothing left to select.

### Supporting extraction

`environments/beamng.py` is 1421 lines and the collapse touches most of it. The
beamngpy **sensor construction** moves to `environments/beamng_sensors.py`:
`_lidar_config`, `_lidar_creation_kwargs`, `_resolve_lidar_mount_pos`,
`_cache_ego_local_bbox`, `_process_lidar`, `_process_camera`. `beamng_multi.py`
duplicates every one of these in `_create_slot_sensor` / `_lidar_config_for` /
`_perceive`; both callers move onto the extracted helpers.

`beamng_geometry.py` keeps its role — pure math with no beamngpy import
(`process_lidar`, `ego_local_extents_from_bbox`, `body_orientation_features`). The new
module holds only the sensor *construction* that currently exists twice.

## One car

`VEHICLES` keeps its dict shape but holds the single `RACE_CAR` entry, so per-vehicle
plumbing (colour, part config) is untouched. `BEAMNG_VEHICLES`,
`BeamNGOptions.vehicle_id`, `VehicleSlot.vehicle_id`, `Catalog.beamng_vehicles`,
`ctx.vehicleIds` and every "Vehicle" form field are removed. Multi and race slots now
differ only by `color`.

### Action table retune

The `ACTIONS` table was tuned for a burnside taxi: full throttle straight, `0.4`
throttle at `0.6` steering. On a mid-engine RWD race car that is a spin on most corner
entries. Same seven entries (the discrete head size is unchanged), less throttle where
the wheel is turned:

```python
ACTIONS = [
    {"throttle": 0.0,  "steering":  0.0,  "brake": 0.0},   # 0 coast
    {"throttle": 1.0,  "steering":  0.0,  "brake": 0.0},   # 1 full throttle straight
    {"throttle": 0.6,  "steering": -0.25, "brake": 0.0},   # 2 power-on slight left
    {"throttle": 0.6,  "steering":  0.25, "brake": 0.0},   # 3 power-on slight right
    {"throttle": 0.0,  "steering":  0.0,  "brake": 1.0},   # 4 brake
    {"throttle": 0.15, "steering": -0.55, "brake": 0.0},   # 5 lift + sharp left
    {"throttle": 0.15, "steering":  0.55, "brake": 0.0},   # 6 lift + sharp right
]
```

## Racing reward

`environments/beamng_reward.py` keeps its shape — one function called by the single,
multi and race envs, so all three reward identical behaviour — but changes what it
rewards. `compute_merged_reward` becomes `compute_race_reward` with two new optional
arguments, `rival_progress_m` and `last_rival_progress_m`, defaulting to `None` so
single-car callers are unaffected.

**Kept:** telescoping progress toward the waypoint (`×3`), speed projected on the
target direction (`×3`), the LiDAR obstacle-proximity penalty, the graded off-track
warn penalty and the hard off-track reset, `MAX_STEPS` termination, and the
`INVULN_GRACE_STEPS` damage-immunity window on a checkpoint hit.

**Changed:**

1. **Time pressure.** `STEP_PENALTY` charged every step. This is the term that makes
   fast beat clean-but-slow; today nothing does. It subsumes `STATIONARY_PENALTY`,
   which is deleted — a stopped car already earns no progress and pays the step
   penalty, so a separate speed threshold is redundant.
2. **Flat checkpoint bonus + segment-time bonus.** `CHECKPOINT_BONUS_PER_IDX *
   waypoint_idx` becomes a flat `CHECKPOINT_BONUS` plus
   `max(0, SEGMENT_TARGET_STEPS - steps_since_checkpoint) * SEGMENT_TIME_COEF`. The
   growing bonus made late-path reward an order of magnitude larger than early-path
   reward for *identical* driving, which destabilises the value function; the
   segment-time bonus puts the incentive where it belongs — reach **each** checkpoint
   sooner. Adds one caller-side field, `steps_since_checkpoint`, threaded through
   `RewardOutcome` alongside the existing `invuln_steps`.
3. **Finish bonus.** On completing the path, `FINISH_BONUS + (max_steps - steps) *
   FINISH_TIME_COEF` — the strongest single "go fast" signal, replacing the flat
   `LAP_BONUS`.
4. **Softer contact.** `DAMAGE_DELTA_COEF` is halved and the hard-hit `done=True` is
   dropped, kept only for `damage >= MAX_DAMAGE`. Rubbing wheels mid-overtake must not
   end the race; writing the car off still must.
5. **Leader term (race only).** When `rival_progress_m` is supplied:

   ```
   GAP_COEF * ((mine - rival) - (last_mine - last_rival))
   ```

   It telescopes, so the episode total is `GAP_COEF × final gap in metres` — one clean
   signal that rewards overtaking *and* defending, with no per-step position
   bookkeeping. Plus `WIN_BONUS` / `LOSE_PENALTY` settled when a car finishes first.

### Track progress in metres

The gap term needs a scalar position along the shared path. `core/trajectory.py`
already has `_segment_length` and `_project_onto_polyline`; both are promoted to public
names and a new `track_progress_m(waypoints, waypoint_idx, pos)` in
`beamng_geometry.py` returns the cumulative arc length up to `waypoint_idx` plus the
projection onto the current segment. Pure math, unit-testable without a simulator.

## Course mode

`environments/beamng_race.py` — `BeamNGRaceEnv`, extending `BeamNGMultiEnv`. That class
already owns everything hard: one scenario with N vehicles, one physics step for all,
per-slot sensors, observations, reward and markers, and a bounded shutdown. Racing
changes four things.

**Shared path + starting grid.** `_assign_paths` today gives each slot a *distinct*
path and raises when slots outnumber paths. The race override puts every slot on the
same path and offsets the spawns via a new pure helper
`starting_grid(spawn_pos, spawn_rot, n, lateral_m=3.0, stagger_m=6.0)` in
`beamng_geometry.py` — perpendicular plus longitudinal offsets from the spawn heading,
so two cars never spawn inside each other.

**Collisions.** Nothing to enable. BeamNG vehicles in one scenario collide physically;
multi-agent *training* only avoids contact because its paths are ≥ `MIN_PATH_SEPARATION_M`
apart. The damage terms already price contact.

**Gap-aware reward.** Each step, compute every slot's `track_progress_m` and pass the
best *other* slot's progress as `rival_progress_m`. The single-car path passes `None`.

**Whole-race reset.** Training resets one finished vehicle while the others drive on
(`reset_vehicle`). A race resets **both together** via `reset_all()` when `race_over()`
— any car finished, or all cars done (crashed / step cap). Per-slot target markers are
skipped in race mode: two coloured spheres on one waypoint overlap, and
`add_checkpoints` already draws the rings both cars aim at.

### Human competitor

A slot with `agent=None, human=True` receives no `apply_action`; the player drives it
from the keyboard. That vehicle needs input/camera focus — `bng.switch_vehicle(...)`
(**exact beamngpy name to verify**; may be `bng.vehicles.switch_vehicle`).

A human race cannot run in lockstep, so `BeamNGRaceEnv(realtime=...)`: lockstep
`bng.step(10)` for agent-vs-agent (deterministic and fast), `bng.resume()` plus a
~10 Hz poll loop for agent-vs-human — the free-running pattern `human_play` already
uses.

### `core/race_runner.py` — `RaceRunner`

Per tick: collect actions from agent slots → apply → advance (step or sleep) → observe
→ reward with the gap term → if `learning`, `agent.update(...)` and periodic
`agent.save(...)`. On `race_over()`, record `{winner, margin_m, steps, finished}` per
racer and `reset_all()`. Honours `stop_requested()` at the tick boundary like
`MultiAgentRunner`, so Esc in the TUI still closes the simulator cleanly.

`MultiAgentRunner` is untouched — multi-agent training keeps its role as the throughput
trainer: N agents, N separate paths, no contact.

## Plumbing

`core/pipeline_actions.py`:

- `BeamNGOptions` drops `vehicle_id`, gains `sensor: str = "lidar"`.
- `build_agent` sizes from `obs_size` / `action_size`; the metadata lookup and
  `_beamng_extra_dims` go away.
- `run_train` / `run_evaluate` / `_benchmark_env` drop the `reward_mode=algo_name`
  branch.
- New `RacerSpec` / `CourseRequest` / `run_course`:

```python
@dataclass
class RacerSpec:
    algo: str
    sensor: str = "lidar"
    model_path: str = ""
    color: str = "White"
    trajectory_hints: int = 0
    body_orientation: bool = False

@dataclass
class CourseRequest:
    map_name: str
    racers: list[RacerSpec]   # 1 when opponent == "human", else 2
    opponent: str = "algo"    # "algo" | "human"
    laps: int = 1             # > 1 rejected until closed circuits exist
    races: int = 1
    learning: bool = False
```

`core/tui_backend.py` adds `_cmd_course` and `"course"` to `_COMMANDS`. The `evaluate`,
`benchmark` and `trajectory` commands stay registered — importable, just unexposed.

TUI (`tui/src/`): `WorkflowId` → `train | multi_train | human_play | course | quit`;
`BeamNGFields` swaps `vehicle_id` for `sensor`; `trainSavePath` becomes
`outputs/{algo}_{sensor}{suffix}.pth`; the evaluate/benchmark/trajectory forms and
payload builders are deleted along with `runTrajectorySequence`, `trajectoryCancelled`
and `vehicleIds`. `buildCourseForm` uses **two fixed racer blocks** rather than the
dynamic add/remove list `buildMultiTrainForm` uses — a race is always exactly two
entrants.

## Checkpoints

Old `outputs/*.pth` files are not migrated. Even where `n_states` still matches, they
were trained on the burnside's dynamics; on a RWD race car they are worthless.
Training starts fresh. Trajectory caches in `outputs/trajectories/` stay valid — road
geometry is car-independent and no quaternion convention changes here.

## Testing

- `beamng_spec`: `obs_size` reproduces 14 / 38 / 262 and each option's dims;
  `output_for_algo` covers all four algorithms and rejects unknown names.
- `track_progress_m` and `starting_grid`: pure math, no simulator — monotonic progress
  along a polyline, and grid slots that are pairwise separated and correctly oriented.
- Reward: a fast traversal scores above a slow traversal of the same path; the gap term
  telescopes to `GAP_COEF × final gap`; `rival_progress_m=None` leaves single-car
  reward unchanged; a hard hit no longer ends the episode but `MAX_DAMAGE` still does.
- `BeamNGRaceEnv` / `RaceRunner`: the fake-`bng` pattern from `tests/test_beamng_multi.py`
  — shared path assignment, `race_over` conditions, whole-race reset, and that a human
  slot receives no `apply_action`.
- Taxi removal: `tests/test_benchmarks.py` and `tests/test_seeding.py` use the *real*
  Taxi env as a cheap fixture. Port them onto a small dummy discrete env rather than
  deleting the coverage.
- Regression: flag-off observation lengths equal the pre-change lengths for every
  sensor.

## Out of scope

- Closed-circuit track generation and real lap counting (`laps` stays 1).
- More than two entrants in a race.
- Racing lines, braking points, or any curvature lookahead beyond the existing
  `trajectory_hints`.
- Tyre/fuel/damage models beyond the existing damage sensor.
