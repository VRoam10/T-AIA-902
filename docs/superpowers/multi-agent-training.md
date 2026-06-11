# Multi-Agent BeamNG Training

Train several RL algorithms **at the same time** in one BeamNG scenario. Each
vehicle is driven by its own agent/checkpoint, runs its own independent
episodes, and is drawn in its own colour with a matching target-waypoint
marker. A single physics step advances every vehicle, so N algorithms learn in
parallel on the same trajectory.

Companion design docs:
- Spec: [specs/2026-06-05-beamng-multi-agent-training-design.md](specs/2026-06-05-beamng-multi-agent-training-design.md)
- Plan: [plans/2026-06-05-beamng-multi-agent-training.md](plans/2026-06-05-beamng-multi-agent-training.md)

---

## Quick start

```bash
python main.py
```

Then choose **`6. Multi-agent training (BeamNG)`** and follow the prompts:

1. **Map** — pick from `gridmap_v2`, `italy`, `west_coast_usa`. (Pre-warm its
   trajectory first with menu option **5** if it has no cache.)
2. **Add vehicles**, one at a time:
   - **Algorithm** — `dqn`, `ddpg`, or `td3`.
   - **Environment** — the registered BeamNG envs compatible with that
     algorithm (e.g. `beamng`, `beamng_lidar`, `beamng_continuous`,
     `beamng_camera`, and their `_predicted` variants). Each car can run a
     **different** environment, so its perception, observation size, and
     waypoint hints come from its own env.
   - **Vehicle model** — Burnside (Taxi), Gavril T-Series, or Ibishu Pigeon.
   - **Model save path** — default
     `outputs/multi-agents/<algo>_<env>_<index>.pth`. If a checkpoint already
     exists you can **C**ontinue (resume) or **R**eset it.
   - A colour is auto-assigned by index (Yellow, Red, Blue, Green, Orange,
     White, Black).
   - Repeat until you answer "no" to *Add another vehicle?*.
3. **Episodes per agent** and **Time limit (minutes)** — training stops when
   **either** every agent reaches the episode count **or** the wall-clock limit
   is hit, whichever comes first (`0` minutes = no time limit).

Each vehicle starts in its own lane on a side-by-side starting grid, drives the
shared trajectory, and respawns to its lane when its episode ends — while the
others keep going.

---

## What you see in-game

- **N coloured cars** lined up abreast at the start, each in its own lane.
- **One coloured sphere per car** hovering over the waypoint that car is
  currently chasing; the sphere matches the car's paint and advances as that
  car hits its checkpoints. (This is the per-vehicle version of the
  single-agent env's green target marker.)
- The shared **checkpoint rings** for every waypoint, as in single-agent runs.

---

## Architecture

| File | Responsibility |
|------|----------------|
| [environments/beamng_geometry.py](../../environments/beamng_geometry.py) | Pure, stateless LiDAR geometry (`LidarConfig`, `world_to_local`, `lidar_keep_mask`, `process_lidar`, `ego_local_extents_from_bbox`). Shared by both the single- and multi-vehicle envs — one implementation, no duplication. |
| [environments/beamng_camera_util.py](../../environments/beamng_camera_util.py) | Pure camera-frame processing (`process_camera_frame`): grayscale + resize + normalize. Shared by the single-vehicle camera env and the multi-env. |
| [environments/beamng.py](../../environments/beamng.py) | The single-vehicle env. Its LiDAR and camera math now delegate to the shared helpers. |
| [environments/beamng_multi.py](../../environments/beamng_multi.py) | `VehicleSlot` (per-vehicle state) + `BeamNGMultiEnv` (the shared scene). Also `build_slots()` and the colour helpers. |
| [core/multi_runner.py](../../core/multi_runner.py) | `MultiAgentRunner` — the parallel training loop. |
| [core/cli.py](../../core/cli.py) | Menu option 6 + the testable `build_multi_session()` builder. |

### `VehicleSlot`

A dataclass holding **everything** for one vehicle so two agents never share
mutable state:

- **Identity/config:** `name` (`ego_0`…), `color`, `vehicle_id`, `agent`,
  `reward_mode`, `action_space`, `save_path`.
- **Env profile:** `env_name`, `perception` (`lidar` / `lidar_grid` /
  `camera`), `trajectory_hints`, `n_states` — derived from the car's chosen env.
- **Sensors:** its own `vehicle`, `electrics`, `damage_sensor`, and either a
  `lidar` or a `camera` depending on perception.
- **Grid pose:** `spawn_pos`, `spawn_rot` (its assigned lane on the grid).
- **Episode state:** `waypoint_idx`, distances, `checkpoint_hit`, `steps`,
  `ego_local_extents`, `last_obs`, `active_marker_id`, …
- **Training stats:** `episode`, `reward_history`, `steps_history`.

`reset_episode()` zeroes the running episode state but keeps the episode
counter and histories.

### `BeamNGMultiEnv`

Owns the shared `bng` connection, `scenario`, `trajectory`, and `waypoints`.
It reuses the single-vehicle env's constants (the discrete `ACTIONS` table,
LiDAR config, waypoint/reward thresholds) but keeps all mutable state in the
per-vehicle slots. Key methods:

- `launch()` / `_load_scenario()` — build the scenario, add every vehicle on
  the starting grid, add checkpoint rings, then create one LiDAR per vehicle.
- `reset_all()` — teleport every vehicle to its grid lane, zero episode state,
  prime each `last_obs`.
- `observe(slot)` — poll that slot's sensors → its observation vector.
- `apply_action(slot, action)` — map a discrete or continuous action to
  `vehicle.control()` (no physics step).
- `step_physics()` — a single `bng.step(10)` advancing **all** vehicles.
- `compute_reward(slot, obs)` — `(reward, done)` using that slot's reward mode.
- `reset_vehicle(slot)` — teleport one finished vehicle back to its lane.
- `_update_slot_marker(slot)` — draw/refresh that slot's coloured sphere.
- `close()` — remove every LiDAR, close `bng`.

### `MultiAgentRunner.train(env, n_episodes, time_limit, save_every)`

Each tick:

1. For every still-active slot: `action = agent.select_action(slot.last_obs)`,
   `env.apply_action(slot, action)`.
2. `env.step_physics()` — once for all.
3. For every active slot: `next_obs = env.observe(slot)`,
   `reward, done = env.compute_reward(slot, next_obs)`,
   `agent.update(...)`, accumulate, `slot.steps += 1`.
4. On a slot's `done`: record reward/steps, `decay_epsilon()`, bump its episode
   counter, checkpoint every `save_every`, then `env.reset_vehicle(slot)` so it
   starts its next episode while the others keep driving.

At the end every agent is saved and a per-agent reward/steps plot is written.

---

## Per-car environment & algorithm

Every car picks **both** an algorithm and an environment, and they can differ
from car to car. The environment determines perception (and therefore the
observation vector and its size); the algorithm determines the action space and
output dimensionality.

Each observation is `6` kinematic floats + a perception block + `2 ×
trajectory_hints`:

| Environment | Perception block | n_states | Sensor |
|-------------|------------------|----------|--------|
| `beamng` | 8 (single-row LiDAR) | 14 | LiDAR |
| `beamng_predicted` | 8 + 2 hints | 16 | LiDAR |
| `beamng_continuous` | 8 | 14 | LiDAR |
| `beamng_continuous_predicted` | 8 + 2 hints | 16 | LiDAR |
| `beamng_lidar` | 32 (4×8 LiDAR grid) | 38 | LiDAR (grid) |
| `beamng_camera` | 256 (16×16 dashcam) | 262 | Camera |
| `beamng_camera_predicted` | 256 + 2 hints | 264 | Camera |

Algorithm → action space and reward:

| Algorithm | Action space | Reward |
|-----------|--------------|--------|
| `dqn` | discrete (7-action table) | `default` |
| `ddpg` | continuous (`[-1,1]`) | `ddpg` on LiDAR perception, else `default` |
| `td3` | continuous (`[-1,1]`) | `ddpg` on LiDAR perception, else `default` |

The CLI only offers environments compatible with the chosen algorithm
(`registry.compatible_environments`), so e.g. DQN is restricted to the discrete
LiDAR envs. `q_learning` is tabular (taxi-only) and is **excluded** entirely.

`build_multi_session()` builds each agent sized to **its own** env's
observation length (`slot_n_states(env)`) with the algorithm's action
dimensionality. In the scene, `observe(slot)` branches on the slot's perception
(LiDAR single-row / LiDAR grid via `process_lidar`, or camera via
`process_camera_frame`), and the DDPG reward's obstacle-proximity term only
applies to LiDAR perceptions (a camera has no LiDAR bins). The kinematic block
and per-slot reward/path-error logic mirror the single-agent envs exactly
(reading/writing `slot.*`), so each car behaves like a single-agent run of its
own algorithm+env — all stepped together.

---

## Configuration knobs

| Where | Setting | Default | Effect |
|-------|---------|---------|--------|
| `BeamNGMultiEnv.GRID_LANE_OFFSET` | lane spacing (m) | `4.0` | Distance between adjacent cars on the start line. Lower it (e.g. `3.0`) if outer cars start off a narrow road. |
| `_MARKER_RGBA` / `_color_rgba()` | colour → sphere RGBA | — | Maps a vehicle's colour name to its marker colour (unknown names fall back to green). |
| `_ENV_PROFILES` / `_PERCEPTION_FEATURES` / `_LIDAR_PERCEPTION` | env → perception/hints/feature-count/LiDAR params | — | How each registered env name maps to a perception type, observation size, and LiDAR config. |
| `BeamNGMultiEnv.CAM_*` | dashcam resolution / FOV / mount | 84×84 → 16×16 | Camera config for camera-perception cars (mirrors `BeamNGCameraEnv`). |
| `_MULTI_ALGOS` (cli) | selectable algos | `["dqn","ddpg","td3"]` | Algorithms offered in the menu. |
| `_MULTI_OUTPUT_DIR` (cli) | output folder | `outputs/multi-agents` | Where checkpoints + plots default to. |

`trajectory_hints` is no longer a session-wide prompt — it comes from each
car's chosen env (the `_predicted` variants add one waypoint hint).

The starting grid fans vehicles along the spawn heading's **right** axis,
centred on the spawn point (e.g. 4 cars at −6 / −2 / +2 / +6 m), all facing
forward, so none starts behind another.

---

## Collisions

Collisions are a property of your BeamNG build / scenario setup — **not**
controlled by this code. With collisions enabled, the side-by-side grid keeps
cars from overlapping at the start, and each car's LiDAR legitimately sees its
neighbours as real obstacles. (If you disable collisions, cars can pass through
each other and the grid is purely cosmetic.)

---

## Outputs

- **Checkpoints:** one per agent at its `save_path` (default
  `outputs/multi-agents/<algo>_<env>_<index>.pth`), saved every `save_every`
  episodes and at session end.
- **Plots:** one reward/steps plot per agent written **beside its checkpoint**
  as `<slot-name>_training.png` (so it follows the chosen save directory),
  produced at session end (reuses `PipelineRunner._save_plot`).

`outputs/` (including `outputs/multi-agents/`) is git-ignored.

---

## Testing

Unit tests run without a live BeamNG (mocked connection/sensors):

```bash
python -m pytest tests/test_beamng_geometry.py tests/test_beamng_camera_util.py \
                 tests/test_beamng.py tests/test_beamng_multi.py \
                 tests/test_multi_runner.py tests/test_cli_multi.py -v
```

Coverage: pure LiDAR geometry, pure camera-frame processing, the single-env
delegation, slot state, env profiles + per-env `n_states`, `build_slots`
(including reward gating + camera), action mapping, reward branches,
observation, lifecycle (teleport/step/close), the starting grid, the coloured
markers, and the runner loop's stop conditions.

The live end-to-end check (spawning real cars and watching them train) requires
BeamNG.drive and is documented as the final step of the implementation plan.

---

## Limitations / notes

- Performance scales with vehicle count and perception: N sensors (LiDARs
  and/or cameras) + N agent updates per tick. Cameras are the heaviest — a few
  camera cars cost meaningfully more GPU/CPU than LiDAR cars. Practical for a
  handful of vehicles, not dozens.
- A resumed checkpoint must match its car's env: the agent is sized to that
  env's observation length, so loading a checkpoint trained on a different env
  (different `n_states`) will fail. Pick the same env you trained it on.
- The debug-only LiDAR Lua logs "all points filtered" / "outside FOV" from the
  single-agent env are not reproduced after the geometry refactor (only "no
  points" and the full-bins log remain, behind `LOG_LIDAR`).
