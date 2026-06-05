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
2. **Trajectory hints per vehicle** — `0` (default) or `1`. Applies to *all*
   vehicles so their observation length matches.
3. **Add vehicles**, one at a time:
   - **Algorithm** — `dqn`, `ddpg`, or `td3`.
   - **Vehicle model** — Burnside (Taxi), Gavril T-Series, or Ibishu Pigeon.
   - **Model save path** — default `outputs/<algo>_multi_<index>.pth`. If a
     checkpoint already exists you can **C**ontinue (resume) or **R**eset it.
   - A colour is auto-assigned by index (Yellow, Red, Blue, Green, Orange,
     White, Black).
   - Repeat until you answer "no" to *Add another vehicle?*.
4. **Episodes per agent** and **Time limit (minutes)** — training stops when
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
| [environments/beamng.py](../../environments/beamng.py) | The single-vehicle env. Its LiDAR math now delegates to `beamng_geometry`. |
| [environments/beamng_multi.py](../../environments/beamng_multi.py) | `VehicleSlot` (per-vehicle state) + `BeamNGMultiEnv` (the shared scene). Also `build_slots()` and the colour helpers. |
| [core/multi_runner.py](../../core/multi_runner.py) | `MultiAgentRunner` — the parallel training loop. |
| [core/cli.py](../../core/cli.py) | Menu option 6 + the testable `build_multi_session()` builder. |

### `VehicleSlot`

A dataclass holding **everything** for one vehicle so two agents never share
mutable state:

- **Identity/config:** `name` (`ego_0`…), `color`, `vehicle_id`, `agent`,
  `reward_mode`, `action_space`, `save_path`.
- **Sensors:** its own `vehicle`, `electrics`, `damage_sensor`, `lidar`.
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

## How mixed algorithms share one scene

All vehicles share **one observation contract** (the LiDAR env's 14-float
vector, `6` kinematic + `8` LiDAR, plus `2 × trajectory_hints`). Algorithms
differ only in:

| Algorithm | Action space | `reward_mode` | Notes |
|-----------|--------------|---------------|-------|
| `dqn` | discrete (7-action table) | `default` | |
| `ddpg` | continuous (`[-1,1]`) | `ddpg` | |
| `td3` | continuous (`[-1,1]`) | `ddpg` | |

`q_learning` is tabular (taxi-only) and is **excluded** from BeamNG racing.

`build_multi_session()` constructs the env first to read its shared
`n_states`, then builds each agent against that size with its own action
dimensionality. The per-slot reward/observation/path-error logic mirrors the
single-agent env exactly (it reads/writes `slot.*` instead of `self.*`), so a
DQN vehicle behaves like a single-agent DQN run and a DDPG/TD3 vehicle gets the
DDPG-style reward — just all at once.

---

## Configuration knobs

| Where | Setting | Default | Effect |
|-------|---------|---------|--------|
| `BeamNGMultiEnv.GRID_LANE_OFFSET` | lane spacing (m) | `4.0` | Distance between adjacent cars on the start line. Lower it (e.g. `3.0`) if outer cars start off a narrow road. |
| `_MARKER_RGBA` / `_color_rgba()` | colour → sphere RGBA | — | Maps a vehicle's colour name to its marker colour (unknown names fall back to green). |
| CLI prompt | `trajectory_hints` | `0` | Adds the next waypoint's local coords to every vehicle's observation (must match the checkpoints' training). |
| `_MULTI_ALGOS` (cli) | selectable algos | `["dqn","ddpg","td3"]` | Algorithms offered in the menu. |

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
  `outputs/<algo>_multi_<index>.pth`), saved every `save_every` episodes and at
  session end.
- **Plots:** one reward/steps plot per agent at
  `outputs/<slot-name>_multi_training.png`, written at session end (reuses
  `PipelineRunner._save_plot`).

`outputs/` is git-ignored.

---

## Testing

Unit tests run without a live BeamNG (mocked connection/sensors):

```bash
python -m pytest tests/test_beamng_geometry.py tests/test_beamng.py \
                 tests/test_beamng_multi.py tests/test_multi_runner.py \
                 tests/test_cli_multi.py -v
```

Coverage: pure LiDAR geometry, the single-env delegation, slot state,
`build_slots`, action mapping, reward branches, observation, lifecycle
(teleport/step/close), the starting grid, the coloured markers, and the runner
loop's stop conditions.

The live end-to-end check (spawning real cars and watching them train) requires
BeamNG.drive and is documented as the final step of the implementation plan.

---

## Limitations / notes

- Performance scales with vehicle count: N LiDARs + N agent updates per tick.
  Practical for a handful of vehicles, not dozens.
- All vehicles must share the same `trajectory_hints` so their observation
  length matches every loaded checkpoint.
- The debug-only LiDAR Lua logs "all points filtered" / "outside FOV" from the
  single-agent env are not reproduced after the geometry refactor (only "no
  points" and the full-bins log remain, behind `LOG_LIDAR`).
