# Multi-Agent Simultaneous Training in BeamNG — Design

**Date:** 2026-06-05
**Status:** Approved (pending spec review)

## Summary

Replace the discarded "parked NPC vehicles" test with a real feature: spawn N
vehicles in a single BeamNG scenario (collisions are disabled) and train each
one with its own RL algorithm/checkpoint **at the same time**. A single physics
step advances every vehicle, but each vehicle runs its own independent episode
lifecycle (own observation, reward, episode counter, checkpoint), so several
algorithms learn in parallel on the same trajectory.

## Motivation

The current pipeline trains exactly one agent against one single-vehicle
environment (`BeamNGDrivingEnv` → `PipelineRunner`). Because BeamNG vehicles no
longer collide, multiple agents can share one scenario without interfering
physically. Training them together saves sim startup cost and lets us compare
learning behaviour side by side in one window.

## Key enabling facts

- `Vehicle.teleport(pos, rot_quat, reset=True)` exists in the installed
  `beamngpy` and resets velocity **and** damage — so one vehicle's episode can
  be reset without restarting the whole scenario (which would reset everyone).
- The `beamng` env (discrete, 7 actions) and `beamng_continuous` env
  (continuous, 3 actions) **share the same 14-float observation**. Therefore
  DQN, DDPG and TD3 can all consume one shared observation vector; they differ
  only in action space and reward mode.
- `q_learning` is tabular and registered for `taxi` only — it is **excluded**
  from BeamNG multi-agent training.

## Decisions (from brainstorming)

| Question | Decision |
| --- | --- |
| Purpose | Simultaneous training (not just an eval race) |
| Perception | One env type, mixed algos — every vehicle gets a LiDAR + kinematic observation (14 floats, + optional waypoint hints) |
| Spawn layout | All vehicles at the same spawn point (they overlap, then diverge) |
| Selection | Interactive CLI menu |
| Per-vehicle episode end | Teleport that vehicle to spawn and continue immediately; others keep driving |
| Session end | Whichever comes first: every agent reaches N episodes, or a wall-clock time limit |
| Structure | New env file + new runner file |
| Code reuse | **Approach A** — extract shared LiDAR geometry helpers used by both the single- and multi-vehicle envs |

## Architecture

### Files

1. **`environments/beamng_geometry.py`** *(new)* — pure, stateless functions
   extracted from `beamng.py`:
   - world → ego-local transform
   - `lidar_keep_mask(local_x, local_y, local_z, ego_extents, ...)` →
     `(keep_mask, debug_dict)`
   - `process_lidar(point_cloud, pos, heading, ego_extents, cfg)` → normalized
     bins (`float32[v*h*ch]`)
   - `ego_local_extents_from_bbox(bbox, state, margin)` → extents tuple
   These take all inputs explicitly (no `self`), so both envs share one
   implementation.

2. **`environments/beamng_multi.py`** *(new)* — `VehicleSlot` + `BeamNGMultiEnv`.

3. **`core/multi_runner.py`** *(new)* — `MultiAgentRunner` parallel training loop.

4. **`environments/beamng.py`** *(edit)* —
   - delete `_spawn_npc_vehicles`, the `NPC_*` class constants, and the call to
     `_spawn_npc_vehicles()` inside `_load_scenario`;
   - route its LiDAR math through `beamng_geometry` (so the helper has exactly
     one implementation). Single-agent behaviour must remain unchanged.

5. **`core/cli.py`** *(edit)* — add a menu entry "Multi-agent training (BeamNG)".

### `VehicleSlot` (one per vehicle)

Holds **all** per-vehicle state so nothing is shared by accident.

- **Identity / config:** `name` (`ego_0`, `ego_1`, …), `color`, `vehicle_id`,
  `agent`, `reward_mode` (`"default"` for DQN, `"ddpg"` for DDPG/TD3),
  `action_space` (`"discrete"` / `"continuous"`), `save_path`.
- **Sensors:** `vehicle`, `electrics`, `damage_sensor`, `lidar` (its own).
- **Episode state:** `waypoint_idx`, `last_damage`, `last_dist`, `current_dist`,
  `current_pos`, `checkpoint_dist`, `checkpoint_hit`, `steps`,
  `ego_local_extents`, `last_obs`, `done`.
- **Training stats:** `episode`, `ep_reward`, `ep_losses`, `reward_history`,
  `steps_history`.

### `BeamNGMultiEnv`

Owns the shared scene: `bng`, `scenario`, `trajectory`, `waypoints`,
`map_name`, `trajectory_hints` (session-level; identical for all vehicles so
the observation length matches every agent).

Methods:

- `launch()` / `_load_scenario()` — build the scenario, add **all** vehicles at
  the **same** spawn point, add the checkpoint rings once, `make` / `load` /
  `start`, then create one `Lidar` per slot (LiDAR must be created after the
  scenario starts). Cache each slot's ego-local bbox extents.
- `reset_all()` — teleport every vehicle to spawn, zero each slot's episode
  state, settle physics, fill each `slot.last_obs`.
- `observe(slot) -> np.ndarray` — poll that slot's sensors → 14-float (+ hints)
  observation, using the shared geometry helpers and that slot's
  `ego_local_extents`.
- `apply_action(slot, action)` — map discrete (`int` → `ACTIONS` table) or
  continuous (`ndarray` → throttle/steer/brake) to `vehicle.control()`. Does
  **not** step physics.
- `step_physics()` — a single `bng.step(10)` that advances every vehicle.
- `compute_reward(slot, obs) -> (reward, done)` — uses that slot's
  `reward_mode` and per-vehicle state (waypoint advance, damage delta, distance
  progress, etc.).
- `reset_vehicle(slot)` — `vehicle.teleport(spawn_pos, spawn_rot, reset=True)`
  and zero the slot's episode state.
- Optional per-vehicle colored target sphere via `bng.debug` (shows which
  waypoint each vehicle is chasing). Shared checkpoint rings remain.
- `close()` — remove every LiDAR, then close `bng`.

The per-vehicle observation and reward logic mirror `BeamNGDrivingEnv` but read
and write `slot.*` instead of `self.*`. The waypoint-advance and reward math are
short and are reimplemented as slot-aware methods (only the pure LiDAR geometry
is shared via `beamng_geometry`).

### `MultiAgentRunner.train(env, n_episodes, time_limit)`

Stop condition = **whichever comes first**: every slot has completed
`n_episodes`, or wall-clock ≥ `time_limit`.

Each tick:

1. For each active slot: `action = slot.agent.select_action(slot.last_obs)`;
   `env.apply_action(slot, action)`; stash `(state, action)`.
2. `env.step_physics()` — once for all vehicles.
3. For each active slot: `next_obs = env.observe(slot)`;
   `reward, done = env.compute_reward(slot, next_obs)`;
   `slot.agent.update(state, action, reward, next_obs, done)`; accumulate
   reward/loss; `slot.last_obs = next_obs`.
4. On a slot's `done`: append `ep_reward`/`steps` to history,
   `agent.decay_epsilon()`, `episode += 1`, save checkpoint every `save_every`,
   then `env.reset_vehicle(slot)` so its next episode starts immediately while
   the others keep going.

Output: a per-slot progress line (episode, reward, ε). At session end, save
every agent and write a per-agent training plot (reusing
`PipelineRunner._save_plot`).

### CLI flow

New main-menu entry → pick map → loop "add a vehicle":

- pick algorithm (restricted to BeamNG-compatible: `dqn`, `ddpg`, `td3`);
- pick vehicle model + color;
- model save path, with continue/reset handling like `_train_menu`;
- `reward_mode` and `action_space` derived from the algorithm.

Repeat until the user is done adding vehicles, then prompt for number of
episodes and a time-limit in minutes. Build the `BeamNGMultiEnv` + the agents
(each agent built with the shared observation's `n_states` and its own
`n_actions`) and run the `MultiAgentRunner`. Finally `env.close()`.

## Trade-offs & caveats

- **Performance:** N LiDAR sensors plus N agent updates per tick. Practical for
  a handful of vehicles, not dozens. Documented, not optimized.
- **Observation match:** all slots share `trajectory_hints`, so a checkpoint
  trained with a different hint setting will not fit — the CLI surfaces the
  expected `n_states`.
- **Markers:** the single global active-marker becomes per-vehicle (colored) or
  is dropped; the shared checkpoint rings are unchanged.
- **Same-spawn overlap:** vehicles overlap at spawn; harmless because collisions
  are disabled.

## Out of scope (YAGNI)

- Eval-only "race" mode (this design is training; an eval variant can reuse the
  same env later).
- Kinematic-only fast mode (sensors are always on per the chosen perception).
- Heterogeneous per-vehicle observation types (camera + lidar in one scene).
- Auto-loading every checkpoint in `outputs/`.
