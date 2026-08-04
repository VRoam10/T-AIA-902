"""Multi-vehicle BeamNG environment for simultaneous parallel training.

One scenario holds N vehicles (collisions disabled). A single physics step
advances every vehicle; each vehicle keeps its own episode state in a
VehicleSlot so several algorithms train in parallel on one trajectory.
"""

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors import Damage, Electrics, GForces
except ImportError:
    BeamNGpy = Scenario = Vehicle = Damage = Electrics = GForces = None

from core.trajectory import MapTrajectories, load_or_generate
from environments import beamng_sensors, beamng_spec
from environments.beamng import BeamNGDrivingEnv
from environments.beamng_features import road_info_features, wheel_info_features
from environments.beamng_geometry import (
    body_orientation_features,
    starting_grid,
)
from environments.beamng_reward import compute_race_reward
from environments.beamng_spawn import corrected_spawn, measure_spawn_z_correction


@dataclass
class VehicleSlot:
    """All per-vehicle state: identity, sensors, episode state, training stats.

    Nothing here is shared between vehicles — the multi env reads and writes
    these fields per slot so two algorithms never alias each other's state.
    """

    # Identity / config
    name: str
    color: str
    agent: Any
    save_path: str

    # Configuration axes (see environments.beamng_spec)
    sensor: str = beamng_spec.DEFAULT_SENSOR  # "lidar" | "adv_lidar" | "camera"
    output: str = "fixed"  # "fixed" | "continuous"
    # A human entrant (race mode only): the player drives it, so we apply no
    # controls and it has no agent, observation or action head.
    human: bool = False
    trajectory_hints: int = 0
    body_orientation: bool = False
    road_info: bool = False
    wheel_info: bool = False
    n_states: int = 14  # observation length for this vehicle's config

    # Sensors (assigned during scenario load)
    vehicle: Any = None
    electrics: Any = None
    damage_sensor: Any = None
    lidar: Any = None
    camera: Any = None
    roads_sensor: Any = None
    gforces: Any = None

    # Per-vehicle starting-grid pose (assigned during scenario load)
    spawn_pos: tuple = (0.0, 0.0, 0.0)
    spawn_rot: tuple = (0.0, 0.0, 0.0, 1.0)
    # How far this slot's cached spawn height sits above where its car rests,
    # measured from the clung scenario spawn (see environments.beamng_spawn).
    # Teleports add it so a reset places the car instead of dropping it.
    spawn_z_correction: float = 0.0
    path_idx: int = 0
    waypoints: list = field(default_factory=list)

    # Episode state
    waypoint_idx: int = 0
    last_damage: float = 0.0
    last_dist: float = 0.0
    current_dist: float = 0.0
    current_pos: tuple = (0.0, 0.0, 0.0)
    checkpoint_hit: bool = False
    invuln_steps: int = 0  # damage-immune steps remaining (granted on checkpoint hit)
    steps_since_checkpoint: int = 0  # drives the segment-time bonus
    steps: int = 0
    finished: bool = False  # cleared the last checkpoint (the race winner, if first)
    # Race-mode gap bookkeeping: own and best-rival progress at the previous tick,
    # so the reward's gap term can telescope. Unused when running solo.
    last_progress_m: float = 0.0
    last_rival_progress_m: float = 0.0
    ego_local_extents: tuple | None = None
    last_obs: np.ndarray | None = None
    done: bool = False
    active_marker_id: str | None = None  # in-game target sphere handle

    # Per-episode running accumulators
    ep_reward: float = 0.0
    ep_losses: list = field(default_factory=list)
    ep_speeds: list = field(default_factory=list)

    # Cross-episode training stats
    episode: int = 0
    reward_history: list = field(default_factory=list)
    steps_history: list = field(default_factory=list)
    speed_history: list = field(default_factory=list)
    distance_history: list = field(default_factory=list)

    def reset_episode(self) -> None:
        """Zero running episode state. Keeps episode counter + histories."""
        self.waypoint_idx = 0
        self.last_damage = 0.0
        self.last_dist = 0.0
        self.current_dist = 0.0
        self.checkpoint_hit = False
        self.invuln_steps = 0
        self.steps_since_checkpoint = 0
        self.steps = 0
        self.finished = False
        self.last_progress_m = 0.0
        self.last_rival_progress_m = 0.0
        self.ep_reward = 0.0
        self.ep_losses = []
        self.ep_speeds = []
        self.done = False


def slot_n_states(
    sensor: str,
    trajectory_hints: int = 0,
    body_orientation: bool = False,
    road_info: bool = False,
    wheel_info: bool = False,
) -> int:
    """Observation length for a vehicle running the given sensor and options.

    Kept as a named entry point for the slot-building code; the arithmetic itself
    lives in :func:`environments.beamng_spec.obs_size`, which the single-vehicle
    env and the agent-sizing code also use.
    """
    return beamng_spec.obs_size(sensor, trajectory_hints, body_orientation, road_info, wheel_info)


# Vehicle-colour names -> target-marker RGBA, so each vehicle's waypoint sphere
# matches its car. Falls back to the single-agent env's green for unknown names.
_MARKER_RGBA = {
    "yellow": (1.0, 1.0, 0.0, 0.8),
    "red": (1.0, 0.0, 0.0, 0.8),
    "blue": (0.0, 0.4, 1.0, 0.8),
    "green": (0.0, 1.0, 0.2, 0.8),
    "orange": (1.0, 0.5, 0.0, 0.8),
    "white": (1.0, 1.0, 1.0, 0.8),
    "black": (0.1, 0.1, 0.1, 0.8),
}
_DEFAULT_MARKER_RGBA = (0.0, 1.0, 0.2, 0.8)


def _color_rgba(name: str) -> tuple[float, float, float, float]:
    """Map a BeamNG colour name to a marker RGBA tuple (case-insensitive)."""
    return _MARKER_RGBA.get((name or "").strip().lower(), _DEFAULT_MARKER_RGBA)


def build_slots(specs: list[dict]) -> list[VehicleSlot]:
    """Turn a list of vehicle specs into VehicleSlots.

    Each spec dict: ``{"algo", "agent", "color", "save_path", "sensor",
    "trajectory_hints", "body_orientation", "road_info", "wheel_info"}``.
    ``sensor`` defaults to the spec module's default; ``trajectory_hints`` to 0;
    the observation flags to False.

    The output axis is derived from the algorithm rather than carried in the spec —
    a DQN head cannot emit continuous controls and DDPG/TD3 emit nothing else, so
    letting a spec disagree with its algorithm would only create a way to be wrong.
    """
    slots = []
    for i, spec in enumerate(specs):
        sensor = spec.get("sensor", beamng_spec.DEFAULT_SENSOR)
        trajectory_hints = spec.get("trajectory_hints", 0)
        body_orientation = spec.get("body_orientation", False)
        road_info = spec.get("road_info", False)
        wheel_info = spec.get("wheel_info", False)
        slots.append(
            VehicleSlot(
                name=f"ego_{i}",
                color=spec["color"],
                agent=spec["agent"],
                save_path=spec["save_path"],
                sensor=sensor,
                output=beamng_spec.output_for_algo(spec["algo"]),
                trajectory_hints=trajectory_hints,
                body_orientation=body_orientation,
                road_info=road_info,
                wheel_info=wheel_info,
                n_states=slot_n_states(
                    sensor, trajectory_hints, body_orientation, road_info, wheel_info
                ),
            )
        )
    return slots


class BeamNGMultiEnv:
    """Owns one BeamNG scenario shared by N vehicles, each with its own slot.

    Reuses the single-vehicle env's constants (ACTIONS table, LiDAR config,
    waypoint/reward thresholds) via BeamNGDrivingEnv class attributes, but keeps
    every mutable bit of episode state in per-vehicle VehicleSlots.
    """

    # Reuse the discrete action table and tunables from the single-vehicle env, so
    # a vehicle behaves identically whether trained alone or alongside others.
    ACTIONS = BeamNGDrivingEnv.ACTIONS
    N_ACTIONS_DISCRETE = len(BeamNGDrivingEnv.ACTIONS)  # 7
    WAYPOINT_RADIUS = BeamNGDrivingEnv.WAYPOINT_RADIUS
    MAX_STEPS = BeamNGDrivingEnv.MAX_STEPS
    MAX_DAMAGE = BeamNGDrivingEnv.MAX_DAMAGE
    CHECKPOINT_DIST_NORM_M = BeamNGDrivingEnv.CHECKPOINT_DIST_NORM_M
    HALF_TRACK_WIDTH = BeamNGDrivingEnv.HALF_TRACK_WIDTH

    RACE_CAR = BeamNGDrivingEnv.RACE_CAR

    # Starting-grid geometry, used whenever vehicles share one path (race mode, or
    # training on a game track). Lateral must clear the car's width and stagger its
    # length; the race car is ~1.96 m wide and ~4.58 m long.
    GRID_LATERAL_M = 3.0
    GRID_STAGGER_M = 6.0

    def __init__(
        self,
        slots: list[VehicleSlot],
        beamng_home: str,
        beamng_user: str = None,
        host: str = "localhost",
        port: int = 25252,
        headless: bool = False,
        map_name: str = "gridmap_v2",
        random_path: bool = False,
        track: str | None = None,
    ):
        self.slots = slots
        self.beamng_home = beamng_home
        self.beamng_user = beamng_user
        self.host = host
        self.port = port
        self.headless = headless
        self.map_name = map_name
        self.random_path = random_path
        # One of the game's own race tracks (a quickrace key), or None for the
        # generated road-network paths. See core.quickrace.
        self.track = track

        self.bng: BeamNGpy = None
        self.scenario: Scenario = None
        self.trajectories: MapTrajectories | None = None

    def apply_action(self, slot: VehicleSlot, action) -> None:
        """Map an agent action to vehicle controls. Does not step physics.

        Delegates the mapping to the single-vehicle env so the two cannot drift:
        the same action must produce the same controls whether a policy is trained
        alone or in a shared scenario.
        """
        if slot.output == "fixed" or isinstance(action, (int, np.integer)):
            ctrl = self.ACTIONS[int(action)]
            throttle, steering, brake = ctrl["throttle"], ctrl["steering"], ctrl["brake"]
        else:
            action = np.clip(np.asarray(action, dtype=np.float32).ravel(), -1.0, 1.0)
            if action.shape[0] == 2:
                accel = float(action[0])
                throttle, steering, brake = max(0.0, accel), float(action[1]), max(0.0, -accel)
            else:
                throttle = float(max(0.0, action[0]))
                steering = float(action[1])
                brake = float(max(0.0, action[2]))
        slot.vehicle.control(throttle=throttle, steering=steering, brake=brake)

    def _path_errors(self, slot, pos, state):
        """Heading/lateral error to slot's current waypoint; advances on arrival.

        Sets slot.current_dist for the DDPG progress reward.
        """
        if not slot.waypoints or not state:
            slot.current_dist = 0.0
            return 0.0, 0.0, 0.0

        target = slot.waypoints[slot.waypoint_idx % len(slot.waypoints)]
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = float(np.hypot(dx, dy))
        slot.current_dist = dist

        if dist < self.WAYPOINT_RADIUS:
            slot.waypoint_idx += 1
            slot.checkpoint_hit = True
            self._update_slot_marker(slot)
            if slot.waypoint_idx < len(slot.waypoints):
                new_t = slot.waypoints[slot.waypoint_idx]
                slot.current_dist = float(np.hypot(new_t[0] - pos[0], new_t[1] - pos[1]))

        vel = state.get("vel", (1.0, 0.0, 0.0))
        vehicle_heading = np.arctan2(vel[1], vel[0])
        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
        lateral_err = dist * np.sin(heading_err)
        return float(heading_err), float(lateral_err), dist

    def compute_reward(self, slot, obs, **race_kwargs):
        """Racing reward, shared with the single-vehicle env via beamng_reward.

        Training vehicles run on separate paths and never meet, so no rival
        arguments are passed and the gap term contributes nothing. The race env
        forwards its gap/rival keywords through ``race_kwargs``.
        """
        outcome = compute_race_reward(
            obs,
            perception=slot.sensor,
            n_perception=beamng_spec.perception_features(slot.sensor),
            waypoints_len=len(slot.waypoints),
            waypoint_idx=slot.waypoint_idx,
            checkpoint_hit=slot.checkpoint_hit,
            last_dist=slot.last_dist,
            current_dist=slot.current_dist,
            last_damage=slot.last_damage,
            steps=slot.steps,
            invuln_steps=slot.invuln_steps,
            steps_since_checkpoint=slot.steps_since_checkpoint,
            max_steps=self.MAX_STEPS,
            max_damage=self.MAX_DAMAGE,
            **race_kwargs,
        )
        slot.last_dist = outcome.last_dist
        slot.last_damage = outcome.last_damage
        slot.invuln_steps = outcome.invuln_steps
        slot.checkpoint_hit = outcome.checkpoint_hit
        slot.waypoint_idx = outcome.waypoint_idx
        slot.steps_since_checkpoint = outcome.steps_since_checkpoint
        slot.finished = outcome.finished
        return outcome.reward, outcome.done

    def observe(self, slot: VehicleSlot) -> np.ndarray:
        """Poll a slot's sensors and return its normalized observation vector."""
        slot.vehicle.poll_sensors()

        elec = slot.electrics.data or {}
        dmg = slot.damage_sensor.data or {}
        speed = float(elec.get("wheelspeed", 0.0))
        steering = float(elec.get("steering", 0.0))
        damage = float(dmg.get("damage", 0.0))

        state = slot.vehicle.state or {}
        pos = state.get("pos", (0.0, 0.0, 0.0))
        vel = state.get("vel", (1.0, 0.0, 0.0))
        dir_vec = state.get("dir", vel)
        vehicle_heading = float(np.arctan2(dir_vec[1], dir_vec[0]))

        heading_err, lateral_err, dist = self._path_errors(slot, pos, state)

        perception = self._perceive(slot, pos, vehicle_heading)

        slot.current_pos = pos

        waypoint_hints = self._waypoint_hints(slot, pos, vehicle_heading)

        return np.concatenate(
            [
                np.array(
                    [
                        np.clip(speed / 50.0, -1.0, 1.0),
                        np.clip(steering, -1.0, 1.0),
                        np.clip(heading_err / np.pi, -1.0, 1.0),
                        np.clip(lateral_err / 5.0, -1.0, 1.0),
                        np.clip(damage / 1000.0, 0.0, 1.0),
                        np.clip(dist / self.CHECKPOINT_DIST_NORM_M, 0.0, 2.0),
                    ],
                    dtype=np.float32,
                ),
                perception,
                waypoint_hints,
                self._slot_extra_features(slot, state, pos, vehicle_heading, elec),
            ]
        )

    def _perceive(self, slot: VehicleSlot, pos, vehicle_heading) -> np.ndarray:
        """Return the slot's perception feature block (LiDAR bins or camera pixels)."""
        block, _debug, _frame = beamng_sensors.perception_block(
            sensor=slot.sensor,
            lidar=slot.lidar,
            camera=slot.camera,
            pos=pos,
            heading=vehicle_heading,
            ego_extents=slot.ego_local_extents,
        )
        return block

    def _slot_extra_features(self, slot, state, pos, heading, elec=None) -> np.ndarray:
        """Optional observation tail for a slot (body orientation / road / wheel).

        Calls the shared feature helpers; empty when all flags are off.
        """
        state = state or {}
        blocks = []
        if slot.body_orientation:
            blocks.append(
                body_orientation_features(
                    state.get("dir", (0.0, 1.0, 0.0)), state.get("up", (0.0, 0.0, 1.0))
                )
            )
        if slot.road_info:
            payload = slot.roads_sensor.poll() if slot.roads_sensor is not None else None
            blocks.append(road_info_features(payload, self.HALF_TRACK_WIDTH, pos, heading))
        if slot.wheel_info:
            forces = slot.gforces.data if slot.gforces is not None else None
            blocks.append(
                wheel_info_features(
                    elec or {},
                    forces,
                    state.get("vel", (0.0, 0.0, 0.0)),
                    state.get("dir", (1.0, 0.0, 0.0)),
                )
            )
        if not blocks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(blocks)

    def _waypoint_hints(self, slot, pos, vehicle_heading) -> np.ndarray:
        """Vehicle-local (forward, left) coords for the next trajectory_hints waypoints."""
        if not slot.trajectory_hints or not slot.waypoints:
            return np.empty(0, dtype=np.float32)
        NORM = 100.0
        cos_h = np.cos(-vehicle_heading)
        sin_h = np.sin(-vehicle_heading)
        hints: list[float] = []
        for i in range(slot.trajectory_hints):
            idx = (slot.waypoint_idx + i) % len(slot.waypoints)
            wp = slot.waypoints[idx]
            rel_x = wp[0] - pos[0]
            rel_y = wp[1] - pos[1]
            local_x = rel_x * cos_h - rel_y * sin_h
            local_y = rel_x * sin_h + rel_y * cos_h
            hints.append(float(np.clip(local_x / NORM, -1.0, 1.0)))
            hints.append(float(np.clip(local_y / NORM, -1.0, 1.0)))
        return np.array(hints, dtype=np.float32)

    def launch(self):
        """Start BeamNG, resolve all map paths, and load the multi-vehicle scenario."""
        self.bng = BeamNGpy(
            self.host,
            self.port,
            home=self.beamng_home,
            user=self.beamng_user,
            headless=self.headless,
        )
        self.bng.open(launch=True)
        self.trajectories = self._resolve_trajectory()
        self._assign_paths()
        self._load_scenario()

    def _assign_paths(self):
        """Give each vehicle its own path; error if vehicles outnumber paths."""
        paths = self.trajectories.paths
        if self.track:
            # A game track is one authored racing line, so "a path each" cannot
            # apply: everyone drives it, spread over a starting grid so they do
            # not spawn inside one another. Same arrangement the race mode uses.
            self._assign_shared_path(paths[0])
            return
        if len(self.slots) > len(paths):
            raise ValueError(
                f"{len(self.slots)} vehicles requested but map '{self.map_name}' has "
                f"only {len(paths)} distinct path(s). Reduce the vehicle count to "
                f"<= {len(paths)} or pick a map with more quick-travel points."
            )
        if self.random_path:
            order = list(range(len(paths)))
            random.shuffle(order)
            for slot, idx in zip(self.slots, order, strict=False):
                slot.path_idx = idx
                self._apply_path(slot, paths[idx])
        else:
            for i, slot in enumerate(self.slots):
                slot.path_idx = i
                self._apply_path(slot, paths[i])

    def _apply_path(self, slot, path):
        slot.waypoints = list(path.sparse_waypoints)
        slot.spawn_pos = path.spawn_pos
        slot.spawn_rot = path.spawn_rot

    def _assign_shared_path(self, path, path_idx: int = 0) -> None:
        """Put every slot on one path, spread across a starting grid.

        Shared by race mode (everyone on the same trace, by definition) and by
        training on a game track (one authored line, so there is nothing else to
        put them on).
        """
        grid = starting_grid(
            path.spawn_pos,
            path.spawn_rot,
            len(self.slots),
            lateral_m=self.GRID_LATERAL_M,
            stagger_m=self.GRID_STAGGER_M,
        )
        for slot, pos in zip(self.slots, grid, strict=True):
            slot.path_idx = path_idx
            self._apply_path(slot, path)
            slot.spawn_pos = pos  # the grid slot, not the shared centreline spawn

    def _pick_distinct_path_idx(self, slot) -> int:
        """A random path index not currently held by any other slot."""
        taken = {s.path_idx for s in self.slots if s is not slot}
        free = [i for i in range(len(self.trajectories.paths)) if i not in taken]
        return random.choice(free)

    def _load_track(self, key: str) -> MapTrajectories:
        """One of the game's race tracks, wrapped as this map's single path.

        Every entrant races the same line, which is what the shared-track race
        mode wants; training on a game track puts all vehicles on it too (there is
        only one path, so the per-vehicle distinct-path rule cannot apply).
        """
        from core import quickrace

        race = quickrace.load(self.map_name, key, self.beamng_home)
        traj = quickrace.to_trajectory(race)
        print(
            f"[track] {self.map_name}/{race.key}: {race.kind}, "
            f"{len(race.checkpoints)} checkpoints, {race.length_m():.0f} m"
        )
        return MapTrajectories(
            map_name=self.map_name,
            generated_at=traj.generated_at,
            paths=[traj],
        )

    def _resolve_trajectory(self):
        import time

        from core.trajectory import CACHE_DIR

        if self.track:
            # A game track is read from the level files, so it needs neither a
            # cache nor a probe scenario.
            return self._load_track(self.track)

        cache_path = CACHE_DIR / f"{self.map_name}.json"
        if cache_path.exists():
            return load_or_generate(self.map_name, bng=None)

        probe = Scenario(self.map_name, "trajectory_probe", description="Road probe")
        probe_vehicle = Vehicle("probe_vehicle", model="etk800")
        probe.add_vehicle(probe_vehicle, pos=(0.0, 0.0, 100.0), rot_quat=(0.0, 0.0, 0.0, 1.0))
        probe.make(self.bng)
        self.bng.load_scenario(probe)
        self.bng.start_scenario()
        time.sleep(0.5)
        return load_or_generate(self.map_name, self.bng)

    def _load_scenario(self):
        import time

        self.scenario = Scenario(self.map_name, "rl_multi_driving", description="RL Multi-Agent")

        for slot in self.slots:
            # One car model for everyone; only the paint differs, so a result
            # reflects the policies rather than the machinery.
            slot.vehicle = Vehicle(slot.name, **{**self.RACE_CAR, "color": slot.color})
            slot.electrics = Electrics()
            slot.damage_sensor = Damage()
            slot.vehicle.attach_sensor("electrics", slot.electrics)
            slot.vehicle.attach_sensor("damage", slot.damage_sensor)
            if slot.wheel_info:
                slot.gforces = GForces()
                slot.vehicle.attach_sensor("gforces", slot.gforces)
            self.scenario.add_vehicle(
                slot.vehicle,
                pos=slot.spawn_pos,
                rot_quat=slot.spawn_rot,
                cling=True,
            )

        if self.random_path:
            all_waypoints = [wp for p in self.trajectories.paths for wp in p.sparse_waypoints]
        else:
            all_waypoints = [wp for slot in self.slots for wp in slot.waypoints]
        scales = [(5.0, 5.0, 1.0)] * len(all_waypoints)
        self.scenario.add_checkpoints(all_waypoints, scales)

        self.scenario.make(self.bng)
        self.bng.set_deterministic(beamng_spec.PHYSICS_STEPS_PER_SECOND)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(1.0)

        # Every car above was spawned with cling=True and is now resting at its
        # own true surface height, so each slot measures its own correction —
        # grid slots sit metres apart laterally and a cambered road is not flat.
        for slot in self.slots:
            slot.spawn_z_correction = measure_spawn_z_correction(
                self.bng, slot.vehicle, slot.spawn_pos[2]
            )
            if slot.spawn_z_correction != 0.0:
                print(
                    f"[spawn] {slot.name}: teleport height corrected by "
                    f"{slot.spawn_z_correction:+.2f} m"
                )

        for slot in self.slots:
            self._create_slot_sensor(slot)
            self._update_slot_marker(slot)

    def _create_slot_sensor(self, slot: VehicleSlot):
        """Attach the perception sensor (LiDAR or camera) for one slot.

        Sensors must be created after the scenario starts. Camera slots get a
        dashcam; LiDAR slots get a LiDAR sized for their sensor plus a cached ego
        bbox, used both for self-hit filtering and to place the roof mount.
        """
        if slot.road_info:
            slot.roads_sensor = beamng_sensors.create_roads_sensor(
                f"roads_{slot.name}", self.bng, slot.vehicle
            )

        if slot.sensor == "camera":
            slot.camera = beamng_sensors.create_camera(
                f"cam_{slot.name}", self.bng, slot.vehicle, visualise=False
            )
            return

        slot.ego_local_extents = beamng_sensors.cache_ego_local_bbox(slot.vehicle)
        slot.lidar = beamng_sensors.create_lidar(
            f"lidar_{slot.name}", self.bng, slot.vehicle, slot.sensor, slot.ego_local_extents
        )

    def _update_slot_marker(self, slot: VehicleSlot):
        """Draw/refresh slot's target-waypoint sphere in its vehicle colour.

        Per-slot counterpart of BeamNGDrivingEnv._update_active_marker: each
        vehicle gets its own sphere coloured to match its car. Silently skipped
        without a live bng / on older beamngpy builds.
        """
        if self.bng is None or not slot.waypoints:
            return
        try:
            debug = self.bng.debug
            if slot.active_marker_id is not None:
                try:
                    debug.remove_spheres([slot.active_marker_id])
                except Exception:
                    pass
            target = slot.waypoints[slot.waypoint_idx % len(slot.waypoints)]
            pos = (target[0], target[1], target[2] + 2.0)
            ids = debug.add_spheres(
                coordinates=[pos],
                radii=[3.0],
                rgba_colors=[_color_rgba(slot.color)],
            )
            slot.active_marker_id = ids[0]
        except AttributeError:
            pass

    def reset_all(self):
        """Teleport every vehicle to spawn, zero episode state, prime last_obs."""
        if self.bng is None:
            self.launch()
        for slot in self.slots:
            slot.reset_episode()
            slot.vehicle.teleport(
                corrected_spawn(slot.spawn_pos, slot.spawn_z_correction),
                rot_quat=slot.spawn_rot,
                reset=True,
            )
            slot.vehicle.control(throttle=0.0, steering=0.0, brake=0.0)
        self.bng.step(5)
        for slot in self.slots:
            slot.last_obs = self.observe(slot)
            slot.last_dist = slot.current_dist
            self._update_slot_marker(slot)

    def reset_vehicle(self, slot: VehicleSlot):
        """Teleport one finished vehicle to its (possibly new) path for the next episode."""
        if self.random_path and self.trajectories is not None:
            slot.path_idx = self._pick_distinct_path_idx(slot)
            self._apply_path(slot, self.trajectories.paths[slot.path_idx])
        slot.vehicle.teleport(
            corrected_spawn(slot.spawn_pos, slot.spawn_z_correction),
            rot_quat=slot.spawn_rot,
            reset=True,
        )
        slot.reset_episode()
        # Let the teleport/repair land before the priming poll: sensors polled
        # with no intervening step still report the pre-teleport pose/damage,
        # which seeds last_dist with a huge fake progress delta and triggers an
        # instant fake crash on the first reward step of the new episode.
        if self.bng is not None:
            self.bng.step(5)
        if slot.lidar is not None or slot.electrics is not None:
            slot.last_obs = self.observe(slot)
            slot.last_dist = slot.current_dist
        self._update_slot_marker(slot)

    def step_physics(self):
        """Advance every vehicle by one env step.

        One shared advance for the whole field, so vehicles cannot desynchronise
        and contact stays symmetric.
        """
        self.bng.step(beamng_spec.PHYSICS_STEPS_PER_ENV_STEP)

    def close(self):
        if self.bng is None:
            return
        import threading

        for slot in self.slots:
            for sensor_attr in ("lidar", "camera", "roads_sensor"):
                beamng_sensors.remove_sensor(getattr(slot, sensor_attr))
                setattr(slot, sensor_attr, None)
        t = threading.Thread(target=self.bng.close, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.bng = None
