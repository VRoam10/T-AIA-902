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
    from beamngpy.sensors import Camera, Damage, Electrics, Lidar, RoadsSensor
except ImportError:
    BeamNGpy = Scenario = Vehicle = Camera = Damage = Electrics = Lidar = RoadsSensor = None

from config import (  # noqa: F401  (LOG_LIDAR reserved for future logging parity)
    LIDAR_VISUALISE,
    LOG_LIDAR,
)
from core.trajectory import MapTrajectories, load_or_generate
from environments.beamng import BeamNGDrivingEnv
from environments.beamng_camera_util import process_camera_frame
from environments.beamng_geometry import (
    LidarConfig,
    body_orientation_features,
    ego_local_extents_from_bbox,
    process_lidar,
    wheel_terrain_features,
)


@dataclass
class VehicleSlot:
    """All per-vehicle state: identity, sensors, episode state, training stats.

    Nothing here is shared between vehicles — the multi env reads and writes
    these fields per slot so two algorithms never alias each other's state.
    """

    # Identity / config
    name: str
    color: str
    vehicle_id: str
    agent: Any
    reward_mode: str  # "default" (DQN) or "ddpg" (DDPG/TD3)
    action_space: str  # "discrete" or "continuous"
    save_path: str

    # Environment profile (assigned by build_slots from the chosen env name)
    env_name: str = "beamng"
    perception: str = "lidar"  # "lidar" | "lidar_grid" | "camera"
    trajectory_hints: int = 0
    body_orientation: bool = False
    wheel_terrain: bool = False
    n_states: int = 14  # observation length for this vehicle's env

    # Sensors (assigned during scenario load)
    vehicle: Any = None
    electrics: Any = None
    damage_sensor: Any = None
    lidar: Any = None
    camera: Any = None
    roads_sensor: Any = None

    # Per-vehicle starting-grid pose (assigned during scenario load)
    spawn_pos: tuple = (0.0, 0.0, 0.0)
    spawn_rot: tuple = (0.0, 0.0, 0.0, 1.0)
    path_idx: int = 0
    waypoints: list = field(default_factory=list)

    # Episode state
    waypoint_idx: int = 0
    last_damage: float = 0.0
    last_dist: float = 0.0
    current_dist: float = 0.0
    current_pos: tuple = (0.0, 0.0, 0.0)
    checkpoint_dist: float = 0.0
    checkpoint_hit: bool = False
    steps: int = 0
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
        self.checkpoint_dist = 0.0
        self.checkpoint_hit = False
        self.steps = 0
        self.ep_reward = 0.0
        self.ep_losses = []
        self.ep_speeds = []
        self.done = False


# Algorithms whose action space is continuous (actor outputs in [-1, 1]).
_CONTINUOUS_ALGOS = {"ddpg", "td3"}

# Registered BeamNG env name -> perception type.
# trajectory_hints is now a per-session prompt, not encoded in the env name.
_ENV_PROFILES = {
    "beamng": "lidar",
    "beamng_continuous": "lidar",
    "beamng_lidar": "lidar_grid",
    "beamng_camera": "camera",
}

# Perception type -> number of perception features in the observation vector.
_PERCEPTION_FEATURES = {"lidar": 8, "lidar_grid": 32, "camera": 256}

# Perception type -> LiDAR sensor params (vertical bins/resolution/FOV). Camera
# perception has no entry (it uses a Camera sensor instead).
_LIDAR_PERCEPTION = {
    "lidar": {"v_bins": 1, "vert_res": 32, "vert_angle": 26.9},
    "lidar_grid": {"v_bins": 4, "vert_res": 16, "vert_angle": 20.0},
}

_KINEMATIC_FEATURES = 6  # speed, steering, heading_err, lateral_err, damage, dist


def env_profile(env_name: str) -> str:
    """Return the perception type for a registered BeamNG env name."""
    return _ENV_PROFILES.get(env_name, "lidar")


def slot_n_states(
    env_name: str,
    trajectory_hints: int = 0,
    body_orientation: bool = False,
    wheel_terrain: bool = False,
) -> int:
    """Observation length for a vehicle running the given env with the given options."""
    perception = env_profile(env_name)
    return (
        _KINEMATIC_FEATURES
        + _PERCEPTION_FEATURES[perception]
        + 2 * trajectory_hints
        + (2 if body_orientation else 0)
        + (2 if wheel_terrain else 0)
    )


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

    Each spec dict: {"algo", "env", "agent", "vehicle_id", "color", "save_path",
    "trajectory_hints", "body_orientation", "wheel_terrain"}.
    trajectory_hints defaults to 0 when absent; body_orientation and
    wheel_terrain both default to False when absent.
    reward_mode uses the LiDAR-aware DDPG reward only for continuous algos on a
    LiDAR perception (camera and discrete DQN fall back to the default reward).
    """
    slots = []
    for i, spec in enumerate(specs):
        algo = spec["algo"]
        env_name = spec.get("env", "beamng")
        trajectory_hints = spec.get("trajectory_hints", 0)
        body_orientation = spec.get("body_orientation", False)
        wheel_terrain = spec.get("wheel_terrain", False)
        perception = env_profile(env_name)
        continuous = algo in _CONTINUOUS_ALGOS
        ddpg_reward = continuous and perception in ("lidar", "lidar_grid")
        slots.append(
            VehicleSlot(
                name=f"ego_{i}",
                color=spec["color"],
                vehicle_id=spec["vehicle_id"],
                agent=spec["agent"],
                reward_mode="ddpg" if ddpg_reward else "default",
                action_space="continuous" if continuous else "discrete",
                save_path=spec["save_path"],
                env_name=env_name,
                perception=perception,
                trajectory_hints=trajectory_hints,
                body_orientation=body_orientation,
                wheel_terrain=wheel_terrain,
                n_states=slot_n_states(env_name, trajectory_hints, body_orientation, wheel_terrain),
            )
        )
    return slots


class BeamNGMultiEnv:
    """Owns one BeamNG scenario shared by N vehicles, each with its own slot.

    Reuses the single-vehicle env's constants (ACTIONS table, LiDAR config,
    waypoint/reward thresholds) via BeamNGDrivingEnv class attributes, but keeps
    every mutable bit of episode state in per-vehicle VehicleSlots.
    """

    # Reuse the discrete action table and tunables from the single-vehicle env.
    ACTIONS = BeamNGDrivingEnv.ACTIONS
    N_ACTIONS_DISCRETE = len(BeamNGDrivingEnv.ACTIONS)  # 7
    WAYPOINT_RADIUS = BeamNGDrivingEnv.WAYPOINT_RADIUS
    MAX_STEPS = BeamNGDrivingEnv.MAX_STEPS
    MAX_DAMAGE = BeamNGDrivingEnv.MAX_DAMAGE
    CHECKPOINT_WARN_DIST = BeamNGDrivingEnv.CHECKPOINT_WARN_DIST
    CHECKPOINT_RESET_DIST = BeamNGDrivingEnv.CHECKPOINT_RESET_DIST

    # LiDAR geometry constants (single forward row, same as base env).
    LIDAR_RAYS = BeamNGDrivingEnv.LIDAR_RAYS
    LIDAR_V_BINS = BeamNGDrivingEnv.LIDAR_V_BINS
    LIDAR_CHANNELS_PER_RAY = BeamNGDrivingEnv.LIDAR_CHANNELS_PER_RAY
    LIDAR_FOV_DEG = BeamNGDrivingEnv.LIDAR_FOV_DEG
    LIDAR_VERT_ANGLE = BeamNGDrivingEnv.LIDAR_VERT_ANGLE
    LIDAR_MAX_DIST = BeamNGDrivingEnv.LIDAR_MAX_DIST
    LIDAR_GROUND_CLEARANCE = BeamNGDrivingEnv.LIDAR_GROUND_CLEARANCE
    LIDAR_SELF_MARGIN = BeamNGDrivingEnv.LIDAR_SELF_MARGIN
    LIDAR_MOUNT_POS = BeamNGDrivingEnv.LIDAR_MOUNT_POS
    LIDAR_MOUNT_DIR = BeamNGDrivingEnv.LIDAR_MOUNT_DIR
    LIDAR_MOUNT_UP = BeamNGDrivingEnv.LIDAR_MOUNT_UP
    LIDAR_VERT_RES = BeamNGDrivingEnv.LIDAR_VERT_RES
    LIDAR_ROOF_CLEARANCE = BeamNGDrivingEnv.LIDAR_ROOF_CLEARANCE
    BBOX_MAX_HALF_EXTENT = BeamNGDrivingEnv.BBOX_MAX_HALF_EXTENT

    VEHICLES = BeamNGDrivingEnv.VEHICLES

    HALF_TRACK_WIDTH = 0.7  # metres — half vehicle track, for per-wheel road-edge projection

    # Dashcam config for camera-perception vehicles (mirrors BeamNGCameraEnv).
    CAM_RESOLUTION = (84, 84)
    CAM_OUT_SIZE = (16, 16)
    CAM_FOV_Y = 70
    CAM_POS = (0, -0.5, 1.5)
    CAM_DIR = (0, -1, 0)

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
    ):
        self.slots = slots
        self.beamng_home = beamng_home
        self.beamng_user = beamng_user
        self.host = host
        self.port = port
        self.headless = headless
        self.map_name = map_name
        self.random_path = random_path

        self.bng: BeamNGpy = None
        self.scenario: Scenario = None
        self.trajectories: MapTrajectories | None = None

    def _lidar_config_for(self, slot: VehicleSlot) -> LidarConfig:
        """LiDAR binning config for a slot's perception (single-row or 2D grid)."""
        p = _LIDAR_PERCEPTION[slot.perception]
        return LidarConfig(
            rays=self.LIDAR_RAYS,
            v_bins=p["v_bins"],
            channels=self.LIDAR_CHANNELS_PER_RAY,
            fov_deg=self.LIDAR_FOV_DEG,
            vert_angle=p["vert_angle"],
            max_dist=self.LIDAR_MAX_DIST,
            self_margin=self.LIDAR_SELF_MARGIN,
            ground_clearance=self.LIDAR_GROUND_CLEARANCE,
        )

    def apply_action(self, slot: VehicleSlot, action) -> None:
        """Map an agent action to vehicle controls. Does not step physics."""
        if slot.action_space == "discrete" or isinstance(action, (int, np.integer)):
            ctrl = self.ACTIONS[int(action)]
            throttle, steering, brake = ctrl["throttle"], ctrl["steering"], ctrl["brake"]
        else:
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            if action.shape[0] == 2:
                accel = float(action[0])
                steering = float(action[1])
                throttle = max(0.0, accel)
                brake = max(0.0, -accel)
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

    def compute_reward(self, slot, obs):
        if slot.reward_mode == "ddpg":
            return self._reward_ddpg(slot, obs)
        return self._reward_default(slot, obs)

    def _reward_default(self, slot, obs):
        speed, steering, _heading_err, _lateral_err, damage_norm = obs[:5]
        damage = damage_norm * 1000.0
        done = False
        reward = 0.0

        if speed < 0.05:
            reward -= 2.0
        reward -= abs(steering) * 0.2

        if damage > slot.last_damage + 50:
            reward -= 50.0
        if damage >= self.MAX_DAMAGE:
            done = True
        slot.last_damage = damage

        if slot.steps >= self.MAX_STEPS:
            done = True

        if slot.checkpoint_hit:
            reward += 100.0 * slot.waypoint_idx
            slot.checkpoint_hit = False

        if slot.waypoint_idx >= len(slot.waypoints):
            reward += 200.0
            done = True

        dist = slot.checkpoint_dist
        if dist >= self.CHECKPOINT_RESET_DIST:
            reward -= 100.0
            done = True
        elif dist >= self.CHECKPOINT_WARN_DIST:
            reward -= (
                (dist - self.CHECKPOINT_WARN_DIST)
                / (self.CHECKPOINT_RESET_DIST - self.CHECKPOINT_WARN_DIST)
                * 10.0
            )

        return float(reward), done

    def _reward_ddpg(self, slot, obs):
        speed, _steering, heading_err, _lateral_err, damage_norm = obs[:5]
        lidar_bins = obs[5:]
        damage = damage_norm * 1000.0
        alignment = np.cos(heading_err * np.pi)
        done = False
        reward = 0.0

        dist_delta = slot.last_dist - slot.current_dist
        reward += dist_delta * 3.0
        slot.last_dist = slot.current_dist

        reward += speed * alignment * 3.0
        reward += alignment * 0.5

        if speed < 0.05:
            reward -= 1.0

        min_lidar = float(np.min(lidar_bins)) if lidar_bins.size else 1.0
        if min_lidar < 0.2:
            reward -= (1.0 - min_lidar) * 5.0
        elif min_lidar < 0.4:
            reward -= (1.0 - min_lidar) * 2.0

        damage_delta = damage - slot.last_damage
        if damage_delta > 0:
            reward -= damage_delta * 0.3
        if damage_delta > 150:
            reward -= 30.0
            done = True
        if damage >= self.MAX_DAMAGE:
            done = True
        slot.last_damage = damage

        if slot.steps >= self.MAX_STEPS:
            done = True

        if slot.checkpoint_hit:
            reward += 50.0
            slot.checkpoint_hit = False

        if slot.waypoint_idx >= len(slot.waypoints):
            reward += 200.0
            slot.waypoint_idx = 0
            done = True

        return float(reward), done

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
        if slot.waypoints:
            target = slot.waypoints[slot.waypoint_idx % len(slot.waypoints)]
            slot.checkpoint_dist = float(np.hypot(pos[0] - target[0], pos[1] - target[1]))

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
                        np.clip(dist / self.CHECKPOINT_WARN_DIST, 0.0, 2.0),
                    ],
                    dtype=np.float32,
                ),
                perception,
                waypoint_hints,
                self._slot_extra_features(slot, state),
            ]
        )

    def _perceive(self, slot: VehicleSlot, pos, vehicle_heading) -> np.ndarray:
        """Return the slot's perception feature block (LiDAR bins or camera pixels)."""
        if slot.perception == "camera":
            colour = slot.camera.poll().get("colour", None) if slot.camera is not None else None
            return process_camera_frame(colour, self.CAM_OUT_SIZE)
        point_cloud = slot.lidar.poll().get("pointCloud", None) if slot.lidar is not None else None
        bins, _debug = process_lidar(
            point_cloud, pos, vehicle_heading, slot.ego_local_extents, self._lidar_config_for(slot)
        )
        return bins

    def _slot_extra_features(self, slot, state) -> np.ndarray:
        """Optional observation tail for a slot (body orientation / wheel terrain).

        Calls the shared geometry helpers; empty when both flags are off.
        """
        state = state or {}
        blocks = []
        if slot.body_orientation:
            blocks.append(
                body_orientation_features(
                    state.get("dir", (0.0, 1.0, 0.0)), state.get("up", (0.0, 0.0, 1.0))
                )
            )
        if slot.wheel_terrain:
            payload = slot.roads_sensor.poll() if slot.roads_sensor is not None else None
            blocks.append(wheel_terrain_features(payload, self.HALF_TRACK_WIDTH))
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

    def _pick_distinct_path_idx(self, slot) -> int:
        """A random path index not currently held by any other slot."""
        taken = {s.path_idx for s in self.slots if s is not slot}
        free = [i for i in range(len(self.trajectories.paths)) if i not in taken]
        return random.choice(free)

    def _resolve_trajectory(self):
        import time

        from core.trajectory import CACHE_DIR

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
            vcfg = self.VEHICLES.get(slot.vehicle_id, self.VEHICLES["taxi"])
            vcfg = {**vcfg, "color": slot.color}
            slot.vehicle = Vehicle(slot.name, **vcfg)
            slot.electrics = Electrics()
            slot.damage_sensor = Damage()
            slot.vehicle.attach_sensor("electrics", slot.electrics)
            slot.vehicle.attach_sensor("damage", slot.damage_sensor)
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
        self.bng.set_deterministic(30)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(1.0)

        for slot in self.slots:
            self._create_slot_sensor(slot)
            self._update_slot_marker(slot)

    def _create_slot_sensor(self, slot: VehicleSlot):
        """Attach the perception sensor (LiDAR or camera) for one slot.

        Sensors must be created after the scenario starts. Camera slots get a
        dashcam; LiDAR/LiDAR-grid slots get a LiDAR sized for their perception
        plus a cached ego bbox for self-hit filtering.
        """
        if slot.wheel_terrain:
            slot.roads_sensor = RoadsSensor(f"roads_{slot.name}", self.bng, slot.vehicle)

        if slot.perception == "camera":
            slot.camera = Camera(
                f"cam_{slot.name}",
                self.bng,
                slot.vehicle,
                pos=self.CAM_POS,
                dir=self.CAM_DIR,
                field_of_view_y=self.CAM_FOV_Y,
                resolution=self.CAM_RESOLUTION,
                is_render_colours=True,
                is_render_depth=False,
                is_render_annotations=False,
                is_visualised=False,
                is_static=False,
            )
            return

        self._cache_ego_local_bbox(slot)
        p = _LIDAR_PERCEPTION[slot.perception]
        slot.lidar = Lidar(
            f"lidar_{slot.name}",
            self.bng,
            slot.vehicle,
            pos=self._resolve_slot_lidar_mount_pos(slot),
            dir=self.LIDAR_MOUNT_DIR,
            up=self.LIDAR_MOUNT_UP,
            requested_update_time=0.05,
            frequency=30,
            vertical_resolution=p["vert_res"],
            vertical_angle=p["vert_angle"],
            horizontal_angle=self.LIDAR_FOV_DEG,
            max_distance=self.LIDAR_MAX_DIST,
            is_360_mode=True,
            is_rotate_mode=False,
            is_using_shared_memory=False,
            is_visualised=LIDAR_VISUALISE,
            is_snapping_desired=False,
            is_force_inside_triangle=False,
        )

    def _resolve_slot_lidar_mount_pos(self, slot: VehicleSlot) -> tuple[float, float, float]:
        """Mirror BeamNGDrivingEnv._resolve_lidar_mount_pos using the slot's cached ego box."""
        if slot.ego_local_extents is None:
            return self.LIDAR_MOUNT_POS
        _, _, _, _, _, z_max = slot.ego_local_extents
        return (0.0, 0.0, float(z_max + self.LIDAR_ROOF_CLEARANCE))

    def _cache_ego_local_bbox(self, slot: VehicleSlot):
        try:
            slot.vehicle.poll_sensors()
            bbox = slot.vehicle.get_bbox()
        except Exception:
            slot.ego_local_extents = None
            return
        state = slot.vehicle.state or {}
        slot.ego_local_extents = ego_local_extents_from_bbox(
            bbox, state, self.LIDAR_SELF_MARGIN, self.BBOX_MAX_HALF_EXTENT
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
            slot.vehicle.teleport(slot.spawn_pos, rot_quat=slot.spawn_rot, reset=True)
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
        slot.vehicle.teleport(slot.spawn_pos, rot_quat=slot.spawn_rot, reset=True)
        slot.reset_episode()
        if slot.lidar is not None or slot.electrics is not None:
            slot.last_obs = self.observe(slot)
            slot.last_dist = slot.current_dist
        self._update_slot_marker(slot)

    def step_physics(self):
        """Advance every vehicle by one env step (10 physics ticks)."""
        self.bng.step(10)

    def close(self):
        if self.bng is None:
            return
        import threading

        for slot in self.slots:
            for sensor_attr in ("lidar", "camera", "roads_sensor"):
                sensor = getattr(slot, sensor_attr)
                if sensor is not None:
                    t = threading.Thread(target=sensor.remove, daemon=True)
                    t.start()
                    t.join(timeout=3.0)
                    setattr(slot, sensor_attr, None)
        t = threading.Thread(target=self.bng.close, daemon=True)
        t.start()
        t.join(timeout=5.0)
        self.bng = None
