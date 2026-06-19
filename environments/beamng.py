import random
import sys
import threading
import time

import numpy as np

try:
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors import Camera, Damage, Electrics, Lidar, RoadsSensor
except ImportError:
    BeamNGpy = Scenario = Vehicle = Camera = Damage = Electrics = Lidar = RoadsSensor = None

from config import (
    LIDAR_VISUALISE,
    LOG_CAMERA,
    LOG_CHECKPOINT_HIT,
    LOG_CHECKPOINT_RESPAWN,
    LOG_CHECKPOINT_WARN,
    LOG_LIDAR,
)
from core.trajectory import TrajectoryData, load_or_generate
from environments.beamng_camera_util import process_camera_frame
from environments.beamng_geometry import (
    LidarConfig,
    body_orientation_features,
    wheel_terrain_features,
)


class BeamNGDrivingEnv:
    """
    Gymnasium-style environment wrapping BeamNG.drive via beamngpy.

    State  (5 + LIDAR_RAYS floats, all normalized to ~[-1, 1] or [0, 1]):
        speed          - wheel speed normalized to 50 m/s
        steering       - current steering angle (-1 to 1)
        heading_error  - angle between vehicle heading and next waypoint direction
        lateral_error  - perpendicular distance from path (normalized to 5 m)
        damage         - cumulative vehicle damage (normalized)
        lidar[0..N-1]  - nearest obstacle distance in each angular bin (0 = close, 1 = clear)

    Actions (discrete, 7):
        0 - idle / coast
        1 - full throttle straight
        2 - throttle + slight left
        3 - throttle + slight right
        4 - brake
        5 - throttle + sharp left
        6 - throttle + sharp right
    """

    ACTIONS = [
        {"throttle": 0.0, "steering": 0.0, "brake": 0.0},  # 0: idle
        {"throttle": 1.0, "steering": 0.0, "brake": 0.0},  # 1: straight
        {"throttle": 0.7, "steering": -0.3, "brake": 0.0},  # 2: slight left
        {"throttle": 0.7, "steering": 0.3, "brake": 0.0},  # 3: slight right
        {"throttle": 0.0, "steering": 0.0, "brake": 1.0},  # 4: brake
        {"throttle": 0.4, "steering": -0.6, "brake": 0.0},  # 5: sharp left
        {"throttle": 0.4, "steering": 0.6, "brake": 0.0},  # 6: sharp right
    ]

    N_ACTIONS = len(ACTIONS)

    # LiDAR configuration
    LIDAR_RAYS = 8  # number of horizontal angular bins (azimuth)
    LIDAR_V_BINS = 1  # number of vertical bins (elevation). 1 = single row (legacy).
    LIDAR_CHANNELS_PER_RAY = 1  # currently: (distance,). Future: (distance, v_rel, ttc, ...)
    LIDAR_FOV_DEG = 120.0  # total forward-facing field of view in degrees (azimuth)
    LIDAR_MAX_DIST = 50.0  # metres — normalization range
    LIDAR_GROUND_CLEARANCE = 0.30  # metres above ego bbox floor before a point counts as obstacle
    LIDAR_SELF_MARGIN = 0.30  # metres expansion of ego OBB when rejecting self-hits

    # Physical sensor mount/params — overridable per subclass.
    LIDAR_MOUNT_POS = (0, -2.2, 1.15)  # vehicle-local: forward of bumper, hood-line height
    LIDAR_MOUNT_DIR = (0, -1, 0)  # forward in vehicle space
    LIDAR_MOUNT_UP = (0, 0, 1)
    LIDAR_VERT_RES = 8  # vertical layers emitted by the sensor
    LIDAR_VERT_ANGLE = 6.0  # total vertical FOV in degrees; also the elevation-binning range

    # 5 kinematic + (vertical × horizontal × channels) lidar features.
    N_STATES = 6 + LIDAR_RAYS * LIDAR_V_BINS * LIDAR_CHANNELS_PER_RAY  # 14 by default

    CHECKPOINT_WARN_DIST = 200.0  # metres — start penalising when this far from checkpoint
    CHECKPOINT_RESET_DIST = 300.0  # metres — teleport back to spawn and big malus beyond this

    WAYPOINT_RADIUS = 8.0  # metres — how close before advancing to next waypoint
    MAX_STEPS = 500
    MAX_DAMAGE = 1000.0  # damage threshold that ends the episode
    HALF_TRACK_WIDTH = 0.7  # metres — half vehicle track, for per-wheel road-edge projection

    AVAILABLE_MAPS = ["gridmap_v2", "italy", "west_coast_usa", "smallgrid"]

    VEHICLES = {
        "taxi": {
            "model": "burnside",
            "licence": "Taxi",
            "color": "Yellow",
            "part_config": "vehicles/burnside/4door_early_v8_3M_taxi.pc",
        },
        "gavril_t_series": {
            "model": "us_semi",
            "licence": "T-Series",
            "color": "White",
            "part_config": "vehicles/us_semi/t83_sleeper.pc",
        },
        "ibishu_pigeon": {
            "model": "pigeon",
            "licence": "Pigeon",
            "color": "Red",
            "part_config": "vehicles/pigeon/base.pc",
        },
        "gavril_d_series": {
            "model": "pickup",
            "licence": "D-Series",
            "color": "green",
            "part_config": "vehicles/pickup/d25_longbed_4wd_lifted_A.pc",
        },
    }

    def __init__(
        self,
        beamng_home: str,
        beamng_user: str = None,
        host: str = "localhost",
        port: int = 25252,
        headless: bool = False,
        reward_mode: str = "default",
        vehicle_id: str = "taxi",
        map_name: str = "gridmap_v2",
        trajectory_hints: int = 0,
        body_orientation: bool = False,
        wheel_terrain: bool = False,
        random_path: bool = False,
    ):
        """
        Args:
            beamng_home: Path to BeamNG.drive installation directory.
                         e.g. r'C:\\Program Files (x86)\\Steam\\steamapps\\common\\BeamNG.drive'
            beamng_user: Optional path to BeamNG user folder (where mods/configs live).
            host: BeamNG server host (default localhost).
            port: BeamNG server port (default 25252).
        """
        self.beamng_home = beamng_home
        self.beamng_user = beamng_user
        self.host = host
        self.port = port

        self.bng: BeamNGpy = None
        self.vehicle: Vehicle = None
        self.scenario: Scenario = None
        self.electrics: Electrics = None
        self.damage_sensor: Damage = None
        self.lidar: Lidar = None
        self.roads_sensor: RoadsSensor = None

        self.reward_mode = reward_mode  # "default" or "ddpg"
        self.vehicle_id = vehicle_id
        self.map_name = map_name

        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._last_dist = 0.0
        self._steps = 0
        self._active_marker_id: str | None = None
        self._checkpoint_dist = 0.0
        self.headless = headless
        self.trajectory_hints = trajectory_hints
        self.body_orientation = body_orientation
        self.wheel_terrain = wheel_terrain
        self.random_path = random_path
        self.n_states = (
            self.N_STATES
            + trajectory_hints * 2
            + (2 if body_orientation else 0)
            + (2 if wheel_terrain else 0)
        )

        # Filled on first _launch() — either read from cache or generated then.
        self.trajectory: TrajectoryData | None = None
        self._paths: list[TrajectoryData] = []
        self.waypoints: list[tuple[float, float, float]] = []
        self._current_pos = (0.0, 0.0, 0.0)

        # Cached ego OBB extents in vehicle-local frame (x_min, x_max, y_min, y_max, z_min, z_max).
        # Populated once per scenario load; used by _process_lidar to reject self-hits.
        self._ego_local_extents: tuple[float, float, float, float, float, float] | None = None

        # Last-poll LiDAR filtering breakdown (counts + nearest kept point), for debug.
        self._lidar_debug: dict = {}

    def _select_waypoints(self) -> list[tuple[float, float, float]]:
        assert self.trajectory is not None
        return list(self.trajectory.sparse_waypoints)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the episode and return the initial observation."""
        if self.bng is None:
            self._launch()
        else:
            if self.random_path:
                # Teleport (reset=True) repositions AND resets the vehicle to the
                # newly chosen path's spawn. Do NOT call scenario.restart() here:
                # restart() snaps the car back to the baked path[0] spawn and
                # fights the teleport, so the path never actually changes. This
                # mirrors the multi-agent env's reset_vehicle (teleport, no restart).
                self._pick_episode_path()
                self.vehicle.teleport(
                    self.trajectory.spawn_pos,
                    rot_quat=self.trajectory.spawn_rot,
                    reset=True,
                )
            else:
                self.bng.scenario.restart()
            self._update_active_marker(0)
            # Test LiDAR après restart
            try:
                data = self.lidar.poll()
                pc = data.get("pointCloud", None)

                if pc is None:
                    print("[TEST LIDAR] pointCloud = None après restart")
                else:
                    print("[TEST LIDAR] points après restart :", len(pc))

            except Exception as e:
                print("[TEST LIDAR] ERREUR après restart :", repr(e))

        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._last_dist = 0.0
        self._steps = 0
        self._checkpoint_hit = False

        # Hold still for a moment so physics settle
        self.vehicle.control(throttle=0.0, steering=0.0, brake=0.0)
        self.bng.step(5)

        obs = self._observe()
        # Initialize last_dist after first observe so progress reward starts at 0
        self._last_dist = self._current_dist
        return obs

    def step(self, action):
        """
        Apply an action and advance the simulation.

        Accepts either:
          - int: discrete action index (from ACTIONS table)
          - np.ndarray of shape (2,): continuous [accel, steering] in [-1, 1]
            where accel >= 0 -> throttle, accel < 0 -> brake

        Returns:
            obs (np.ndarray), reward (float), done (bool), info (dict)
        """
        if isinstance(action, (int, np.integer)):
            ctrl = self.ACTIONS[action]
            throttle = ctrl["throttle"]
            steering = ctrl["steering"]
            brake = ctrl["brake"]
        else:
            action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            accel = float(action[0])
            steering = float(action[1])
            if accel >= 0:
                throttle = accel
                brake = 0.0
            else:
                throttle = 0.0
                brake = -accel

        self.vehicle.control(
            throttle=throttle,
            steering=steering,
            brake=brake,
        )

        # 10 physics steps ≈ 100 ms of simulation time
        self.bng.step(10)
        self._steps += 1

        obs = self._observe()
        reward, done = self._compute_reward(obs)
        info = {"steps": self._steps, "waypoint_idx": self._waypoint_idx}
        return obs, reward, done, info

    def human_play(self):
        """Load the scenario and give control back to the human player (no sensor output)."""
        if self.bng is None:
            self._launch(human_control=True)
        else:
            self._load_scenario(human_control=True)

        self._waypoint_idx = 0
        self._update_active_marker(0)

        self.bng.resume()
        print("[BeamNGDrivingEnv] Human control active — drive in-game. Press Ctrl+C to stop.")

        try:
            while True:
                self.vehicle.poll_sensors()  # keeps vehicle state cache fresh
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[BeamNGDrivingEnv] Human play stopped.")

    def human_play_lidar(self):
        """Human play with LiDAR bins printed to stdout each tick."""
        if self.bng is None:
            self._launch(human_control=True)
        else:
            self._load_scenario(human_control=True)

        self._waypoint_idx = 0
        self._update_active_marker(0)

        self.bng.resume()
        print(
            "[BeamNGDrivingEnv] Human control active (LiDAR) — drive in-game. Press Ctrl+C to stop."
        )
        if self.lidar is None:
            print(
                "[BeamNGDrivingEnv] Warning: LiDAR sensor not attached — bins will show fallback values."
            )

        try:
            while True:
                self.vehicle.poll_sensors()
                state = self.vehicle.state or {}
                pos = state.get("pos", (0.0, 0.0, 0.0))
                vel = state.get("vel", (1.0, 0.0, 0.0))
                dir_vec = state.get("dir", vel)
                vehicle_heading = float(np.arctan2(dir_vec[1], dir_vec[0]))

                lidar_data = (
                    self.lidar.poll().get("pointCloud", None) if self.lidar is not None else None
                )
                lidar_bins = self._process_lidar(lidar_data, pos, vehicle_heading)
                print(f"[LiDAR bins] {' '.join(f'{v:.2f}' for v in lidar_bins)}")
                d = self._lidar_debug
                if d:
                    print(
                        f"[LiDAR dbg] total={d.get('total', 0)} self={d.get('self', 0)} "
                        f"ground={d.get('ground', 0)} kept={d.get('kept', 0)} "
                        f"fov={d.get('fov', 0)} extents_none={d.get('extents_none')} "
                        f"nearest={d.get('min_dist_m', float('nan')):.1f}m "
                        f"z={d.get('min_dist_z', float('nan')):+.2f} "
                        f"ground_z={d.get('ground_z', float('nan')):+.2f}"
                    )

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[BeamNGDrivingEnv] Human play stopped.")

    def close(self, kill_sim: bool = True):
        """Close this environment.

        BeamNGpy `close()` kills the BeamNG process by default. Human-play uses
        `kill_sim=False` so returning to the menu or switching options only
        disconnects this client and leaves the already-open game running.
        """
        if self.bng is not None:
            self._remove_lidar()
            self._remove_roads_sensor()
            close_fn = self.bng.close if kill_sim else self.bng.disconnect
            t = threading.Thread(target=close_fn, daemon=True)
            t.start()
            t.join(timeout=5.0)
            self.bng = None
            self.vehicle = None

    def _remove_lidar(self):
        """Detach the current LiDAR before replacing the ego vehicle/scenario."""
        if self.lidar is None:
            return
        t = threading.Thread(target=self.lidar.remove, daemon=True)
        t.start()
        t.join(timeout=3.0)
        self.lidar = None

    def _attach_roads_sensor(self):
        """Attach a RoadsSensor when wheel_terrain is on; replace any prior one."""
        if not self.wheel_terrain:
            return
        self._remove_roads_sensor()
        self.roads_sensor = RoadsSensor("roads", self.bng, self.vehicle)

    def _remove_roads_sensor(self):
        if getattr(self, "roads_sensor", None) is None:
            return
        t = threading.Thread(target=self.roads_sensor.remove, daemon=True)
        t.start()
        t.join(timeout=3.0)
        self.roads_sensor = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch(self, human_control=False):
        """Start BeamNG.drive and load the scenario for the first time."""
        self.bng = BeamNGpy(
            self.host,
            self.port,
            home=self.beamng_home,
            user=self.beamng_user,
            headless=self.headless,
        )
        self.bng.open(launch=True)
        self.trajectory = self._resolve_trajectory()
        self.waypoints = self._select_waypoints()
        self._current_pos = self.trajectory.spawn_pos
        self._load_scenario(human_control=human_control)

    def _resolve_trajectory(self) -> TrajectoryData:
        """Load cached trajectories (all paths); default to the longest road."""
        from core.trajectory import CACHE_DIR

        cache_path = CACHE_DIR / f"{self.map_name}.json"
        if cache_path.exists():
            self._paths = load_or_generate(self.map_name, bng=None).paths
        else:
            # No cache → run a probe scenario so we can call get_road_network
            probe = Scenario(self.map_name, "trajectory_probe", description="Road probe")
            probe_vehicle = Vehicle("probe_vehicle", model="etk800")
            probe.add_vehicle(probe_vehicle, pos=(0.0, 0.0, 100.0), rot_quat=(0.0, 0.0, 0.0, 1.0))
            probe.make(self.bng)
            self.bng.load_scenario(probe)
            self.bng.start_scenario()
            time.sleep(0.5)
            self._paths = load_or_generate(self.map_name, self.bng).paths
        self.trajectory = self._paths[0]
        return self.trajectory

    def _randomize_waypoints(self):
        self.waypoints = random.sample(self.waypoints, len(self.waypoints))

    def _pick_episode_path(self) -> None:
        """When random_path is on, choose a random path for the next episode."""
        if not self.random_path or not self._paths:
            return
        self.trajectory = random.choice(self._paths)
        self.waypoints = self._select_waypoints()

    def _load_scenario(self, human_control=False):
        # self._randomize_waypoints()
        # A LiDAR is bound to the current BeamNG vehicle. Remove it before
        # replacing `self.vehicle` so changing vehicle/model cannot leave a
        # stale sensor attached to the previous ego.
        self._remove_lidar()
        self._remove_roads_sensor()
        self._ego_local_extents = None
        self.scenario = Scenario(
            self.map_name,
            "rl_driving",
            description="RL Training Scenario",
        )

        vcfg = self.VEHICLES.get(self.vehicle_id, self.VEHICLES["taxi"])
        self.vehicle = Vehicle("ego_vehicle", **vcfg)
        self.electrics = Electrics()
        self.damage_sensor = Damage()
        self.vehicle.attach_sensor("electrics", self.electrics)
        self.vehicle.attach_sensor("damage", self.damage_sensor)

        self.scenario.add_vehicle(
            self.vehicle,
            pos=self.trajectory.spawn_pos,
            rot_quat=self.trajectory.spawn_rot,
            cling=True,
        )

        # Visual checkpoint rings for every waypoint (visible in-game as hoops, training and human play).
        checkpoint_wps = (
            [wp for p in self._paths for wp in p.sparse_waypoints]
            if self.random_path
            else self.waypoints
        )
        scales = [(5.0, 5.0, 1.0)] * len(checkpoint_wps)
        self.scenario.add_checkpoints(checkpoint_wps, scales)

        self.scenario.make(self.bng)
        self.bng.set_deterministic(30)  # ensure repeatable physics for same scenario
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(1.0)  # let the game settle before polling

        self._cache_ego_local_bbox()

        # Lidar must be created after the scenario starts (it communicates with the sim directly).
        self.lidar = Lidar(
            "lidar",
            self.bng,
            self.vehicle,
            **self._lidar_creation_kwargs(),
        )

        # Draw the initial active-waypoint marker
        self._update_active_marker(0)
        self._attach_roads_sensor()

    def _observe(self) -> np.ndarray:
        """Poll sensors and return the normalized observation vector (length n_states)."""
        self.vehicle.poll_sensors()

        elec = self.electrics.data or {}
        dmg = self.damage_sensor.data or {}

        speed = float(elec.get("wheelspeed", 0.0))
        steering = float(elec.get("steering", 0.0))
        damage = float(dmg.get("damage", 0.0))

        state = self.vehicle.state or {}
        pos = state.get("pos", (0.0, 0.0, 0.0))
        vel = state.get("vel", (1.0, 0.0, 0.0))
        dir_vec = state.get("dir", vel)
        vehicle_heading = float(np.arctan2(dir_vec[1], dir_vec[0]))

        heading_err, lateral_err, dist = self._path_errors(pos, state)

        lidar_bins = self._process_lidar(
            self.lidar.poll().get("pointCloud", None) if self.lidar is not None else None,
            pos,
            vehicle_heading,
        )

        self._current_pos = pos
        if self.waypoints:
            target = self.waypoints[self._waypoint_idx % len(self.waypoints)]
            self._checkpoint_dist = float(np.hypot(pos[0] - target[0], pos[1] - target[1]))

        waypoint_hints = self._get_waypoint_hints(pos, vehicle_heading)

        obs = np.concatenate(
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
                lidar_bins,
                waypoint_hints,
                self._extra_features(state),
            ]
        )

        return obs

    def _lidar_config(self) -> LidarConfig:
        return LidarConfig(
            rays=self.LIDAR_RAYS,
            v_bins=self.LIDAR_V_BINS,
            channels=self.LIDAR_CHANNELS_PER_RAY,
            fov_deg=self.LIDAR_FOV_DEG,
            vert_angle=self.LIDAR_VERT_ANGLE,
            max_dist=self.LIDAR_MAX_DIST,
            self_margin=self.LIDAR_SELF_MARGIN,
            ground_clearance=self.LIDAR_GROUND_CLEARANCE,
        )

    def _cache_ego_local_bbox(self):
        """Sample the ego OBB once and store its extents in vehicle-local frame.

        The bbox is queried in world space but, expressed relative to the vehicle
        reference node and de-rotated by the current heading, the extents are
        invariant under rigid motion — so caching once per scenario load is
        enough. Used by _process_lidar to reject self-hits geometrically.
        """
        try:
            # State must be fresh: get_bbox() returns WORLD coords, so we need the
            # current world pos/heading to de-rotate them. Without a poll, state is
            # None at scenario-load time and pos would default to (0,0,0), producing
            # world-scale garbage extents that disable the LiDAR. Poll first.
            self.vehicle.poll_sensors()
            bbox = self.vehicle.get_bbox()
        except Exception:
            self._ego_local_extents = None
            return

        state = self.vehicle.state or {}
        # Bail rather than guess: a missing pos here means we cannot align the
        # world-space bbox to the local frame, and a wrong box is worse than none
        # (the ground/self filters fall back to safe defaults when extents is None).
        if not bbox or "pos" not in state:
            self._ego_local_extents = None
            return

        corners = np.asarray(list(bbox.values()), dtype=np.float32)
        pos = np.asarray(state.get("pos", (0.0, 0.0, 0.0)), dtype=np.float32)
        dir_vec = np.asarray(state.get("dir", (1.0, 0.0, 0.0)), dtype=np.float32)
        heading = float(np.arctan2(dir_vec[1], dir_vec[0]))

        rel = corners - pos
        c, s = np.cos(-heading), np.sin(-heading)
        lx = rel[:, 0] * c - rel[:, 1] * s
        ly = rel[:, 0] * s + rel[:, 1] * c
        lz = rel[:, 2]
        m = self.LIDAR_SELF_MARGIN
        self._ego_local_extents = (
            float(lx.min() - m),
            float(lx.max() + m),
            float(ly.min() - m),
            float(ly.max() + m),
            float(lz.min() - m),
            float(lz.max() + m),
        )

    def _lidar_keep_mask(self, local_x, local_y, local_z) -> np.ndarray:
        """Reject points that are (a) inside the ego OBB or (b) ground returns.

        Ground threshold is relative to the ego bbox floor when available, so it
        tracks the actual vehicle ride height instead of assuming a ref-node-at-
        ground convention that varies per Jbeam.
        """
        n_total = int(local_x.size)
        inside_self = np.zeros(n_total, dtype=bool)

        if self._ego_local_extents is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = self._ego_local_extents
            inside_self = (
                (local_x >= x_min)
                & (local_x <= x_max)
                & (local_y >= y_min)
                & (local_y <= y_max)
                & (local_z >= z_min)
                & (local_z <= z_max)
            )
            # z_min already has -LIDAR_SELF_MARGIN baked in, so add it back to
            # recover the TRUE bbox floor, then require points to clear it by
            # LIDAR_GROUND_CLEARANCE. (Using z_min directly cancelled the margin
            # against the clearance, leaving ~0 real clearance, so ground returns
            # leaked through and every bin read a spurious mid-range distance.)
            floor = z_min + self.LIDAR_SELF_MARGIN
            ground_z = floor + self.LIDAR_GROUND_CLEARANCE
        else:
            ground_z = self.LIDAR_GROUND_CLEARANCE

        below_ground = local_z <= ground_z
        keep = ~inside_self & ~below_ground

        self._lidar_debug = {
            "total": n_total,
            "self": int(inside_self.sum()),
            "ground": int((below_ground & ~inside_self).sum()),
            "kept": int(keep.sum()),
            "extents_none": self._ego_local_extents is None,
            "ground_z": float(ground_z),
        }
        return keep

    def _resolve_lidar_mount_pos(self) -> tuple[float, float, float]:
        """Return a vehicle-local LiDAR seed near the roof for BeamNG snapping.

        BeamNG's LiDAR supports `is_snapping_desired`: the simulator moves the
        requested vehicle-space `pos` to the nearest vehicle triangle. The seed
        should be close to the desired surface. A point just above the bbox roof
        lets BeamNG pick the actual roof triangle. Falls back to the configured
        constant if bbox sampling failed.
        """
        if self._ego_local_extents is None:
            return self.LIDAR_MOUNT_POS
        _, _, _, _, _, z_max = self._ego_local_extents
        return (0.0, 0.0, float(z_max + self.LIDAR_SELF_MARGIN))

    def _lidar_creation_kwargs(self) -> dict:
        """Return BeamNGpy LiDAR kwargs shared by scenario creation and tests."""
        return {
            "pos": self._resolve_lidar_mount_pos(),
            "dir": self.LIDAR_MOUNT_DIR,
            "up": self.LIDAR_MOUNT_UP,
            "requested_update_time": 0.05,
            "frequency": 30,
            "vertical_resolution": self.LIDAR_VERT_RES,
            "vertical_angle": self.LIDAR_VERT_ANGLE,
            "horizontal_angle": self.LIDAR_FOV_DEG,
            "max_distance": self.LIDAR_MAX_DIST,
            "is_360_mode": False,
            "is_rotate_mode": False,
            "is_using_shared_memory": False,
            "is_visualised": LIDAR_VISUALISE,
            "is_snapping_desired": True,
            "is_force_inside_triangle": True,
        }

    def _process_lidar(self, point_cloud, vehicle_pos, vehicle_heading) -> np.ndarray:
        """Bin a raw LiDAR point cloud into a LIDAR_V_BINS x LIDAR_RAYS grid.

        Pipeline: world -> ego-local -> self/ground filter -> forward-FOV mask ->
        elevation+azimuth binning. Returns a flat float32 array of shape
        (LIDAR_V_BINS * LIDAR_RAYS * LIDAR_CHANNELS_PER_RAY,) in [0, 1], where 0
        means an obstacle is right there and 1 means the cell is clear.

        Layout is row-major by vertical bin then horizontal bin then channel:
        index = (v * LIDAR_RAYS + h) * LIDAR_CHANNELS_PER_RAY + c. With
        LIDAR_V_BINS == 1 this reduces to the legacy single row of LIDAR_RAYS
        values (vertical structure collapsed), so existing models stay valid.
        """
        v_bins = self.LIDAR_V_BINS
        h_bins = self.LIDAR_RAYS
        ch = self.LIDAR_CHANNELS_PER_RAY
        n_out = v_bins * h_bins * ch
        distances = np.ones(n_out, dtype=np.float32)  # default: clear

        if point_cloud is None or len(point_cloud) == 0:
            if LOG_LIDAR and self.bng is not None:
                self.bng.queue_lua_command("log('I', 'RL', 'Lidar: no points')")
            return distances

        pts = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)

        rel = pts - np.asarray(vehicle_pos, dtype=np.float32)
        cos_h = np.cos(-vehicle_heading)
        sin_h = np.sin(-vehicle_heading)
        local_x = rel[:, 0] * cos_h - rel[:, 1] * sin_h
        local_y = rel[:, 0] * sin_h + rel[:, 1] * cos_h
        local_z = rel[:, 2]

        keep = self._lidar_keep_mask(local_x, local_y, local_z)
        local_x = local_x[keep]
        local_y = local_y[keep]
        local_z = local_z[keep]
        if local_x.size == 0:
            if LOG_LIDAR and self.bng is not None:
                self.bng.queue_lua_command("log('I', 'RL', 'Lidar: all points filtered')")
            return distances

        angles = np.arctan2(local_y, local_x)
        dists = np.hypot(local_x, local_y)

        half_fov = np.radians(self.LIDAR_FOV_DEG / 2.0)
        in_fov = np.abs(angles) <= half_fov
        angles = angles[in_fov]
        dists = dists[in_fov]
        local_z = local_z[in_fov]
        if angles.size == 0:
            if LOG_LIDAR and self.bng is not None:
                self.bng.queue_lua_command("log('I', 'RL', 'Lidar: all points outside FOV')")
            return distances

        nearest = int(np.argmin(dists))
        self._lidar_debug["fov"] = int(angles.size)
        self._lidar_debug["min_dist_m"] = float(dists[nearest])
        self._lidar_debug["min_dist_z"] = float(local_z[nearest])

        h_edges = np.linspace(-half_fov, half_fov, h_bins + 1)
        h_idx = np.clip(np.digitize(angles, h_edges) - 1, 0, h_bins - 1)

        if v_bins == 1:
            v_idx = np.zeros(angles.shape, dtype=np.intp)
        else:
            half_vfov = np.radians(self.LIDAR_VERT_ANGLE / 2.0)
            elevation = np.arctan2(local_z, dists)
            v_edges = np.linspace(-half_vfov, half_vfov, v_bins + 1)
            v_idx = np.clip(np.digitize(elevation, v_edges) - 1, 0, v_bins - 1)

        for v in range(v_bins):
            for h in range(h_bins):
                sel = dists[(v_idx == v) & (h_idx == h)]
                if sel.size:
                    distances[(v * h_bins + h) * ch] = np.clip(
                        sel.min() / self.LIDAR_MAX_DIST, 0.0, 1.0
                    )

        if LOG_LIDAR and self.bng is not None:
            self.bng.queue_lua_command(
                "log('I', 'RL', 'Lidar: [{}]')".format(", ".join(f"{v:.3f}" for v in distances))
            )
        return distances

    def _get_waypoint_hints(self, pos, vehicle_heading) -> np.ndarray:
        """Return vehicle-local (forward, left) coords for the next trajectory_hints waypoints.

        Each waypoint contributes 2 floats normalized to [-1, 1] over a 100 m range.
        Returns an empty array when trajectory_hints == 0.
        """
        if not self.trajectory_hints or not self.waypoints:
            return np.empty(0, dtype=np.float32)
        NORM = 100.0
        cos_h = np.cos(-vehicle_heading)
        sin_h = np.sin(-vehicle_heading)
        hints: list[float] = []
        for i in range(self.trajectory_hints):
            idx = (self._waypoint_idx + i) % len(self.waypoints)
            wp = self.waypoints[idx]
            rel_x = wp[0] - pos[0]
            rel_y = wp[1] - pos[1]
            local_x = rel_x * cos_h - rel_y * sin_h
            local_y = rel_x * sin_h + rel_y * cos_h
            hints.append(float(np.clip(local_x / NORM, -1.0, 1.0)))
            hints.append(float(np.clip(local_y / NORM, -1.0, 1.0)))
        return np.array(hints, dtype=np.float32)

    def _body_orientation_features(self, state) -> np.ndarray:
        """[pitch, roll] from the vehicle's forward/up vectors (see geometry helper)."""
        return body_orientation_features(
            state.get("dir", (0.0, 1.0, 0.0)), state.get("up", (0.0, 0.0, 1.0))
        )

    def _wheel_terrain_features(self) -> np.ndarray:
        """[left, right] road-edge position from the RoadsSensor (neutral without one)."""
        payload = self.roads_sensor.poll() if self.roads_sensor is not None else None
        return wheel_terrain_features(payload, self.HALF_TRACK_WIDTH)

    def _extra_features(self, state) -> np.ndarray:
        """Optional observation tail: body orientation and/or wheel terrain.

        Appended after the waypoint hints. Empty array when both flags are off,
        so flag-off observations are unchanged.
        """
        blocks = []
        if self.body_orientation:
            blocks.append(self._body_orientation_features(state))
        if self.wheel_terrain:
            blocks.append(self._wheel_terrain_features())
        if not blocks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(blocks)

    def _path_errors(self, pos, state):
        """Return (heading_error_rad, lateral_error_m) relative to next waypoint.

        Also stores self._current_dist for progress reward computation.
        """
        if not self.waypoints or not state:
            self._current_dist = 0.0
            return 0.0, 0.0

        target = self.waypoints[self._waypoint_idx % len(self.waypoints)]
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = float(np.hypot(dx, dy))
        self._current_dist = dist

        # Advance waypoint when close enough
        if dist < self.WAYPOINT_RADIUS:
            self._waypoint_idx += 1
            self._checkpoint_hit = True
            self._update_active_marker(self._waypoint_idx)
            if LOG_CHECKPOINT_HIT:
                self.bng.queue_lua_command("log('I', 'RL', 'checkpoint hit')")
            if self._waypoint_idx < len(self.waypoints):
                new_t = self.waypoints[self._waypoint_idx]
                self._current_dist = float(np.hypot(new_t[0] - pos[0], new_t[1] - pos[1]))

        vel = state.get("vel", (1.0, 0.0, 0.0))
        vehicle_heading = np.arctan2(vel[1], vel[0])
        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi

        lateral_err = dist * np.sin(heading_err)
        return float(heading_err), float(lateral_err), dist

    def _compute_reward(self, obs):
        if self.reward_mode == "ddpg":
            return self._compute_reward_ddpg(obs)
        return self._compute_reward_default(obs)

    def _compute_reward_default(self, obs):
        """Original reward function for discrete-action algorithms (DQN, Q-learning)."""
        speed, steering, heading_err, lateral_err, damage_norm = obs[:5]
        damage = damage_norm * 1000.0

        done = False
        reward = 0.0

        # 4. Penalise being stationary — the agent must move
        if speed < 0.05:
            reward -= 2.0

        # 5. Penalise excessive steering
        reward -= abs(steering) * 0.2

        # 6. Penalise (and terminate on) significant damage
        if damage > self._last_damage + 50:
            reward -= 50.0
        if damage >= self.MAX_DAMAGE:
            done = True
        self._last_damage = damage

        # 7. Step limit
        if self._steps >= self.MAX_STEPS:
            done = True

        # 8. Checkpoint bonus (big reward for reaching waypoints)
        if self._checkpoint_hit:
            reward += 100.0 * self._waypoint_idx
            self._checkpoint_hit = False

        # 9. Lap completion bonus
        if self._waypoint_idx >= len(self.waypoints):
            reward += 200.0
            done = True

        # Distance-from-checkpoint penalty
        dist = self._checkpoint_dist
        if dist >= self.CHECKPOINT_RESET_DIST:
            # Too far from checkpoint: big malus and teleport back to spawn
            reward -= 100.0
            done = True
            if LOG_CHECKPOINT_RESPAWN:
                self.bng.queue_lua_command("log('I', 'RL', 'too far from checkpoint — respawned')")
        elif dist >= self.CHECKPOINT_WARN_DIST:
            # Getting off track: proportional penalty
            reward -= (
                (dist - self.CHECKPOINT_WARN_DIST)
                / (self.CHECKPOINT_RESET_DIST - self.CHECKPOINT_WARN_DIST)
                * 10.0
            )
            if LOG_CHECKPOINT_WARN:
                self.bng.queue_lua_command("log('I', 'RL', 'too far from checkpoint — minus')")

        return float(reward), done

    def _compute_reward_ddpg(self, obs):
        """Reward function optimized for continuous-action algorithms (DDPG, TD3).

        Main signals:
        1. Progress toward next waypoint (getting closer = good)
        2. Speed projected onto waypoint direction (driving toward it = good)
        3. Checkpoint bonuses (reaching waypoints = very good)
        4. Penalties: obstacles, damage, out of bounds
        """
        speed, _steering, heading_err, _lateral_err, damage_norm = obs[:5]
        lidar_bins = obs[5:]
        damage = damage_norm * 1000.0
        # heading_err is normalized by pi in obs, undo it for cos
        alignment = np.cos(heading_err * np.pi)

        done = False
        reward = 0.0

        # 1. Progress reward: bonus for getting closer to waypoint, penalty for drifting away
        dist_delta = self._last_dist - self._current_dist  # positive = getting closer
        reward += dist_delta * 3.0
        self._last_dist = self._current_dist

        # 2. Speed projected toward waypoint: speed * cos(heading_error)
        reward += speed * alignment * 3.0

        # 3. Small alignment bonus even when slow
        reward += alignment * 0.5

        # 4. Penalise being stationary
        if speed < 0.05:
            reward -= 1.0

        # 5. Penalise obstacle proximity (LiDAR)
        min_lidar = float(np.min(lidar_bins))
        if min_lidar < 0.2:
            reward -= (1.0 - min_lidar) * 5.0
        elif min_lidar < 0.4:
            reward -= (1.0 - min_lidar) * 2.0

        # 6. Damage
        damage_delta = damage - self._last_damage
        if damage_delta > 0:
            reward -= damage_delta * 0.3
        if damage_delta > 150:
            reward -= 30.0
            done = True
        if damage >= self.MAX_DAMAGE:
            done = True
        self._last_damage = damage

        # 7. Step limit
        if self._steps >= self.MAX_STEPS:
            done = True

        # 8. Checkpoint bonus
        if self._checkpoint_hit:
            reward += 50.0
            self._checkpoint_hit = False

        # 9. Lap completion
        if self._waypoint_idx >= len(self.waypoints):
            reward += 200.0
            self._waypoint_idx = 0
            done = True

        return float(reward), done

    def _update_active_marker(self, idx: int):
        """Draw a bright sphere in-game on the current target waypoint.

        Uses bng.debug (beamngpy >= 1.26).  Silently skipped on older builds.
        """
        if self.bng is None:
            return
        try:
            debug = self.bng.debug
            if self._active_marker_id is not None:
                try:
                    debug.remove_spheres([self._active_marker_id])
                except Exception:
                    pass

            target = self.waypoints[idx % len(self.waypoints)]
            pos = (target[0], target[1], target[2] + 2.0)
            ids = debug.add_spheres(
                coordinates=[pos],
                radii=[3.0],
                rgba_colors=[(0.0, 1.0, 0.2, 0.8)],
            )
            self._active_marker_id = ids[0]
        except AttributeError:
            pass


class BeamNGLidarEnv(BeamNGDrivingEnv):
    """
    Discrete-action BeamNG env exposing the LiDAR as a 2D depth grid
    (vertical layers x horizontal bins) instead of a single collapsed row.

    Each observation cell holds the nearest-obstacle distance for one
    (elevation, azimuth) slice, normalized to [0, 1] (0 = obstacle here,
    1 = clear). The vertical dimension lets the policy reason about obstacle
    *height* (a wall fills several vertical bins; a low object only the bottom
    one), which a single row cannot represent.

    Same 7 discrete actions and reward as the base `beamng` env; only the
    observation's LiDAR block grows from 8 to LIDAR_V_BINS * LIDAR_RAYS values.
    Self-hit and ground filtering are unchanged, so the ego is still never
    detected and asphalt does not flood the lower rows.
    """

    LIDAR_V_BINS = 4  # vertical elevation bins
    LIDAR_VERT_RES = 16  # more layers to populate the wider vertical FOV
    LIDAR_VERT_ANGLE = 20.0  # wider vertical FOV (±10°) so the rows span useful elevations

    # 6 kinematic + (4 vertical × 8 horizontal × 1 channel) = 38
    N_STATES = (
        6 + BeamNGDrivingEnv.LIDAR_RAYS * LIDAR_V_BINS * BeamNGDrivingEnv.LIDAR_CHANNELS_PER_RAY
    )


class BeamNGContinuousEnv(BeamNGDrivingEnv):
    """
    BeamNG environment with a 3D continuous action space.

    The algorithm directly controls throttle, steering, and brake as separate
    outputs. The actor produces values in [-1, 1] (Tanh); they are mapped as:
        action[0] -> throttle in [0, 1]   (negative half is ignored / zeroed)
        action[1] -> steering in [-1, 1]  (used directly)
        action[2] -> brake    in [0, 1]   (negative half is ignored / zeroed)

    Uses dense waypoints and the DDPG-style reward by default.
    """

    N_ACTIONS = 3

    def __init__(
        self,
        beamng_home: str,
        beamng_user: str = None,
        host: str = "localhost",
        port: int = 25252,
        headless: bool = False,
        vehicle_id: str = "taxi",
        map_name: str = "gridmap_v2",
        trajectory_hints: int = 0,
        body_orientation: bool = False,
        wheel_terrain: bool = False,
        random_path: bool = False,
    ):
        super().__init__(
            beamng_home=beamng_home,
            beamng_user=beamng_user,
            host=host,
            port=port,
            headless=headless,
            reward_mode="default",
            vehicle_id=vehicle_id,
            map_name=map_name,
            trajectory_hints=trajectory_hints,
            body_orientation=body_orientation,
            wheel_terrain=wheel_terrain,
            random_path=random_path,
        )

    def step(self, action):
        """
        Apply a continuous action and advance the simulation.

        action: np.ndarray of shape (2,) in [-1, 1]
            [acceleration, steering]  — positive acceleration = throttle, negative = brake
        OR shape (3,) in [-1, 1]
            [throttle_raw, steering, brake_raw]
        """
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if action.shape[0] == 2:
            # 2D: single acceleration channel encodes throttle (+) and brake (-)
            throttle = float(max(0.0, action[0]))
            steering = float(action[1])
            brake = float(max(0.0, -action[0]))
        else:
            throttle = float(max(0.0, action[0]))
            steering = float(action[1])
            brake = float(max(0.0, action[2]))

        self.vehicle.control(throttle=throttle, steering=steering, brake=brake)
        self.bng.step(10)
        self._steps += 1

        obs = self._observe()
        reward, done = self._compute_reward(obs)
        info = {"steps": self._steps, "waypoint_idx": self._waypoint_idx}
        return obs, reward, done, info


class BeamNGCameraEnv(BeamNGContinuousEnv):
    """
    BeamNG continuous-action environment using a front-facing dashcam
    instead of LiDAR for perception.

    State (5 + CAM_PIXELS floats):
        speed, steering, heading_error, lateral_error, damage  — kinematic
        cam[0..N-1]  — flattened grayscale pixels, normalized to [0, 1]

    Actions: same 3D continuous as BeamNGContinuousEnv
        action[0] -> throttle [0, 1]  (actor output clipped to positive half)
        action[1] -> steering [-1, 1]
        action[2] -> brake    [0, 1]  (actor output clipped to positive half)
    """

    CAM_RESOLUTION = (84, 84)
    CAM_OUT_SIZE = (16, 16)
    N_ACTIONS = 3
    N_STATES = 6 + CAM_OUT_SIZE[0] * CAM_OUT_SIZE[1]  # 262

    def __init__(
        self,
        beamng_home: str,
        beamng_user: str = None,
        host: str = "localhost",
        port: int = 25252,
        headless: bool = False,
        vehicle_id: str = "taxi",
        map_name: str = "gridmap_v2",
        trajectory_hints: int = 0,
        body_orientation: bool = False,
        wheel_terrain: bool = False,
        random_path: bool = False,
    ):
        super().__init__(
            beamng_home=beamng_home,
            beamng_user=beamng_user,
            host=host,
            port=port,
            headless=headless,
            vehicle_id=vehicle_id,
            map_name=map_name,
            trajectory_hints=trajectory_hints,
            body_orientation=body_orientation,
            wheel_terrain=wheel_terrain,
            random_path=random_path,
        )
        self.camera: Camera = None
        self.last_frame: np.ndarray | None = None  # 2-D grayscale (CAM_OUT_SIZE), updated each step

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _load_scenario(self, human_control=False):
        if self.camera is not None:
            self.camera.remove()
            self.camera = None

        self.scenario = Scenario(
            self.map_name, "rl_driving_camera", description="RL Camera Training"
        )
        vcfg = self.VEHICLES.get(self.vehicle_id, self.VEHICLES["taxi"])
        self.vehicle = Vehicle("ego_vehicle", **vcfg)
        self.electrics = Electrics()
        self.damage_sensor = Damage()
        self.vehicle.attach_sensor("electrics", self.electrics)
        self.vehicle.attach_sensor("damage", self.damage_sensor)

        self.scenario.add_vehicle(
            self.vehicle, pos=self.trajectory.spawn_pos, rot_quat=self.trajectory.spawn_rot
        )

        # Visual checkpoint rings for every waypoint (visible in-game as hoops, training and human play).
        checkpoint_wps = (
            [wp for p in self._paths for wp in p.sparse_waypoints]
            if self.random_path
            else self.waypoints
        )
        scales = [(5.0, 5.0, 1.0)] * len(checkpoint_wps)
        self.scenario.add_checkpoints(checkpoint_wps, scales)

        self.scenario.make(self.bng)
        self.bng.set_deterministic(30)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(1.0)

        self.camera = Camera(
            "dashcam",
            self.bng,
            self.vehicle,
            pos=(0, -0.5, 1.5),
            dir=(0, -1, 0),
            field_of_view_y=70,
            resolution=self.CAM_RESOLUTION,
            is_render_colours=True,
            is_render_depth=False,
            is_render_annotations=False,
            is_visualised=True,
            is_static=False,
        )

        self._update_active_marker(0)
        self._attach_roads_sensor()

    def _observe(self) -> np.ndarray:
        self.vehicle.poll_sensors()

        elec = self.electrics.data or {}
        dmg = self.damage_sensor.data or {}

        speed = float(elec.get("wheelspeed", 0.0))
        steering = float(elec.get("steering", 0.0))
        damage = float(dmg.get("damage", 0.0))

        state = self.vehicle.state or {}
        pos = state.get("pos", (0.0, 0.0, 0.0))
        vel = state.get("vel", (1.0, 0.0, 0.0))
        dir_vec = state.get("dir", vel)
        vehicle_heading = float(np.arctan2(dir_vec[1], dir_vec[0]))

        heading_err, lateral_err, dist = self._path_errors(pos, state)
        cam_pixels = self._process_camera()
        waypoint_hints = self._get_waypoint_hints(pos, vehicle_heading)

        self._current_pos = pos
        if self.waypoints:
            target = self.waypoints[self._waypoint_idx % len(self.waypoints)]
            self._checkpoint_dist = float(np.hypot(pos[0] - target[0], pos[1] - target[1]))

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
                cam_pixels,
                waypoint_hints,
                self._extra_features(state),
            ]
        )

    def close(self):
        if self.camera is not None:
            t = threading.Thread(target=self.camera.remove, daemon=True)
            t.start()
            t.join(timeout=3.0)
            self.camera = None
        super().close()

    def human_play_camera(self):
        """Human play with the 16×16 dashcam frame rendered as ASCII art in-place."""
        if self.bng is None:
            self._launch(human_control=True)
        else:
            self._load_scenario(human_control=True)

        self._waypoint_idx = 0
        self._update_active_marker(0)

        self.bng.resume()
        print(
            "[BeamNGCameraEnv] Human control active (Camera) — drive in-game. Press Ctrl+C to stop."
        )

        ramp = " ░▒▓█"
        h = self.CAM_OUT_SIZE[0]
        first = True

        try:
            while True:
                pixels = self._process_camera().reshape(self.CAM_OUT_SIZE)
                rows = ["".join(ramp[min(int(v * 4), 4)] for v in row) for row in pixels]
                if not first:
                    sys.stdout.write(f"\033[{h}A")
                sys.stdout.write("\n".join(rows) + "\n")
                sys.stdout.flush()
                first = False
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[BeamNGCameraEnv] Human play stopped.")

    # ------------------------------------------------------------------
    # Camera processing
    # ------------------------------------------------------------------

    def _process_camera(self) -> np.ndarray:
        """Poll camera and return a flattened grayscale image normalized to [0, 1].

        Delegates the frame math to the shared `process_camera_frame` helper;
        keeps the poll, `last_frame` cache, and optional Lua logging here.
        """
        colour = self.camera.poll().get("colour", None) if self.camera is not None else None
        flat = process_camera_frame(colour, self.CAM_OUT_SIZE)
        self.last_frame = flat.reshape(self.CAM_OUT_SIZE)  # 2-D (CAM_OUT_SIZE), values in [0, 1]
        if LOG_CAMERA and self.bng is not None:
            mn, mx, avg = float(flat.min()), float(flat.max()), float(flat.mean())
            self.bng.queue_lua_command(
                f"log('I', 'RL', 'Camera: min={mn:.3f} max={mx:.3f} mean={avg:.3f}')"
            )
        return flat
