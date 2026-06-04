import random
import threading
import time

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera, Damage, Electrics, Lidar

from config import (
    LOG_CAMERA,
    LOG_CHECKPOINT_HIT,
    LOG_CHECKPOINT_RESPAWN,
    LOG_CHECKPOINT_WARN,
    LOG_LIDAR,
)
from core.trajectory import TrajectoryData, load_or_generate


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
    LIDAR_RAYS = 8  # number of angular bins
    LIDAR_FOV_DEG = 120.0  # total forward-facing field of view in degrees
    LIDAR_MAX_DIST = 50.0  # metres — normalization range

    N_STATES = 6 + LIDAR_RAYS  # 6 kinematic + 8 lidar = 14

    CHECKPOINT_WARN_DIST = 200.0  # metres — start penalising when this far from checkpoint
    CHECKPOINT_RESET_DIST = 300.0  # metres — teleport back to spawn and big malus beyond this

    WAYPOINT_RADIUS = 8.0  # metres — how close before advancing to next waypoint
    MAX_STEPS = 500
    MAX_DAMAGE = 1000.0  # damage threshold that ends the episode

    AVAILABLE_MAPS = ["gridmap_v2", "italy", "west_coast_usa", "smallgrid"]

    VEHICLES = {
        "taxi": {
            "model": "burnside",
            "licence": "Taxi",
            "color": "Yellow",
            "part_config": "vehicles/burnside/4door_early_v8_3M_taxi.pc",
        },
        "gavril_t_series": {
            "model": "us-semi",
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
        self.n_states = self.N_STATES + trajectory_hints * 2

        # Filled on first _launch() — either read from cache or generated then.
        self.trajectory: TrajectoryData | None = None
        self.waypoints: list[tuple[float, float, float]] = []
        self._current_pos = (0.0, 0.0, 0.0)

    def _select_waypoints(self) -> list[tuple[float, float, float]]:
        assert self.trajectory is not None
        use_dense = self.reward_mode == "ddpg" or isinstance(self, BeamNGContinuousEnv)
        return list(
            self.trajectory.dense_waypoints if use_dense else self.trajectory.dense_waypoints
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the episode and return the initial observation."""
        if self.bng is None:
            self._launch()
        else:
            # self._randomize_waypoints()
            self.bng.scenario.restart()
            # Debug-draw spheres are wiped on scenario restart — redraw them.
            self._update_active_marker(1)
            self._draw_start_end_markers()

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
        """Load the scenario and give control back to the human player.

        The simulation runs in real-time — drive with your keyboard/controller
        inside BeamNG as normal.
        """
        if self.bng is None:
            self._launch(human_control=True)
        else:
            self._load_scenario(human_control=True)

        self._waypoint_idx = 0
        self._update_active_marker(1)

        # Release the sim from step-mode so it runs freely in real-time.
        self.bng.resume()
        print("[BeamNGDrivingEnv] Human control active — drive in-game. Press Ctrl+C to stop.")

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

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[BeamNGDrivingEnv] Human play stopped.")

    def close(self):
        """Shut down the BeamNG connection."""
        if self.bng is not None:
            if self.lidar is not None:
                t = threading.Thread(target=self.lidar.remove, daemon=True)
                t.start()
                t.join(timeout=3.0)
                self.lidar = None
            t = threading.Thread(target=self.bng.close, daemon=True)
            t.start()
            t.join(timeout=5.0)
            self.bng = None
            self.vehicle = None

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
        """Return cached trajectory or probe the map to generate one."""
        from core.trajectory import CACHE_DIR

        cache_path = CACHE_DIR / f"{self.map_name}.json"
        if cache_path.exists():
            return load_or_generate(self.map_name, bng=None)

        # No cache → run a probe scenario so we can call get_road_network
        probe = Scenario(self.map_name, "trajectory_probe", description="Road probe")
        probe_vehicle = Vehicle("probe_vehicle", model="etk800")
        probe.add_vehicle(probe_vehicle, pos=(0.0, 0.0, 100.0), rot_quat=(0.0, 0.0, 0.0, 1.0))
        probe.make(self.bng)
        self.bng.load_scenario(probe)
        self.bng.start_scenario()
        time.sleep(0.5)
        return load_or_generate(self.map_name, self.bng)

    def _randomize_waypoints(self):
        self.waypoints = random.sample(self.waypoints, len(self.waypoints))

    def _load_scenario(self, human_control=False):
        # self._randomize_waypoints()
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
        scales = [(5.0, 5.0, 1.0)] * len(self.waypoints)
        self.scenario.add_checkpoints(self.waypoints, scales)

        self.scenario.make(self.bng)
        self.bng.set_deterministic(30)  # ensure repeatable physics for same scenario
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(1.0)  # let the game settle before polling

        # Lidar must be created after the scenario starts (it communicates with the sim directly)
        if self.lidar is not None:
            self.lidar.remove()
        self.lidar = Lidar(
            "lidar",
            self.bng,
            self.vehicle,
            pos=(0, 0, 1.7),
            dir=(0, -1, 0),
            up=(0, 0, 1),
            vertical_resolution=16,
            vertical_angle=10,
            horizontal_angle=self.LIDAR_FOV_DEG,
            max_distance=self.LIDAR_MAX_DIST,
            is_360_mode=False,
            is_rotate_mode=False,
            is_using_shared_memory=False,
            is_visualised=True,
        )

        # Draw the initial active-waypoint marker
        self._update_active_marker(1)
        self._draw_start_end_markers()

    def _observe(self) -> np.ndarray:
        """Poll sensors and return a normalized state vector (7 floats)."""
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
            ]
        )

        return obs

    def _process_lidar(self, point_cloud, vehicle_pos, vehicle_heading) -> np.ndarray:
        """Bin a raw LiDAR point cloud into LIDAR_RAYS angular distance slices.

        Points are in world space and are transformed to vehicle-local coordinates
        using the vehicle's position and heading before binning.
        Returns a float32 array of shape (LIDAR_RAYS,) with values in [0, 1],
        where 0 means an obstacle is right there and 1 means the bin is clear.
        """
        distances = np.ones(self.LIDAR_RAYS, dtype=np.float32)  # default: clear

        if point_cloud is None or len(point_cloud) == 0:
            if LOG_LIDAR:
                self.bng.queue_lua_command("log('I', 'RL', 'Lidar: no points')")
            return distances

        pts = np.array(point_cloud, dtype=np.float32).reshape(-1, 3)

        # Translate: move points relative to vehicle position
        rel_x = pts[:, 0] - vehicle_pos[0]
        rel_y = pts[:, 1] - vehicle_pos[1]

        # Rotate: transform into vehicle-local frame (x=forward, y=left)
        cos_h = np.cos(-vehicle_heading)
        sin_h = np.sin(-vehicle_heading)
        local_x = rel_x * cos_h - rel_y * sin_h
        local_y = rel_x * sin_h + rel_y * cos_h

        # Work in the horizontal (XY) plane only
        angles = np.arctan2(local_y, local_x)
        dists = np.hypot(local_x, local_y)

        # Keep only points within the forward FOV
        half_fov = np.radians(self.LIDAR_FOV_DEG / 2.0)
        mask = np.abs(angles) <= half_fov
        angles = angles[mask]
        dists = dists[mask]

        if len(angles) == 0:
            if LOG_LIDAR:
                self.bng.queue_lua_command("log('I', 'RL', 'Lidar: all points outside FOV')")
            return distances

        bin_edges = np.linspace(-half_fov, half_fov, self.LIDAR_RAYS + 1)
        for i in range(self.LIDAR_RAYS):
            in_bin = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
            if in_bin.any():
                nearest = float(dists[in_bin].min())
                distances[i] = np.clip(nearest / self.LIDAR_MAX_DIST, 0.0, 1.0)

        if LOG_LIDAR:
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
            # Remove the previous marker if it exists
            if self._active_marker_id is not None:
                try:
                    debug.remove_sphere(self._active_marker_id)
                except Exception:
                    pass

            target = self.waypoints[idx % len(self.waypoints)]
            marker_id = f"active_wp_{idx}"
            # Bright green sphere, 3 m radius, slightly above ground
            pos = (target[0], target[1], target[2] + 2.0)
            debug.draw_sphere(
                pos=pos,
                radius=3.0,
                rgba=(0.0, 1.0, 0.2, 0.8),
                cling=False,
                freeze=False,
            )
            self._active_marker_id = marker_id
        except AttributeError:
            # bng.debug not available on this beamngpy version — skip silently
            pass

    def _draw_start_end_markers(self):
        """Draw a blue sphere at the spawn position and a red sphere at the
        last waypoint.

        Best-effort: skipped silently if the beamngpy version doesn't expose
        bng.debug.draw_sphere.
        """
        if self.bng is None or self.trajectory is None or not self.waypoints:
            return
        try:
            debug = self.bng.debug
            start = self.trajectory.spawn_pos
            debug.draw_sphere(
                pos=(start[0], start[1], start[2] + 1.0),
                radius=2.5,
                rgba=(0.0, 0.5, 1.0, 0.8),  # blue = start
                cling=False,
                freeze=False,
            )
            end = self.waypoints[-1]
            debug.draw_sphere(
                pos=(end[0], end[1], end[2] + 2.0),
                radius=2.5,
                rgba=(1.0, 0.0, 0.0, 0.8),  # red = end
                cling=False,
                freeze=False,
            )
        except AttributeError:
            pass


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
        scales = [(5.0, 5.0, 1.0)] * len(self.waypoints)
        self.scenario.add_checkpoints(self.waypoints, scales)

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
            is_visualised=False,
            is_static=False,
        )

        self._update_active_marker(1)
        self._draw_start_end_markers()

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
            ]
        )

    def close(self):
        if self.camera is not None:
            t = threading.Thread(target=self.camera.remove, daemon=True)
            t.start()
            t.join(timeout=3.0)
            self.camera = None
        super().close()

    # ------------------------------------------------------------------
    # Camera processing
    # ------------------------------------------------------------------

    def _process_camera(self) -> np.ndarray:
        """Poll camera and return a flattened grayscale image normalized to [0, 1]."""
        n_pixels = self.CAM_OUT_SIZE[0] * self.CAM_OUT_SIZE[1]
        if self.camera is None:
            return np.ones(n_pixels, dtype=np.float32)

        data = self.camera.poll()
        colour = data.get("colour", None)
        if colour is None:
            return np.ones(n_pixels, dtype=np.float32)

        # beamngpy may return a PIL Image or a numpy array depending on version
        img = np.asarray(colour, dtype=np.float32)

        # Convert RGB(A) to grayscale using luminosity weights
        if img.ndim == 3 and img.shape[2] >= 3:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img.squeeze().astype(np.float32)

        # Resize to CAM_OUT_SIZE
        oh, ow = self.CAM_OUT_SIZE
        try:
            from PIL import Image as PILImage

            pil = PILImage.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
            pil = pil.resize((ow, oh), PILImage.BILINEAR)
            small = np.array(pil, dtype=np.float32)
        except ImportError:
            h, w = gray.shape
            sh, sw = max(1, h // oh), max(1, w // ow)
            small = gray[::sh, ::sw][:oh, :ow]
            if small.shape != (oh, ow):
                padded = np.zeros((oh, ow), dtype=np.float32)
                padded[: small.shape[0], : small.shape[1]] = small
                small = padded

        self.last_frame = small / 255.0  # 2-D (CAM_OUT_SIZE), values in [0, 1]
        if LOG_CAMERA:
            flat = self.last_frame.flatten()
            mn, mx, avg = float(flat.min()), float(flat.max()), float(flat.mean())
            self.bng.queue_lua_command(
                f"log('I', 'RL', 'Camera: min={mn:.3f} max={mx:.3f} mean={avg:.3f}')"
            )
        return self.last_frame.flatten()
