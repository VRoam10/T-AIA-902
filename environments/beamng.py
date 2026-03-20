import random
import time

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics, Lidar


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
        {"throttle": 1.0, "steering": -0.3, "brake": 0.0},  # 2: slight left
        {"throttle": 1.0, "steering": 0.3, "brake": 0.0},  # 3: slight right
        {"throttle": 0.0, "steering": 0.0, "brake": 1.0},  # 4: brake
        {"throttle": 0.5, "steering": -0.6, "brake": 0.0},  # 5: sharp left
        {"throttle": 0.5, "steering": 0.6, "brake": 0.0},  # 6: sharp right
    ]

    N_ACTIONS = len(ACTIONS)

    # LiDAR configuration
    LIDAR_RAYS = 8         # number of angular bins
    LIDAR_FOV_DEG = 120.0  # total forward-facing field of view in degrees
    LIDAR_MAX_DIST = 50.0  # metres — normalization range

    N_STATES = 5 + LIDAR_RAYS  # 5 kinematic + 8 lidar = 13

    # Base waypoints tracing a loop around the automation test track start/finish straight.
    # Random offsets are applied on each scenario load via _randomize_waypoints().
    BASE_WAYPOINTS = [
        (61.0, -744.0, 100.0),
        (102.0, -734.0, 100.0),
        (116.0, -612.0, 100.0),
    ]

    BOUNDARY = {
        "minx": -100.0,
        "maxx": 200.0,
        "miny": -800.0,
        "maxy": -600.0,
    }

    SPAWN_POS = (61.0, -788.0, 101.0)
    SPAWN_ROT = (0.0, 0.0, 1.0, 0.0)
    WAYPOINT_RADIUS = 8.0  # metres — how close before advancing to next waypoint
    MAX_STEPS = 400
    MAX_DAMAGE = 500.0  # damage threshold that ends the episode

    def __init__(
        self,
        beamng_home: str,
        beamng_user: str = None,
        host: str = "localhost",
        port: int = 64256,
    ):
        """
        Args:
            beamng_home: Path to BeamNG.drive installation directory.
                         e.g. r'C:\\Program Files (x86)\\Steam\\steamapps\\common\\BeamNG.drive'
            beamng_user: Optional path to BeamNG user folder (where mods/configs live).
            host: BeamNG server host (default localhost).
            port: BeamNG server port (default 64256).
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

        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._steps = 0
        self._active_marker_id: str | None = None
        self.waypoints = list(self.BASE_WAYPOINTS)
        self._out_of_bounds = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the episode and return the initial observation."""
        if self.bng is None:
            self._launch()
        else:
            self._load_scenario()

        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._steps = 0
        self._checkpoint_hit = False

        # Hold still for a moment so physics settle
        self.vehicle.control(throttle=0.0, steering=0.0, brake=1.0)
        self.bng.step(20)

        return self._observe()

    def step(self, action_idx: int):
        """
        Apply a discrete action and advance the simulation.

        Returns:
            obs (np.ndarray), reward (float), done (bool), info (dict)
        """
        action = self.ACTIONS[action_idx]
        self.vehicle.control(
            throttle=action["throttle"],
            steering=action["steering"],
            brake=action["brake"],
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
            self._launch()
        else:
            self._load_scenario()

        self._waypoint_idx = 0
        self._update_active_marker(1)

        # Release the sim from step-mode so it runs freely in real-time.
        self.bng.resume()
        print("[BeamNGDrivingEnv] Human control active — drive in-game.")

    def close(self):
        """Shut down the BeamNG connection."""
        if self.bng is not None:
            if self.lidar is not None:
                self.lidar.remove()
                self.lidar = None
            self.bng.close()
            self.bng = None
            self.vehicle = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _launch(self):
        """Start BeamNG.drive and load the scenario for the first time."""
        self.bng = BeamNGpy(
            self.host,
            self.port,
            home=self.beamng_home,
            user=self.beamng_user,
        )
        self.bng.open(launch=True)
        self._load_scenario()

    def _randomize_waypoints(self):
        self.waypoints = random.sample(self.BASE_WAYPOINTS, len(self.BASE_WAYPOINTS))

    def _load_scenario(self):
        # self._randomize_waypoints()
        self.scenario = Scenario(
            "gridmap_v2",
            "rl_driving",
            description="RL Training Scenario",
        )

        self.vehicle = Vehicle(
            "ego_vehicle",
            model="burnside",
            licence="Taxi",
            color="Yellow",
            part_config="vehicles/burnside/4door_early_v8_3M_taxi.pc",
        )
        self.electrics = Electrics()
        self.damage_sensor = Damage()
        self.vehicle.attach_sensor("electrics", self.electrics)
        self.vehicle.attach_sensor("damage", self.damage_sensor)

        self.scenario.add_vehicle(
            self.vehicle,
            pos=self.SPAWN_POS,
            rot_quat=self.SPAWN_ROT,
        )

        # Add visual checkpoint rings for every waypoint (visible in-game as hoops).
        scales = [(5.0, 5.0, 1.0)] * len(self.waypoints)
        # Current API: add_checkpoints(positions, scales)
        self.scenario.add_checkpoints(self.waypoints, scales)

        self.scenario.make(self.bng)
        self.bng.set_deterministic(30)  # ensure repeatable physics for same scenario
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(3.0)  # let the game settle before polling

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
            vertical_angle=26.9,
            horizontal_angle=self.LIDAR_FOV_DEG,
            max_distance=self.LIDAR_MAX_DIST,
            is_360_mode=False,
            is_rotate_mode=False,
            is_using_shared_memory=False,
            is_visualised=False,
        )

        # Draw the initial active-waypoint marker
        self._update_active_marker(1)

    def _observe(self) -> np.ndarray:
        """Poll sensors and return a normalized state vector."""
        self.vehicle.poll_sensors()

        elec = self.electrics.data or {}
        dmg = self.damage_sensor.data or {}

        speed = float(elec.get("wheelspeed", 0.0))
        steering = float(elec.get("steering", 0.0))
        damage = float(dmg.get("damage", 0.0))

        state = self.vehicle.state or {}
        pos = state.get("pos", (0.0, 0.0, 0.0))
        vel = state.get("vel", (1.0, 0.0, 0.0))
        vehicle_heading = float(np.arctan2(vel[1], vel[0]))

        heading_err, lateral_err = self._path_errors(pos, state)

        lidar_bins = self._process_lidar(
            self.lidar.poll().get("pointCloud", None) if self.lidar is not None else None,
            pos,
            vehicle_heading,
        )

        self._is_out_of_bounds(pos)

        obs = np.concatenate(
            [
                np.array(
                    [
                        np.clip(speed / 50.0, -1.0, 1.0),
                        np.clip(steering, -1.0, 1.0),
                        np.clip(heading_err / np.pi, -1.0, 1.0),
                        np.clip(lateral_err / 5.0, -1.0, 1.0),
                        np.clip(damage / 1000.0, 0.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
                lidar_bins,
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
            self.bng.queue_lua_command("log('I', 'RL', 'Lidar: all points outside FOV')")
            return distances

        bin_edges = np.linspace(-half_fov, half_fov, self.LIDAR_RAYS + 1)
        for i in range(self.LIDAR_RAYS):
            in_bin = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
            if in_bin.any():
                nearest = float(dists[in_bin].min())
                distances[i] = np.clip(nearest / self.LIDAR_MAX_DIST, 0.0, 1.0)
        
        self.bng.queue_lua_command("log('I', 'RL', 'Lidar: [{}]')".format(", ".join(f"{v:.3f}" for v in distances)))
        return distances

    def _is_out_of_bounds(self, pos):
        """Check if the vehicle is outside the defined boundary."""
        x, y, _ = pos
        b = self.BOUNDARY
        self._out_of_bounds = not (
            b["minx"] <= x <= b["maxx"] and b["miny"] <= y <= b["maxy"]
        )

    def _path_errors(self, pos, state):
        """Return (heading_error_rad, lateral_error_m) relative to next waypoint."""
        if not self.waypoints or not state:
            return 0.0, 0.0

        target = self.waypoints[self._waypoint_idx % len(self.waypoints)]
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = float(np.hypot(dx, dy))

        # Advance waypoint when close enough
        if dist < self.WAYPOINT_RADIUS:
            self._waypoint_idx += 1
            self._checkpoint_hit = True
            self._update_active_marker(self._waypoint_idx)
            self.bng.queue_lua_command("log('I', 'RL', 'checkpoint hit')")

        vel = state.get("vel", (1.0, 0.0, 0.0))
        vehicle_heading = np.arctan2(vel[1], vel[0])
        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi

        lateral_err = dist * np.sin(heading_err)
        return float(heading_err), float(lateral_err)

    def _compute_reward(self, obs):
        """Compute per-step reward and termination flag."""
        speed, steering, heading_err, lateral_err, damage_norm = obs[:5]
        damage = damage_norm * 1000.0

        done = False
        reward = 0.0

        # Encourage forward speed
        reward += speed * 2.0

        # Penalise drifting off path
        reward -= abs(lateral_err) * 1.5
        reward -= abs(heading_err) * 0.5

        # Penalise wobbling the wheel unnecessarily
        reward -= abs(steering) * 0.1

        # Penalise being stationary
        if speed < 0.05:
            reward -= 0.5

        # Penalise (and terminate on) significant damage
        if damage > self._last_damage + 50:
            reward -= 50.0
        if damage >= self.MAX_DAMAGE:
            done = True
        self._last_damage = damage

        # Step limit
        if self._steps >= self.MAX_STEPS:
            done = True

        # Checkpoint bonus
        if self._checkpoint_hit:
            reward += 50.0 * self._waypoint_idx
            self._checkpoint_hit = False

        # Lap completion bonus
        if self._waypoint_idx >= len(self.waypoints):
            reward += 200.0
            self._waypoint_idx = 0
            done = True

        if self._out_of_bounds:
            reward -= 200.0
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
