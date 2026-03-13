import time

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics


class BeamNGDrivingEnv:
    """
    Gymnasium-style environment wrapping BeamNG.drive via beamngpy.

    State  (5 floats, all normalized to ~[-1, 1]):
        speed          - wheel speed normalized to 50 m/s
        steering       - current steering angle (-1 to 1)
        heading_error  - angle between vehicle heading and next waypoint direction
        lateral_error  - perpendicular distance from path (normalized to 5 m)
        damage         - cumulative vehicle damage (normalized)

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
    N_STATES = 5

    # Waypoints tracing a loop around the automation test track start/finish straight.
    # Adjust these coordinates if you use a different spawn or map.
    WAYPOINTS = [
        # (-391.0, -110.0, 1.0),
        (-360.0, -85.0, 1.0),
        (-320.0, -60.0, 1.0),
        (-270.0, -40.0, 1.0),
        (-220.0, -30.0, 1.0),
        (-170.0, -40.0, 1.0),
        (-120.0, -60.0, 1.0),
        (-70.0, -90.0, 1.0),
        (-20.0, -120.0, 1.0),
        # (-391.0, -110.0, 1.0),  # back to start — triggers lap bonus
    ]

    SPAWN_POS = (-391.0, -110.0, 1.0)
    SPAWN_ROT = (0.0, 0.0, 1.0, 0.0)  # ~45-degree heading
    WAYPOINT_RADIUS = 15.0  # metres — how close before advancing to next waypoint
    MAX_STEPS = 1000
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

        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._steps = 0
        self._active_marker_id: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the episode and return the initial observation."""
        if self.bng is None:
            self._launch()
        else:
            self._respawn()

        self._waypoint_idx = 1
        self._last_damage = 0.0
        self._steps = 0

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

    def close(self):
        """Shut down the BeamNG connection."""
        if self.bng is not None:
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

    def _load_scenario(self):
        self.scenario = Scenario(
            "smallgrid",
            "rl_driving",
            description="RL Training Scenario",
        )

        self.vehicle = Vehicle(
            "ego_vehicle", model="etk800", licence="RL", color="Blue"
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
        # beamngpy API varies by version — try each known signature.
        ids    = [f"wp_{i}" for i in range(len(self.WAYPOINTS))]
        scales = [(4.0, 4.0, 0.5)] * len(self.WAYPOINTS)
        try:
            # Current API: add_checkpoints(positions, scales, ids)
            self.scenario.add_checkpoints(self.WAYPOINTS, scales, ids)
        except TypeError:
            try:
                self.scenario.add_checkpoints(
                    positions=self.WAYPOINTS,
                    sizes=scales,
                    ids=ids,
                )
            except TypeError:
                self.scenario.add_checkpoints(self.WAYPOINTS, scales)

        self.scenario.make(self.bng)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(1.0)  # let the game settle before polling

        # Draw the initial active-waypoint marker
        self._update_active_marker(1)

    def _respawn(self):
        """Teleport the vehicle back to the start without reloading the scenario."""
        self.vehicle.teleport(pos=self.SPAWN_POS, rot_quat=self.SPAWN_ROT)
        time.sleep(0.3)

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

        heading_err, lateral_err = self._path_errors(pos, state)

        obs = np.array(
            [
                np.clip(speed / 50.0, -1.0, 1.0),
                np.clip(steering, -1.0, 1.0),
                np.clip(heading_err / np.pi, -1.0, 1.0),
                np.clip(lateral_err / 5.0, -1.0, 1.0),
                np.clip(damage / 1000.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        return obs

    def _path_errors(self, pos, state):
        """Return (heading_error_rad, lateral_error_m) relative to next waypoint."""
        if not self.WAYPOINTS or not state:
            return 0.0, 0.0

        target = self.WAYPOINTS[self._waypoint_idx % len(self.WAYPOINTS)]
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = float(np.hypot(dx, dy))

        # Advance waypoint when close enough
        if dist < self.WAYPOINT_RADIUS:
            self._waypoint_idx += 1
            self._update_active_marker(self._waypoint_idx)

        vel = state.get("vel", (1.0, 0.0, 0.0))
        vehicle_heading = np.arctan2(vel[1], vel[0])
        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi

        lateral_err = dist * np.sin(heading_err)
        return float(heading_err), float(lateral_err)

    def _compute_reward(self, obs):
        """Compute per-step reward and termination flag."""
        speed, steering, heading_err, lateral_err, damage_norm = obs
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
            done = True
        if damage >= self.MAX_DAMAGE:
            done = True
        self._last_damage = damage

        # Step limit
        if self._steps >= self.MAX_STEPS:
            done = True

        # Lap completion bonus
        if self._waypoint_idx >= len(self.WAYPOINTS):
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

            target = self.WAYPOINTS[idx % len(self.WAYPOINTS)]
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
