import logging
import random

import numpy as np
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics

log = logging.getLogger("beamng_env")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[BeamNG] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


class BeamNGDrivingEnv:
    """
    Gymnasium-style environment wrapping BeamNG.drive via beamngpy.

    State  (7 floats, all normalized to ~[-1, 1]):
        speed          - wheel speed normalized to 50 m/s
        steering       - current steering angle (-1 to 1)
        heading_error  - angle between vehicle heading and next waypoint direction
        lateral_error  - perpendicular distance from path (normalized to 20 m)
        damage         - cumulative vehicle damage (normalized)
        dist           - distance to next waypoint (normalized to 150 m)
        alignment      - cos(heading_error), directional alignment signal

    Actions (discrete, 7):
        0 - idle / coast
        1 - full throttle straight
        2 - throttle + slight left
        3 - throttle + slight right
        4 - brake
        5 - throttle + sharp left
        6 - throttle + sharp right

    Also accepts continuous actions as [accel, steering] in [-1, 1].
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
    N_STATES = 7

    # Base waypoints (from main branch — updated positions).
    BASE_WAYPOINTS = [
        (70.0, -736.0, 100.0),
        (116.0, -730.0, 100.0),
        (116.0, -612.0, 100.0),
    ]

    SPAWN_POS = (61.0, -788.0, 101.0)
    SPAWN_ROT = (0.0, 0.0, 1.0, 0.0)
    WAYPOINT_RADIUS = 8.0  # metres — how close before advancing to next waypoint
    MAX_STEPS = 1000
    MAX_DAMAGE = 1000.0  # damage threshold that ends the episode
    DAMAGE_SPIKE = 150.0  # single-step damage spike that ends the episode

    # Map boundary — rectangle around the waypoint area + margin.
    # Car gets penalised and episode ends if it leaves this zone.
    BOUNDARY_MIN = (20.0, -830.0)   # (x_min, y_min)
    BOUNDARY_MAX = (160.0, -570.0)  # (x_max, y_max)

    def __init__(
        self,
        beamng_home: str,
        beamng_user: str = None,
        host: str = "localhost",
        port: int = 64256,
        headless: bool = True,
    ):
        """
        Args:
            beamng_home: Path to BeamNG.drive installation directory.
            beamng_user: Optional path to BeamNG user folder (where mods/configs live).
            host: BeamNG server host (default localhost).
            port: BeamNG server port (default 64256).
            headless: If True, launch without rendering (faster training).
        """
        self.beamng_home = beamng_home
        self.beamng_user = beamng_user
        self.host = host
        self.port = port
        self.headless = headless

        self.bng: BeamNGpy = None
        self.vehicle: Vehicle = None
        self.scenario: Scenario = None
        self.electrics: Electrics = None
        self.damage_sensor: Damage = None

        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._steps = 0
        self._is_moving_backward = False
        self._forward_speed = 0.0
        self._active_marker_id: str | None = None
        self.waypoints = list(self.BASE_WAYPOINTS)

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
        self._is_moving_backward = False
        self._checkpoint_hit = False

        # Teleport back to spawn and kill all velocity
        self.vehicle.teleport(pos=self.SPAWN_POS, rot_quat=self.SPAWN_ROT)
        self.vehicle.queue_lua_command("obj:setVelocity(Point3F(0, 0, 0))")
        self.vehicle.queue_lua_command("obj:setAngularVelocity(Point3F(0, 0, 0))")
        self.vehicle.control(throttle=0.0, steering=0.0, brake=1.0)
        self.bng.step(20)

        obs = self._observe()
        # Log car direction vs waypoint direction to diagnose rotation
        state = self.vehicle.state or {}
        vdir = state.get("dir", (0, 0, 0))
        wp = self.waypoints[0]
        dx = wp[0] - self._pos[0]
        dy = wp[1] - self._pos[1]
        log.info(
            "RESET  pos=(%.1f, %.1f)  car_dir=(%.2f, %.2f)  wp_dir=(%.1f, %.1f)  fwd_speed=%.2f",
            self._pos[0], self._pos[1],
            vdir[0], vdir[1],
            dx, dy,
            self._forward_speed,
        )
        return obs

    def step(self, action):
        """
        Apply an action and advance the simulation.

        Accepts either an integer (discrete action index) or a numpy array
        of shape (2,) for continuous [accel, steering] in [-1, 1].

        Returns:
            obs (np.ndarray), reward (float), done (bool), info (dict)
        """
        if isinstance(action, int | np.integer):
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

        # Hard block: if car is moving backward, override everything with full brake
        if self._is_moving_backward:
            throttle = 0.0
            brake = 1.0
            steering = 0.0

        self.vehicle.control(
            throttle=throttle,
            steering=steering,
            brake=brake,
        )

        # 20 physics steps ≈ 200 ms sim time — fewer round-trips for faster training
        self.bng.step(20)
        self._steps += 1

        obs = self._observe()
        reward, done = self._compute_reward(obs)

        # Log every 10 steps to track what the car is doing
        if self._steps % 10 == 1:
            log.info(
                "ep_step=%3d  fwd=%.2f  spd=%.2f  act=%s  rew=%.1f  pos=(%.0f,%.0f)  wp=%d%s",
                self._steps,
                self._forward_speed,
                obs[0] * 50.0,
                action,
                reward,
                self._pos[0],
                self._pos[1],
                self._waypoint_idx,
                "  !! BACKWARD" if self._is_moving_backward else "",
            )

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
            quit_on_close=True,
        )
        self.bng.open(launch=True, headless=self.headless)
        self._load_scenario()

    def _randomize_waypoints(self):
        self.waypoints = [
            (x + random.randint(-10, 10), y + random.randint(-10, 10), z)
            for x, y, z in self.BASE_WAYPOINTS
        ]

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
        self.bng.settings.set_deterministic(steps_per_second=30, speed_factor=4)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        # Let physics settle instead of wall-clock sleep
        self.bng.step(60)

        # Draw the initial active-waypoint marker
        self._update_active_marker(1)

    def _respawn(self):
        """Teleport the vehicle back to the start without reloading the scenario."""
        self.vehicle.teleport(pos=self.SPAWN_POS, rot_quat=self.SPAWN_ROT)
        self.bng.step(10)

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

        # Detect backward movement: dot(facing_direction, velocity).
        # Negative = moving opposite to where the car is facing.
        vdir = state.get("dir", (1.0, 0.0, 0.0))
        vel = state.get("vel", (0.0, 0.0, 0.0))
        self._forward_speed = vdir[0] * vel[0] + vdir[1] * vel[1]
        self._is_moving_backward = self._forward_speed < -0.1

        self._pos = pos  # store for boundary check in _compute_reward
        heading_err, lateral_err, dist = self._path_errors(pos, state)

        obs = np.array(
            [
                np.clip(speed / 50.0, -1.0, 1.0),
                np.clip(steering, -1.0, 1.0),
                np.clip(heading_err / np.pi, -1.0, 1.0),
                np.clip(lateral_err / 20.0, -1.0, 1.0),
                np.clip(damage / 1000.0, 0.0, 1.0),
                np.clip(dist / 150.0, 0.0, 1.0),  # distance to waypoint
                np.cos(heading_err),  # alignment signal
            ],
            dtype=np.float32,
        )

        return obs

    def _path_errors(self, pos, state):
        """Return (heading_error_rad, lateral_error_m, distance_m) relative to next waypoint."""
        if not self.waypoints or not state:
            return 0.0, 0.0, 0.0

        target = self.waypoints[self._waypoint_idx % len(self.waypoints)]
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        dist = float(np.hypot(dx, dy))

        # Advance waypoint when close enough
        if dist < self.WAYPOINT_RADIUS:
            self._waypoint_idx += 1
            self._checkpoint_hit = True
            self._update_active_marker(self._waypoint_idx)
            print("checkpoint hit")

        vel = state.get("vel", (1.0, 0.0, 0.0))
        vehicle_heading = np.arctan2(vel[1], vel[0])
        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi

        lateral_err = dist * np.sin(heading_err)
        return float(heading_err), float(lateral_err), dist

    def _compute_reward(self, obs):
        """Compute per-step reward and termination flag."""
        speed, steering, heading_err, lateral_err, damage_norm, dist_norm, alignment = obs
        damage = damage_norm * 1000.0

        done = False
        reward = 0.0

        # 1. Speed toward waypoint: alignment [-1,1] * speed → strong directional signal
        reward += speed * alignment * 5.0

        # 2. Penalise bad heading (facing away from waypoint)
        reward -= (1.0 - alignment) * 1.0

        # 3. Penalise being far from waypoint
        reward -= dist_norm * 0.5

        # 4. Penalise being stationary — the agent must move
        if speed < 0.05:
            reward -= 2.0

        # 5. Penalise excessive steering
        reward -= abs(steering) * 0.2

        # 6. Penalise moving backward
        if self._is_moving_backward:
            reward -= 5.0

        # 7. Time penalty — encourages the agent to reach waypoints quickly
        reward -= 0.1

        # 8. Damage: penalise proportionally, only terminate on big spikes
        damage_delta = damage - self._last_damage
        if damage_delta > 0:
            reward -= damage_delta * 0.3  # proportional penalty
        if damage_delta > self.DAMAGE_SPIKE:
            reward -= 30.0
            done = True
        if damage >= self.MAX_DAMAGE:
            done = True
        self._last_damage = damage

        # 9. Boundary check — end episode if car leaves the allowed zone
        px, py = self._pos[0], self._pos[1]
        if (
            px < self.BOUNDARY_MIN[0]
            or px > self.BOUNDARY_MAX[0]
            or py < self.BOUNDARY_MIN[1]
            or py > self.BOUNDARY_MAX[1]
        ):
            reward -= 50.0
            done = True

        # 10. Step limit
        if self._steps >= self.MAX_STEPS:
            done = True

        # 11. Checkpoint bonus (big reward for reaching waypoints)
        if self._checkpoint_hit:
            reward += 100.0 * self._waypoint_idx
            self._checkpoint_hit = False

        # 12. Lap completion bonus
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
