import random
import socket
import sys
import threading
import time

import numpy as np

try:
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors import Camera, Damage, Electrics, GForces, Lidar, RoadsSensor
except ImportError:
    BeamNGpy = Scenario = Vehicle = Camera = Damage = Electrics = GForces = Lidar = RoadsSensor = None

from config import LOG_CAMERA, LOG_CHECKPOINT_HIT, LOG_LIDAR
from core.stop_signal import stop_requested
from core.trajectory import TrajectoryData, load_or_generate
from environments import beamng_sensors, beamng_spec
from environments.beamng_features import road_info_features, wheel_info_features
from environments.beamng_geometry import body_orientation_features
from environments.beamng_path import NEUTRAL as NEUTRAL_PATH_POS
from environments.beamng_path import PathPosition, project_onto_path
from environments.beamng_reward import compute_race_reward
from environments.beamng_spawn import corrected_spawn, measure_spawn_z_correction


class BeamNGDrivingEnv:
    """
    Gymnasium-style environment wrapping BeamNG.drive via beamngpy.

    One env, two independent axes (see :mod:`environments.beamng_spec`):

      * ``sensor`` — ``"lidar"`` (8 range bins), ``"adv_lidar"`` (4x8 elevation x
        azimuth grid, so the policy can tell a wall from a low object) or
        ``"camera"`` (16x16 grayscale dashcam).
      * ``output`` — ``"fixed"`` (the 7-entry ``ACTIONS`` table) or
        ``"continuous"`` (throttle, steering, brake driven directly).

    Observation (all normalized to ~[-1, 1] or [0, 1]):

        speed          - wheel speed normalized to 50 m/s
        steering        - current steering angle (-1 to 1)
        heading_error  - angle between vehicle heading and next waypoint direction
        lateral_error  - perpendicular distance from path (normalized to 5 m)
        damage         - cumulative vehicle damage (normalized)
        dist           - distance to the target checkpoint
        perception[..] - the sensor's feature block
        hints[..]      - optional vehicle-local coords of upcoming waypoints
        extra[..]      - optional [pitch, roll] and/or [edgeL, edgeR]

    ``n_states`` is always ``beamng_spec.obs_size(...)`` for the active config.
    """

    # Throttle drops sharply as steering rises. The race car is a mid-engine RWD
    # with far more power than grip, so the old taxi-era table (0.4 throttle at
    # 0.6 steering) spun it on most corner entries. Seven entries either way, so
    # the discrete head size is unchanged.
    ACTIONS = [
        {"throttle": 0.0, "steering": 0.0, "brake": 0.0},  # 0: coast
        {"throttle": 1.0, "steering": 0.0, "brake": 0.0},  # 1: full throttle straight
        {"throttle": 0.6, "steering": -0.25, "brake": 0.0},  # 2: power-on slight left
        {"throttle": 0.6, "steering": 0.25, "brake": 0.0},  # 3: power-on slight right
        {"throttle": 0.0, "steering": 0.0, "brake": 1.0},  # 4: brake
        {"throttle": 0.15, "steering": -0.55, "brake": 0.0},  # 5: lift + sharp left
        {"throttle": 0.15, "steering": 0.55, "brake": 0.0},  # 6: lift + sharp right
    ]

    # LiDAR geometry / mount, re-exported from beamng_sensors so subclasses, the
    # multi env and the tests keep one source of truth.
    LIDAR_RAYS = beamng_spec.LIDAR_RAYS
    LIDAR_CHANNELS_PER_RAY = beamng_spec.LIDAR_CHANNELS_PER_RAY
    LIDAR_FOV_DEG = beamng_sensors.LIDAR_FOV_DEG
    LIDAR_MAX_DIST = beamng_sensors.LIDAR_MAX_DIST
    LIDAR_GROUND_CLEARANCE = beamng_sensors.LIDAR_GROUND_CLEARANCE
    LIDAR_SELF_MARGIN = beamng_sensors.LIDAR_SELF_MARGIN
    LIDAR_ROOF_CLEARANCE = beamng_sensors.LIDAR_ROOF_CLEARANCE
    LIDAR_MOUNT_POS = beamng_sensors.LIDAR_MOUNT_POS
    LIDAR_MOUNT_DIR = beamng_sensors.LIDAR_MOUNT_DIR
    LIDAR_MOUNT_UP = beamng_sensors.LIDAR_MOUNT_UP
    BBOX_MAX_HALF_EXTENT = beamng_sensors.BBOX_MAX_HALF_EXTENT

    CAM_OUT_SIZE = beamng_spec.CAM_OUT_SIZE

    # Scale for the observation's `dist` feature only: distance to the target
    # checkpoint is divided by this and clipped to [0, 2]. It no longer gates
    # anything — the off-track penalty and reset that used it were removed because
    # the game's own tracks space checkpoints far wider than any fixed threshold
    # (see environments.beamng_reward).
    CHECKPOINT_DIST_NORM_M = 200.0

    WAYPOINT_RADIUS = 8.0  # metres — how close before advancing to next waypoint
    MAX_STEPS = 5000

    # close() shutdown timing: BeamNGpy close runs in a daemon thread bounded by
    # CLOSE_JOIN_TIMEOUT so a frozen sim cannot hang the pipeline; with kill_sim
    # we then poll the sim port every KILL_WAIT_POLL seconds, up to
    # KILL_WAIT_TIMEOUT, until it stops accepting connections.
    CLOSE_JOIN_TIMEOUT = 5.0
    KILL_WAIT_TIMEOUT = 60.0
    KILL_WAIT_POLL = 0.5
    MAX_DAMAGE = 1000.0  # damage threshold that ends the episode
    HUMAN_RESPAWN_DAMAGE = 100.0  # human play: damage above this counts as a crash
    HALF_TRACK_WIDTH = 0.7  # metres — half vehicle track, for per-wheel road-edge projection

    AVAILABLE_MAPS = list(beamng_spec.AVAILABLE_MAPS)

    # One car for everything, so a head-to-head result reflects the policies and not
    # the machinery. A dict of beamngpy Vehicle kwargs (the shape the multi/race envs
    # already build from), so only `color` varies per entrant.
    RACE_CAR = beamng_spec.RACE_CAR

    def __init__(
        self,
        beamng_home: str,
        beamng_user: str = None,
        host: str = "localhost",
        port: int = 25252,
        headless: bool = False,
        sensor: str = beamng_spec.DEFAULT_SENSOR,
        output: str = "fixed",
        map_name: str = "gridmap_v2",
        trajectory_hints: int = 0,
        body_orientation: bool = False,
        road_info: bool = False,
        wheel_info: bool = False,
        random_path: bool = False,
        dense_episodes: int = 0,
        track: str | None = None,
    ):
        """
        Args:
            beamng_home: Path to BeamNG.drive installation directory.
            beamng_user: Optional path to BeamNG user folder (where mods/configs live).
            host: BeamNG server host (default localhost).
            port: BeamNG server port (default 25252).
            sensor: Perception axis — "lidar", "adv_lidar" or "camera".
            output: Action axis — "fixed" (discrete table) or "continuous".
            dense_episodes: Curriculum warm-up — use dense waypoints (8 m apart,
                            trivially hittable) for this many initial episodes,
                            then switch to sparse (25 m). 0 = sparse from the
                            start (default).
            track: One of the game's own race tracks (a quickrace key such as
                   "mixedCircuit1"), driven instead of the generated road-network
                   paths. None (default) keeps the generated trajectory cache.
                   See :mod:`core.quickrace`.
        """
        if sensor not in beamng_spec.SENSORS:
            raise ValueError(f"unknown sensor {sensor!r}; expected one of {beamng_spec.SENSORS}")
        if output not in beamng_spec.OUTPUTS:
            raise ValueError(f"unknown output {output!r}; expected one of {beamng_spec.OUTPUTS}")

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
        self.camera: Camera = None
        self.roads_sensor: RoadsSensor = None
        self.gforces: GForces = None

        self.sensor = sensor
        self.output = output
        self.map_name = map_name

        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._last_dist = 0.0
        self._steps = 0
        self._invuln_steps = 0  # damage-immune steps remaining (granted on checkpoint hit)
        self._steps_since_checkpoint = 0  # drives the segment-time bonus
        self._checkpoint_hit = False
        # Whether the RoadsSensor may be polled: False from a teleport or a scenario
        # load until the simulator has advanced at least one physics step. Polling in
        # between hangs the sensor's game-engine side on road-dense maps and blocks
        # Python in the socket recv (docs/romain.md, seventh issue).
        self._road_pollable = False
        self._active_marker_id: str | None = None
        self._log_obs = False  # human play enables obs logging; training leaves it off
        self._obs_log_stdout = False  # human play also echoes the obs to the terminal
        self._last_obs_lines: list[str] = []  # most recent formatted obs lines (camera redraw)
        self.headless = headless
        self.trajectory_hints = trajectory_hints
        self.body_orientation = body_orientation
        self.road_info = road_info
        self.wheel_info = wheel_info
        self.random_path = random_path
        self.dense_episodes = dense_episodes
        self.track = track
        self._episode_idx = 0  # incremented on each reset(); drives the waypoint curriculum

        self.n_states = beamng_spec.obs_size(
            sensor, trajectory_hints, body_orientation, road_info, wheel_info
        )
        self.n_actions = beamng_spec.action_size(output)

        # Filled on first _launch() — either read from cache or generated then.
        self.trajectory: TrajectoryData | None = None
        self._paths: list[TrajectoryData] = []
        self.waypoints: list[tuple[float, float, float]] = []
        self._current_pos = (0.0, 0.0, 0.0)

        # The polyline the observation and the reward measure against: the spawn
        # followed by every checkpoint. The spawn is what makes the first segment
        # count — `waypoints` starts after it, so projecting onto that alone would
        # report zero progress until checkpoint 0 was behind the car.
        self._guide_line: list[tuple[float, float, float]] = []
        self._path_pos: PathPosition = NEUTRAL_PATH_POS
        self._last_progress_m = 0.0
        # cos(velocity heading - path tangent), computed once per observation
        # alongside `_path_pos` so the reward reads the same tangent the
        # observation was built from. None before any observation, or without a
        # guide line (see tests/test_beamng_reward.py) — either way a reward
        # computed from it falls back to the checkpoint bearing instead of
        # reading a stale or meaningless value.
        self._path_alignment: float | None = None

        # Measured once per scenario load: how far the cached spawn heights sit
        # above where the car actually rests. Teleports add it so a reset places
        # the car on the road instead of dropping it (see environments.beamng_spawn).
        self._spawn_z_correction = 0.0

        # Cached ego OBB extents in vehicle-local frame, populated once per scenario
        # load; used to reject LiDAR self-hits and to place the roof mount.
        self._ego_local_extents: tuple[float, float, float, float, float, float] | None = None

        # Last-poll LiDAR filtering breakdown (counts + nearest kept point), for debug.
        self._lidar_debug: dict = {}
        # Most recent 2-D camera frame, for the human-play ASCII render.
        self.last_frame: np.ndarray | None = None

    @property
    def n_perception(self) -> int:
        """Length of the perception block for this env's sensor."""
        return beamng_spec.perception_features(self.sensor)

    def _spawn_target(self) -> tuple[float, float, float]:
        """Where to teleport for the current path — the cached spawn, height-corrected.

        The scenario spawn clings to the surface, so it needs no correction; every
        teleport does, because ``Vehicle.teleport`` places the reference point at
        exactly the z it is handed.
        """
        assert self.trajectory is not None
        return corrected_spawn(self.trajectory.spawn_pos, self._spawn_z_correction)

    def _select_waypoints(self) -> list[tuple[float, float, float]]:
        assert self.trajectory is not None
        if self.dense_episodes > 0 and self._episode_idx <= self.dense_episodes:
            return list(self.trajectory.dense_waypoints)
        return list(self.trajectory.sparse_waypoints)

    def _rebuild_guide_line(self) -> None:
        """Refresh the guide polyline from the current trajectory + waypoints."""
        if self.trajectory is None:
            self._guide_line = []
            return
        self._guide_line = [tuple(self.trajectory.spawn_pos), *self.waypoints]

    def _project(self, pos) -> PathPosition:
        """Where ``pos`` sits on the guide polyline."""
        return project_onto_path(self._guide_line, pos)

    def progress_m(self) -> float:
        """Metres covered along the path — the projection's arc length alone.

        Not lap-aware: ``_waypoint_idx`` crosses ``len(waypoints)`` at the finish
        of every run, not only on a second lap of a closed circuit, so a term
        keyed off that crossing (``waypoint_idx // len(waypoints)``) is not a lap
        counter — it added a full path length to progress on every finish, even
        at laps=1. A real lap counter needs a lap-crossing *event*, not this
        index. Progress must stay a function of position alone.
        """
        return self._path_pos.progress_m

    def _advance(self, steps: int) -> None:
        """Advance the simulation and mark the road sensor pollable again."""
        self.bng.step(steps)
        self._road_pollable = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset the episode and return the initial observation."""
        self._episode_idx += 1
        if self.bng is None:
            self._launch()
        else:
            # Both branches below reposition the vehicle (teleport or
            # scenario.restart()), so close the gate once here rather than only
            # under random_path — that left the restart() branch relying on
            # lucky call order, exactly what this gate exists to replace.
            self._road_pollable = False
            if self.random_path:
                # Teleport (reset=True) repositions AND resets the vehicle to the
                # newly chosen path's spawn. Do NOT call scenario.restart() here:
                # restart() snaps the car back to the baked path[0] spawn and
                # fights the teleport, so the path never actually changes. This
                # mirrors the multi-agent env's reset_vehicle (teleport, no restart).
                self._pick_episode_path()
                self.vehicle.teleport(
                    self._spawn_target(),
                    rot_quat=self.trajectory.spawn_rot,
                    reset=True,
                )
            else:
                self.bng.scenario.restart()
            # Waypoint density can flip between episodes (dense warm-up ->
            # sparse curriculum), so re-select every reset. Redundant but
            # harmless right after _pick_episode_path on the random-path branch.
            self.waypoints = self._select_waypoints()
            self._rebuild_guide_line()
            self._update_active_marker(0)

        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._last_dist = 0.0
        self._steps = 0
        self._invuln_steps = 0
        self._steps_since_checkpoint = 0
        self._checkpoint_hit = False

        # Hold still for a moment so physics settle
        self.vehicle.control(throttle=0.0, steering=0.0, brake=0.0)
        self._advance(5)

        obs = self._observe()
        # Initialize last_dist after first observe so progress reward starts at 0
        self._last_dist = self._current_dist
        self._last_progress_m = self.progress_m()
        return obs

    def controls_for(self, action) -> tuple[float, float, float]:
        """Map an action to (throttle, steering, brake) for this env's output axis.

        ``fixed`` indexes the ACTIONS table. ``continuous`` accepts either a 2-vector
        ``[accel, steering]`` (positive accel = throttle, negative = brake) or a
        3-vector ``[throttle, steering, brake]``, both clipped to [-1, 1]. An int
        action is always read as a table index, so a discrete agent driving a
        continuous env still works.
        """
        if self.output == "fixed" or isinstance(action, (int, np.integer)):
            ctrl = self.ACTIONS[int(action)]
            return ctrl["throttle"], ctrl["steering"], ctrl["brake"]

        action = np.clip(np.asarray(action, dtype=np.float32).ravel(), -1.0, 1.0)
        if action.shape[0] == 2:
            accel = float(action[0])
            return max(0.0, accel), float(action[1]), max(0.0, -accel)
        return float(max(0.0, action[0])), float(action[1]), float(max(0.0, action[2]))

    def step(self, action):
        """Apply an action and advance the simulation.

        Returns:
            obs (np.ndarray), reward (float), done (bool), info (dict)
        """
        throttle, steering, brake = self.controls_for(action)
        self.vehicle.control(throttle=throttle, steering=steering, brake=brake)

        # In deterministic mode each simulation step lasts
        # 1/PHYSICS_STEPS_PER_SECOND, so this advances SECONDS_PER_ENV_STEP
        # (~333 ms) of sim time — a ~3 Hz control rate, not the ~10 Hz a
        # long-standing comment here claimed. See beamng_spec for the numbers.
        self._advance(beamng_spec.PHYSICS_STEPS_PER_ENV_STEP)
        self._steps += 1

        obs = self._observe()
        reward, done = self._compute_reward(obs)
        info = {"steps": self._steps, "waypoint_idx": self._waypoint_idx}
        return obs, reward, done, info

    def human_play(self):
        """Load the scenario and give control to the human.

        Logs the full labeled observation each tick to BeamNG's Lua console and to
        the terminal, and advances waypoints/markers as the player drives. With the
        camera sensor the dashcam frame is also rendered as in-place ASCII art; with
        a LiDAR sensor the filtering diagnostics (which the observation itself does
        not carry) are printed alongside.
        """
        if self.bng is None:
            self._launch(human_control=True)
        else:
            self._load_scenario(human_control=True)

        self._waypoint_idx = 0
        self._update_active_marker(0)
        self._log_obs = True
        # The camera branch redraws the obs lines itself as part of the ASCII block,
        # so leave the plain stdout echo off there to avoid double printing.
        camera_view = self.sensor == "camera"
        self._obs_log_stdout = not camera_view

        self.bng.resume()
        # Realtime session: the sim now advances continuously on its own (the
        # observe loop's time.sleep(0.1) below is several physics steps), so the
        # road sensor is always safe to poll here. Unlike the lockstep reset/step
        # paths, nothing ever calls _advance() again in this session, so the gate
        # must be opened explicitly rather than waiting for a step that never comes.
        self._road_pollable = True
        print(
            f"[BeamNGDrivingEnv] Human control active ({self.sensor}) — "
            "drive in-game. Press Ctrl+C to stop."
        )
        if beamng_spec.is_lidar(self.sensor) and self.lidar is None:
            print(
                "[BeamNGDrivingEnv] Warning: LiDAR sensor not attached — "
                "bins will show fallback values."
            )

        ramp = " ░▒▓█"
        prev_total = 0
        first = True

        try:
            while not stop_requested():
                # _observe polls the sensors, logs the full labeled obs, and advances
                # waypoints/markers as the player drives.
                self._observe()
                if not self._maybe_reset_on_completion():
                    self._maybe_respawn_on_crash()

                if camera_view:
                    frame = self.last_frame
                    if frame is None:
                        frame = np.zeros(self.CAM_OUT_SIZE, dtype=np.float32)
                    rows = ["".join(ramp[min(int(v * 4), 4)] for v in row) for row in frame]
                    block = rows + self._last_obs_lines
                    if not first:
                        sys.stdout.write(f"\033[{prev_total}A")
                    # \033[K clears each line to its end so a shorter frame leaves no residue.
                    sys.stdout.write("".join(f"{line}\033[K\n" for line in block))
                    sys.stdout.flush()
                    prev_total = len(block)
                    first = False
                elif self._lidar_debug:
                    d = self._lidar_debug
                    print(
                        f"[LiDAR dbg] total={d.get('total', 0)} self={d.get('self', 0)} "
                        f"ground={d.get('ground', 0)} kept={d.get('kept', 0)} "
                        f"fov={d.get('fov', 0)} extents_none={d.get('extents_none')} "
                        f"nearest={d.get('min_dist_m', float('nan')):.1f}m "
                        f"z={d.get('min_dist_z', float('nan')):+.2f} "
                        f"ground_z={d.get('ground_z', float('nan')):+.2f} "
                        f"z_max_seen={d.get('z_max_seen', float('nan')):+.2f}"
                    )

                time.sleep(0.1)
        except KeyboardInterrupt:
            print("[BeamNGDrivingEnv] Human play stopped.")

    def close(self, kill_sim: bool = True):
        """Close this environment.

        With ``kill_sim`` (the default) BeamNGpy `close()` terminates the BeamNG
        process; ``kill_sim=False`` calls `disconnect()`, dropping this client
        while leaving the game running. Sessions pass the default so quitting the
        TUI never leaves an orphaned simulator.
        """
        if self.bng is not None:
            self._remove_lidar()
            self._remove_camera()
            self._remove_roads_sensor()
            close_fn = self.bng.close if kill_sim else self.bng.disconnect
            t = threading.Thread(target=close_fn, daemon=True)
            t.start()
            t.join(timeout=self.CLOSE_JOIN_TIMEOUT)
            self.bng = None
            self.vehicle = None
            if kill_sim:
                self._wait_sim_shutdown()

    def _wait_sim_shutdown(self):
        """Block until the simulator's port stops accepting connections.

        BeamNGpy ``open(launch=True)`` connects to any instance still listening
        on the port before it considers launching a new one, and ``close()``
        above is fire-and-forget (daemon thread, bounded join) while the actual
        shutdown (scenario close, Quit ack, process kill) can take much longer.
        Without this wait, a back-to-back close -> open connects to the dying
        simulator and raises BNGDisconnectedError once the kill lands, instead of
        relaunching the game. Bounded by KILL_WAIT_TIMEOUT so a frozen sim still
        cannot hang the pipeline.
        """
        deadline = time.time() + self.KILL_WAIT_TIMEOUT
        while time.time() < deadline:
            try:
                socket.create_connection((self.host, self.port), timeout=1.0).close()
            except OSError:
                return
            time.sleep(self.KILL_WAIT_POLL)
        print(
            f"[BeamNGDrivingEnv] Warning: simulator still listening on "
            f"{self.host}:{self.port} after {self.KILL_WAIT_TIMEOUT:.0f}s; "
            "the next launch may connect to the dying instance."
        )

    def _remove_lidar(self):
        """Detach the current LiDAR before replacing the ego vehicle/scenario."""
        beamng_sensors.remove_sensor(getattr(self, "lidar", None))
        self.lidar = None

    def _remove_camera(self):
        beamng_sensors.remove_sensor(getattr(self, "camera", None))
        self.camera = None

    def _attach_roads_sensor(self):
        """Attach a RoadsSensor when road_info is on; replace any prior one."""
        if not self.road_info:
            return
        self._remove_roads_sensor()
        self.roads_sensor = beamng_sensors.create_roads_sensor("roads", self.bng, self.vehicle)

    def _remove_roads_sensor(self):
        beamng_sensors.remove_sensor(getattr(self, "roads_sensor", None))
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
        # Honor random_path on the very first scenario load too (e.g. human play,
        # which drives a single launched episode and never calls reset()). No-op
        # when random_path is off, so the default first path is kept.
        self._pick_episode_path()
        self.waypoints = self._select_waypoints()
        self._rebuild_guide_line()
        self._current_pos = self.trajectory.spawn_pos
        self._load_scenario(human_control=human_control)

    def _resolve_trajectory(self) -> TrajectoryData:
        """Pick the paths to drive: a chosen game track, else the generated cache.

        A ``track`` names one of the game's own race tracks, read from the level
        files with no simulator involvement — so this branch needs neither a cache
        nor a probe scenario. Without one, behaviour is unchanged: load the
        generated cache, or probe the road network to build it.
        """
        from core.trajectory import CACHE_DIR

        if self.track:
            self._paths = [self._load_track(self.track)]
            self.trajectory = self._paths[0]
            return self.trajectory

        cache_path = CACHE_DIR / f"{self.map_name}.json"
        if cache_path.exists():
            self._paths = load_or_generate(self.map_name, bng=None).paths
        else:
            # No cache -> run a probe scenario so we can call get_road_network
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

    def _load_track(self, key: str) -> TrajectoryData:
        """One of the game's race tracks as a TrajectoryData, read from disk."""
        from core import quickrace

        race = quickrace.load(self.map_name, key, self.beamng_home)
        print(
            f"[track] {self.map_name}/{race.key}: {race.kind}, "
            f"{len(race.checkpoints)} checkpoints, {race.length_m():.0f} m"
        )
        return quickrace.to_trajectory(race)

    def _pick_episode_path(self) -> None:
        """When random_path is on, choose a random path for the next episode."""
        if not self.random_path or not self._paths:
            return
        self.trajectory = random.choice(self._paths)
        self.waypoints = self._select_waypoints()
        self._rebuild_guide_line()

    def _reset_human_episode(self) -> None:
        """Teleport to a (possibly new random) path's spawn and reset checkpoints.

        Picks a new random path when random_path is on (else keeps the current
        one), teleports there with a fast reset, and rewinds the checkpoint index
        and marker. Shared by the crash and path-completion handlers.
        """
        self._pick_episode_path()
        self.vehicle.teleport(
            self._spawn_target(),
            rot_quat=self.trajectory.spawn_rot,
            reset=True,
        )
        # Realtime path (human play only): the session stays resumed and the sim
        # keeps advancing on its own through this teleport, so the gate stays open
        # — closing it here would leave it dead for the rest of the session, since
        # nothing calls _advance() again to reopen it (docs/romain.md, seventh issue).
        self._road_pollable = True
        self._waypoint_idx = 0
        self._last_damage = 0.0
        self._update_active_marker(0)

    def _maybe_respawn_on_crash(self) -> bool:
        """Human play: on a crash, deal a fresh random path via a fast teleport.

        Each crash picks a new random path/checkpoints and teleports there, so the
        player gets new checkpoints without relaunching the scenario (a slow full
        reload). No-op when the random-path option is off or the vehicle hasn't
        crashed. Returns True if a respawn happened.
        """
        if not self.random_path:
            return False
        dmg = self.damage_sensor.data if self.damage_sensor is not None else None
        damage = float((dmg or {}).get("damage", 0.0))
        if damage < self.HUMAN_RESPAWN_DAMAGE:
            return False
        self._reset_human_episode()
        return True

    def _maybe_reset_on_completion(self) -> bool:
        """Human play: when the player clears the last checkpoint, reset the path.

        Picks a new random path when random_path is on, otherwise restarts the
        same one, so finishing a path loops you straight into the next drive
        instead of leaving you stranded past the final checkpoint. Returns True
        if a reset happened.
        """
        if not self.waypoints or self._waypoint_idx < len(self.waypoints):
            return False
        self._reset_human_episode()
        return True

    def _load_scenario(self, human_control=False):
        # A LiDAR/Camera is bound to the current BeamNG vehicle. Remove them before
        # replacing `self.vehicle` so changing the scenario cannot leave a stale
        # sensor attached to the previous ego.
        self._remove_lidar()
        self._remove_camera()
        self._remove_roads_sensor()
        self._ego_local_extents = None

        self.scenario = Scenario(
            self.map_name,
            "rl_driving",
            description="RL Driving Scenario",
        )

        self.vehicle = Vehicle("ego_vehicle", **self.RACE_CAR)
        self.electrics = Electrics()
        self.damage_sensor = Damage()
        self.vehicle.attach_sensor("electrics", self.electrics)
        self.vehicle.attach_sensor("damage", self.damage_sensor)
        if self.wheel_info:
            # A classic sensor, so it rides the poll_sensors() round-trip the env
            # already makes rather than costing one of its own.
            self.gforces = GForces()
            self.vehicle.attach_sensor("gforces", self.gforces)

        self.scenario.add_vehicle(
            self.vehicle,
            pos=self.trajectory.spawn_pos,
            rot_quat=self.trajectory.spawn_rot,
            cling=True,
        )

        # Visual checkpoint rings for every waypoint (visible in-game, training and
        # human play). With random_path on, ring every path's waypoints since the
        # episode can be dealt any of them.
        checkpoint_wps = (
            [wp for p in self._paths for wp in p.sparse_waypoints]
            if self.random_path
            else self.waypoints
        )
        self.scenario.add_checkpoints(checkpoint_wps, [(5.0, 5.0, 1.0)] * len(checkpoint_wps))

        self.scenario.make(self.bng)
        # Repeatable physics for the same scenario, and a known step duration.
        self.bng.set_deterministic(beamng_spec.PHYSICS_STEPS_PER_SECOND)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()
        time.sleep(1.0)  # let the game settle before polling

        # The car above was spawned with cling=True, so it is now resting at the
        # true surface height for this spawn. Capture how far that is below the
        # cached height, so teleports place the car instead of dropping it.
        self._spawn_z_correction = measure_spawn_z_correction(
            self.bng, self.vehicle, self.trajectory.spawn_pos[2]
        )
        if self._spawn_z_correction != 0.0:
            print(f"[spawn] teleport height corrected by {self._spawn_z_correction:+.2f} m")

        self._create_perception_sensor(human_control=human_control)

        # Draw the initial active-waypoint marker
        self._update_active_marker(0)
        self._attach_roads_sensor()

        # A scenario load places every vehicle fresh, unsettled, at spawn — not
        # safe to poll the road sensor until the sim has advanced (docs/romain.md,
        # seventh issue). reset()/step() reopen this via _advance(); human_play
        # reopens it explicitly right after resume() since it never steps at all.
        self._road_pollable = False

    def _create_perception_sensor(self, human_control: bool = False):
        """Attach this env's perception sensor. Must run after the scenario starts.

        The camera is visualised only during human play (an on-screen render costs
        GPU every frame, which training does not need).
        """
        if self.sensor == "camera":
            self.camera = beamng_sensors.create_camera(
                "dashcam", self.bng, self.vehicle, visualise=human_control
            )
            return
        # The mount is derived from the ego bbox, so cache the box first.
        self._ego_local_extents = beamng_sensors.cache_ego_local_bbox(self.vehicle)
        self.lidar = beamng_sensors.create_lidar(
            "lidar", self.bng, self.vehicle, self.sensor, self._ego_local_extents
        )

    def _perceive(self, pos, vehicle_heading) -> np.ndarray:
        """Poll this env's sensor and return its perception feature block."""
        block, debug, frame = beamng_sensors.perception_block(
            sensor=self.sensor,
            lidar=self.lidar,
            camera=self.camera,
            pos=pos,
            heading=vehicle_heading,
            ego_extents=self._ego_local_extents,
        )
        if frame is not None:
            self.last_frame = frame
            if LOG_CAMERA and self.bng is not None:
                self.bng.queue_lua_command(
                    f"log('I', 'RL', 'Camera: min={block.min():.3f} "
                    f"max={block.max():.3f} mean={block.mean():.3f}')"
                )
        else:
            self._lidar_debug = debug
            if LOG_LIDAR and self.bng is not None:
                self.bng.queue_lua_command(
                    "log('I', 'RL', 'Lidar: [{}]')".format(", ".join(f"{v:.3f}" for v in block))
                )
        return block

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

        heading_err, dist = self._path_errors(pos, state)
        self._path_pos = self._project(pos)
        # Same tangent the observation is built from, and the same *velocity*
        # heading the fallback's heading_err uses (not vehicle_heading above,
        # which is the nose direction) — so this is exactly the signed velocity
        # component along the path, matching "speed projected onto the
        # direction we want to be going". None without a guide line, so a slot
        # that never got one falls back to the checkpoint bearing instead of
        # reading cos(nose heading) against an arbitrary tangent_rad=0.
        if self._guide_line:
            vel_heading = float(np.arctan2(vel[1], vel[0]))
            self._path_alignment = float(np.cos(vel_heading - self._path_pos.tangent_rad))
        else:
            self._path_alignment = None

        perception = self._perceive(pos, vehicle_heading)

        self._current_pos = pos

        waypoint_hints = self._get_waypoint_hints(pos, vehicle_heading)

        kin = np.array(
            [
                np.clip(speed / 50.0, -1.0, 1.0),
                np.clip(steering, -1.0, 1.0),
                np.clip(heading_err / np.pi, -1.0, 1.0),
                np.clip(self._path_pos.cross_track_m / 5.0, -1.0, 1.0),
                np.clip(damage / 1000.0, 0.0, 1.0),
                np.clip(dist / self.CHECKPOINT_DIST_NORM_M, 0.0, 2.0),
            ],
            dtype=np.float32,
        )
        extra = self._extra_features(state, pos, vehicle_heading, elec)
        obs = np.concatenate([kin, perception, waypoint_hints, extra])

        self._log_observation(kin, perception, waypoint_hints, extra)

        return obs

    def _log_observation(self, kin, perception, waypoint_hints, extra) -> None:
        """Emit the full normalized observation, one labeled line per block.

        Only active during human play (``_log_obs``); training never logs the
        observation. When active it logs to BeamNG's Lua console and, if
        ``_obs_log_stdout`` is set, also prints to the terminal. No-op when the
        simulator is absent (e.g. unit tests).
        """
        if self.bng is None or not self._log_obs:
            return
        lines = self._format_observation_lines(kin, perception, waypoint_hints, extra)
        self._last_obs_lines = lines
        for line in lines:
            self.bng.queue_lua_command(f"log('I', 'RL', '{line}')")
        if self._obs_log_stdout:
            print("\n".join(f"[obs] {line}" for line in lines))

    def _format_observation_lines(self, kin, perception, waypoint_hints, extra) -> list[str]:
        """Build the labeled per-block log lines for the current observation.

        Returns the message strings (without the Lua ``log()`` wrapper) so callers
        can route them to the Lua console, stdout, or both. The arrays are the same
        slices concatenated into the observation, so the printed numbers can never
        drift from what the policy actually sees.
        """
        lines = []

        # --- kinematics (fixed 6, layout matches the concatenation in _observe) ---
        kin = np.asarray(kin).ravel()
        lines.append(
            f"obs kin   | speed={kin[0]:+.2f} steer={kin[1]:+.2f} head={kin[2]:+.2f} "
            f"lat={kin[3]:+.2f} dmg={kin[4]:+.2f} cpdist={kin[5]:+.2f}"
        )

        # --- perception block (per-cell for LiDAR, summarized for camera) ---
        lines.append(beamng_sensors.block_summary(self.sensor, perception))

        # --- waypoint hints (paired forward,left per upcoming waypoint) ---
        hints = np.asarray(waypoint_hints).ravel()
        if hints.size:
            pairs = [
                f"wp{j}=({hints[2 * j]:+.2f},{hints[2 * j + 1]:+.2f})"
                for j in range(hints.size // 2)
            ]
            lines.append(f"obs hints | {' '.join(pairs)}")

        # --- extras (labels derived from the enabled flags, in append order) ---
        extra = np.asarray(extra).ravel()
        if extra.size:
            labels = []
            if self.body_orientation:
                labels += ["pitch", "roll"]
            if self.road_info:
                labels += ["edgeL", "edgeR", "rdhead", "curv", "aheadF", "aheadL"]
            if self.wheel_info:
                labels += ["slip", "slipang", "abs", "latg"]
            while len(labels) < extra.size:
                labels.append(f"x{len(labels)}")
            body = " ".join(f"{labels[k]}={extra[k]:+.2f}" for k in range(extra.size))
            lines.append(f"obs extra | {body}")

        return lines

    def _resolve_lidar_mount_pos(self) -> tuple[float, float, float]:
        """Vehicle-local LiDAR mount, centred above the cached ego roof."""
        return beamng_sensors.lidar_mount_pos(self._ego_local_extents)

    def _lidar_creation_kwargs(self) -> dict:
        """BeamNGpy LiDAR kwargs for this env's sensor."""
        return beamng_sensors.lidar_creation_kwargs(self.sensor, self._ego_local_extents)

    def _process_lidar(self, point_cloud, vehicle_pos, vehicle_heading) -> np.ndarray:
        """Bin a raw LiDAR point cloud into this sensor's feature block.

        Thin wrapper over the shared geometry helper that also stores the filtering
        breakdown for the human-play diagnostics.
        """
        from environments.beamng_geometry import process_lidar

        bins, debug = process_lidar(
            point_cloud,
            vehicle_pos,
            vehicle_heading,
            self._ego_local_extents,
            beamng_sensors.lidar_config(self.sensor),
        )
        self._lidar_debug = debug
        return bins

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

    def _road_info_features(self, pos, heading) -> np.ndarray:
        """The six road-relative features (neutral without a sensor or before a step)."""
        payload = None
        if self.roads_sensor is not None and self._road_pollable:
            payload = self.roads_sensor.poll()
        return road_info_features(payload, self.HALF_TRACK_WIDTH, pos, heading)

    def _wheel_info_features(self, elec, state) -> np.ndarray:
        """The four grip features (neutral without a GForces sensor)."""
        forces = self.gforces.data if self.gforces is not None else None
        return wheel_info_features(
            elec, forces, state.get("vel", (0.0, 0.0, 0.0)), state.get("dir", (1.0, 0.0, 0.0))
        )

    def _extra_features(self, state, pos, heading, elec=None) -> np.ndarray:
        """Optional observation tail: body orientation, road position, wheel grip.

        Appended after the waypoint hints. Empty array when all flags are off, so
        a flag-off observation is byte-for-byte what it was.
        """
        blocks = []
        if self.body_orientation:
            blocks.append(self._body_orientation_features(state))
        if self.road_info:
            blocks.append(self._road_info_features(pos, heading))
        if self.wheel_info:
            blocks.append(self._wheel_info_features(elec or {}, state))
        if not blocks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(blocks)

    def _path_errors(self, pos, state):
        """Return (heading_error_rad, dist) for the next waypoint; advance on arrival.

        Cross-track error no longer comes from here: ``dist * sin(heading_err)`` is a
        function of the two values this returns, so it carried no information. The
        observation uses the guide-line projection instead.

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
            if LOG_CHECKPOINT_HIT and self.bng is not None:
                self.bng.queue_lua_command("log('I', 'RL', 'checkpoint hit')")
            if self._waypoint_idx < len(self.waypoints):
                new_t = self.waypoints[self._waypoint_idx]
                self._current_dist = float(np.hypot(new_t[0] - pos[0], new_t[1] - pos[1]))

        vel = state.get("vel", (1.0, 0.0, 0.0))
        vehicle_heading = np.arctan2(vel[1], vel[0])
        target_heading = np.arctan2(dy, dx)
        heading_err = (target_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi

        return float(heading_err), dist

    def _compute_reward(self, obs):
        """Racing reward, shared with the multi/race envs via beamng_reward.

        No rival here, so the gap term contributes nothing — a solo run is scored
        purely on pace.
        """
        outcome = compute_race_reward(
            obs,
            perception=self.sensor,
            n_perception=self.n_perception,
            waypoints_len=len(self.waypoints),
            waypoint_idx=self._waypoint_idx,
            checkpoint_hit=self._checkpoint_hit,
            last_dist=self._last_dist,
            current_dist=self._current_dist,
            last_damage=self._last_damage,
            steps=self._steps,
            invuln_steps=self._invuln_steps,
            steps_since_checkpoint=self._steps_since_checkpoint,
            max_steps=self.MAX_STEPS,
            max_damage=self.MAX_DAMAGE,
            progress_m=self.progress_m(),
            last_progress_m=self._last_progress_m,
            path_alignment=self._path_alignment,
            segment_len_m=self._path_pos.segment_len_m,
        )
        self._last_dist = outcome.last_dist
        self._last_damage = outcome.last_damage
        self._invuln_steps = outcome.invuln_steps
        self._checkpoint_hit = outcome.checkpoint_hit
        self._waypoint_idx = outcome.waypoint_idx
        self._steps_since_checkpoint = outcome.steps_since_checkpoint
        self._last_progress_m = outcome.progress_m
        return outcome.reward, outcome.done

    def _update_active_marker(self, idx: int):
        """Draw a bright sphere in-game on the current target waypoint.

        Uses bng.debug (beamngpy >= 1.26).  Silently skipped on older builds.
        """
        if self.bng is None or not self.waypoints:
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
