"""The two BeamNG configuration axes, and every size derived from them.

A racing setup is described by two independent choices:

  * ``sensor``  — which perception block feeds the observation
  * ``output``  — whether the action head is discrete or continuous

These used to be tangled into four env class names (``beamng``,
``beamng_lidar``, ``beamng_continuous``, ``beamng_camera``), which could not
express them independently: ``beamng`` and ``beamng_continuous`` had identical
observations and differed only in ``step()``. Now there is one ``beamng`` env
parameterized by both axes, and this module is the single source of truth for
the axes and the observation/action sizes they imply.

``output`` is deliberately *not* a user-facing choice — the algorithm already
determines it (a DQN head cannot emit continuous controls, and DDPG/TD3 emit
nothing else), so ``output_for_algo`` derives it.
"""

SENSORS: tuple[str, ...] = ("lidar", "adv_lidar", "camera")
OUTPUTS: tuple[str, ...] = ("fixed", "continuous")

DEFAULT_SENSOR = "lidar"

# Perception features contributed by each sensor. These reproduce the historical
# per-class layouts exactly (lidar 8, adv_lidar 4x8=32, camera 16x16=256), so the
# observation contract — and the reward's obs[:6] / obs[6:6+n] slicing — is unchanged.
PERCEPTION_FEATURES: dict[str, int] = {"lidar": 8, "adv_lidar": 32, "camera": 256}

# speed, steering, heading_err, lateral_err, damage, dist
KINEMATIC_FEATURES = 6

# Optional observation tails, in the order _observe concatenates them.
# road:  [edge_left, edge_right, road_heading, curvature, ahead_fwd, ahead_left]
ROAD_FEATURES = 6

# wheel: [long_slip, slip_angle, abs_active, lat_g]
WHEEL_FEATURES = 4

# Horizontal azimuth bins, shared by both lidar sensors.
LIDAR_RAYS = 8
# Values stored per cell (currently just distance).
LIDAR_CHANNELS_PER_RAY = 1

# Per-sensor LiDAR geometry: how many elevation bins the observation exposes, and
# the physical sensor's vertical resolution / field of view.
#   lidar     — one collapsed row; the sensor sweeps a wide FOV for dense coverage.
#   adv_lidar — 4 elevation rows, so the policy can tell a wall (fills every row)
#               from a low object (bottom row only). Narrower FOV so the rows span
#               useful elevations rather than mostly sky and mostly road.
LIDAR_GEOMETRY: dict[str, dict[str, float]] = {
    "lidar": {"v_bins": 1, "vert_res": 32, "vert_angle": 26.9},
    "adv_lidar": {"v_bins": 4, "vert_res": 16, "vert_angle": 20.0},
}

# Camera perception: captured at CAM_RESOLUTION, downsampled to CAM_OUT_SIZE
# grayscale, then flattened (16 * 16 = 256 = PERCEPTION_FEATURES["camera"]).
CAM_RESOLUTION = (84, 84)
CAM_OUT_SIZE = (16, 16)
CAM_FOV_Y = 70
CAM_POS = (0, -0.5, 1.5)
CAM_DIR = (0, -1, 0)

# Algorithm -> output axis. A discrete-action algorithm drives the 7-entry ACTIONS
# table; a continuous one drives (throttle, steering, brake) directly.
FIXED_ALGOS: tuple[str, ...] = ("dqn", "dqn_per")
CONTINUOUS_ALGOS: tuple[str, ...] = ("ddpg", "td3")

# Action-space sizes per output axis. "fixed" must match len(BeamNGDrivingEnv.ACTIONS).
FIXED_N_ACTIONS = 7
CONTINUOUS_N_ACTIONS = 3  # throttle, steering, brake

# The one car: the Cherrier Vivace Hillclimb (Sequential) — 682 hp, 1420 kg, AWD.
# Everything races the same machine, so a head-to-head result reflects the policies and
# not the hardware. Kept as a dict of beamngpy Vehicle kwargs (the shape the multi/race
# envs already expect) so only `color` varies per entrant.
RACE_CAR: dict[str, str] = {
    "model": "vivace",
    "licence": "RACE",
    "color": "White",
    "part_config": "vehicles/vivace/hillclimb_SQ.pc",
}

AVAILABLE_MAPS: tuple[str, ...] = (
    "gridmap_v2",
    "italy",
    "west_coast_usa",
    "east_coast_usa",
)

# --- Simulation timing -------------------------------------------------------
# The envs call bng.set_deterministic(PHYSICS_STEPS_PER_SECOND) and then advance
# PHYSICS_STEPS_PER_ENV_STEP simulation steps per env step. In deterministic mode
# each simulation step is exactly 1/PHYSICS_STEPS_PER_SECOND of sim time, so one
# env step is SECONDS_PER_ENV_STEP long and the control rate is its reciprocal.
#
# These are stated here because anything reasoning in *seconds* — the reward's
# segment-time par, the realtime race tick — must derive from them rather than
# assume. (A long-standing comment claimed ~100 ms per env step, which is 3x off.)
PHYSICS_STEPS_PER_SECOND = 30
PHYSICS_STEPS_PER_ENV_STEP = 10
SECONDS_PER_ENV_STEP = PHYSICS_STEPS_PER_ENV_STEP / PHYSICS_STEPS_PER_SECOND  # 0.333 s


def steps_for_distance(metres: float, speed_ms: float) -> int:
    """Env steps needed to cover ``metres`` at ``speed_ms``, rounded up.

    Used to set time-based reward targets from real geometry instead of a guessed
    step count. Returns at least 1.
    """
    if speed_ms <= 0:
        raise ValueError("speed_ms must be positive")
    per_step = speed_ms * SECONDS_PER_ENV_STEP
    return max(1, int(-(-metres // per_step)))


def output_for_algo(algo: str) -> str:
    """Return the output axis ("fixed" or "continuous") implied by an algorithm.

    Raises ValueError for an unregistered algorithm rather than guessing: a wrong
    guess silently builds an agent with the wrong action head, which shows up much
    later as garbage driving.
    """
    if algo in FIXED_ALGOS:
        return "fixed"
    if algo in CONTINUOUS_ALGOS:
        return "continuous"
    raise ValueError(
        f"unknown algorithm {algo!r}; expected one of {FIXED_ALGOS + CONTINUOUS_ALGOS}"
    )


def perception_features(sensor: str) -> int:
    """Number of perception values the given sensor contributes."""
    if sensor not in PERCEPTION_FEATURES:
        raise ValueError(f"unknown sensor {sensor!r}; expected one of {SENSORS}")
    return PERCEPTION_FEATURES[sensor]


def obs_size(
    sensor: str,
    trajectory_hints: int = 0,
    body_orientation: bool = False,
    road_info: bool = False,
    wheel_info: bool = False,
) -> int:
    """Observation length for a sensor plus the optional observation flags.

    Layout (blocks appended in this order, matching ``_observe``):

        kinematic(6) | perception(P) | hints(2*H) | [pitch, roll]? | road(6)? | wheel(4)?
    """
    return (
        KINEMATIC_FEATURES
        + perception_features(sensor)
        + 2 * int(trajectory_hints)
        + (2 if body_orientation else 0)
        + (ROAD_FEATURES if road_info else 0)
        + (WHEEL_FEATURES if wheel_info else 0)
    )


def action_size(output: str) -> int:
    """Number of action outputs for the given output axis."""
    if output == "fixed":
        return FIXED_N_ACTIONS
    if output == "continuous":
        return CONTINUOUS_N_ACTIONS
    raise ValueError(f"unknown output {output!r}; expected one of {OUTPUTS}")


def lidar_geometry(sensor: str) -> dict[str, float]:
    """LiDAR sensor geometry for a lidar sensor. Raises for camera perception."""
    if sensor not in LIDAR_GEOMETRY:
        raise ValueError(f"{sensor!r} is not a LiDAR sensor; expected one of {tuple(LIDAR_GEOMETRY)}")
    return LIDAR_GEOMETRY[sensor]


def is_lidar(sensor: str) -> bool:
    """True when the sensor's perception block is a LiDAR range field.

    The reward's obstacle-proximity penalty only makes sense on ranges, and the
    human-play diagnostics only exist for LiDAR, so both gate on this.
    """
    return sensor in LIDAR_GEOMETRY
