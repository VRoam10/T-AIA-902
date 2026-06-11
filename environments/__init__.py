"""Register all environments with the pipeline registry."""

from core.registry import registry
from environments.taxi import make_taxi

registry.register_environment(
    "taxi",
    factory=make_taxi,
    metadata={"n_states": 500, "n_actions": 6, "state_type": "discrete"},
)


def _make_beamng(reward_mode="default", vehicle_id="taxi", map_name="gridmap_v2"):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGDrivingEnv

    return BeamNGDrivingEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        reward_mode=reward_mode,
        vehicle_id=vehicle_id,
        map_name=map_name,
    )


registry.register_environment(
    "beamng",
    factory=_make_beamng,
    metadata={"n_states": 14, "n_actions": 7, "state_type": "continuous"},
)


def _make_beamng_lidar(vehicle_id="taxi", map_name="gridmap_v2", **_kwargs):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGLidarEnv

    return BeamNGLidarEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
    )


registry.register_environment(
    "beamng_lidar",
    factory=_make_beamng_lidar,
    metadata={"n_states": 38, "n_actions": 7, "state_type": "continuous"},
)


def _make_beamng_continuous(vehicle_id="taxi", map_name="gridmap_v2", **_kwargs):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGContinuousEnv

    return BeamNGContinuousEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
    )


registry.register_environment(
    "beamng_continuous",
    factory=_make_beamng_continuous,
    metadata={"n_states": 14, "n_actions": 3, "state_type": "continuous"},
)


def _make_beamng_camera(vehicle_id="taxi", map_name="gridmap_v2", **_kwargs):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGCameraEnv

    return BeamNGCameraEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
    )


registry.register_environment(
    "beamng_camera",
    factory=_make_beamng_camera,
    metadata={"n_states": 262, "n_actions": 3, "state_type": "continuous"},
)


# --- Predicted variants (trajectory_hints=1: adds 2 floats per next waypoint) ---


def _make_beamng_predicted(reward_mode="default", vehicle_id="taxi", map_name="gridmap_v2"):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGDrivingEnv

    return BeamNGDrivingEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        reward_mode=reward_mode,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=1,
    )


registry.register_environment(
    "beamng_predicted",
    factory=_make_beamng_predicted,
    metadata={"n_states": 16, "n_actions": 7, "state_type": "continuous"},
)


def _make_beamng_continuous_predicted(vehicle_id="taxi", map_name="gridmap_v2", **_kwargs):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGContinuousEnv

    return BeamNGContinuousEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=1,
    )


registry.register_environment(
    "beamng_continuous_predicted",
    factory=_make_beamng_continuous_predicted,
    metadata={"n_states": 16, "n_actions": 3, "state_type": "continuous"},
)


def _make_beamng_lidar_predicted(vehicle_id="taxi", map_name="gridmap_v2", **_kwargs):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGLidarEnv

    return BeamNGLidarEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=1,
    )


registry.register_environment(
    "beamng_lidar_predicted",
    factory=_make_beamng_lidar_predicted,
    metadata={"n_states": 40, "n_actions": 7, "state_type": "continuous"},
)


def _make_beamng_camera_predicted(vehicle_id="taxi", map_name="gridmap_v2", **_kwargs):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGCameraEnv

    return BeamNGCameraEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=1,
    )


registry.register_environment(
    "beamng_camera_predicted",
    factory=_make_beamng_camera_predicted,
    metadata={"n_states": 264, "n_actions": 3, "state_type": "continuous"},
)
