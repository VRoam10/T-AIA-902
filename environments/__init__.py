"""Register all environments with the pipeline registry."""

from core.registry import registry
from environments.taxi import make_taxi

registry.register_environment(
    "taxi",
    factory=make_taxi,
    metadata={"n_states": 500, "n_actions": 6, "state_type": "discrete"},
)


def _make_beamng(
    reward_mode="default",
    vehicle_id="taxi",
    map_name="gridmap_v2",
    trajectory_hints=0,
    body_orientation=False,
    wheel_terrain=False,
    random_path=False,
    dense_episodes=0,
):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGDrivingEnv

    return BeamNGDrivingEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        reward_mode=reward_mode,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=trajectory_hints,
        body_orientation=body_orientation,
        wheel_terrain=wheel_terrain,
        random_path=random_path,
        dense_episodes=dense_episodes,
    )


registry.register_environment(
    "beamng",
    factory=_make_beamng,
    metadata={"n_states": 14, "n_actions": 7, "state_type": "continuous"},
)


def _make_beamng_lidar(
    vehicle_id="taxi",
    map_name="gridmap_v2",
    trajectory_hints=0,
    body_orientation=False,
    wheel_terrain=False,
    random_path=False,
    dense_episodes=0,
    **_kwargs,
):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGLidarEnv

    return BeamNGLidarEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=trajectory_hints,
        body_orientation=body_orientation,
        wheel_terrain=wheel_terrain,
        random_path=random_path,
        dense_episodes=dense_episodes,
    )


registry.register_environment(
    "beamng_lidar",
    factory=_make_beamng_lidar,
    metadata={"n_states": 38, "n_actions": 7, "state_type": "continuous"},
)


def _make_beamng_continuous(
    vehicle_id="taxi",
    map_name="gridmap_v2",
    trajectory_hints=0,
    body_orientation=False,
    wheel_terrain=False,
    random_path=False,
    dense_episodes=0,
    **_kwargs,
):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGContinuousEnv

    return BeamNGContinuousEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=trajectory_hints,
        body_orientation=body_orientation,
        wheel_terrain=wheel_terrain,
        random_path=random_path,
        dense_episodes=dense_episodes,
    )


registry.register_environment(
    "beamng_continuous",
    factory=_make_beamng_continuous,
    metadata={"n_states": 14, "n_actions": 3, "state_type": "continuous"},
)


def _make_beamng_camera(
    vehicle_id="taxi",
    map_name="gridmap_v2",
    trajectory_hints=0,
    body_orientation=False,
    wheel_terrain=False,
    random_path=False,
    dense_episodes=0,
    **_kwargs,
):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGCameraEnv

    return BeamNGCameraEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        vehicle_id=vehicle_id,
        map_name=map_name,
        trajectory_hints=trajectory_hints,
        body_orientation=body_orientation,
        wheel_terrain=wheel_terrain,
        random_path=random_path,
        dense_episodes=dense_episodes,
    )


registry.register_environment(
    "beamng_camera",
    factory=_make_beamng_camera,
    metadata={"n_states": 262, "n_actions": 3, "state_type": "continuous"},
)
