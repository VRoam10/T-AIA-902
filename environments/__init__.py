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
        beamng_home=BEAMNG_HOME, beamng_user=BEAMNG_USER, headless=HEADLESS,
        reward_mode=reward_mode, vehicle_id=vehicle_id, map_name=map_name,
    )


registry.register_environment(
    "beamng",
    factory=_make_beamng,
    metadata={"n_states": 13, "n_actions": 7, "state_type": "continuous"},
)


def _make_beamng_continuous(vehicle_id="taxi", map_name="gridmap_v2", **_kwargs):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGContinuousEnv

    return BeamNGContinuousEnv(
        beamng_home=BEAMNG_HOME, beamng_user=BEAMNG_USER, headless=HEADLESS,
        vehicle_id=vehicle_id, map_name=map_name,
    )


registry.register_environment(
    "beamng_continuous",
    factory=_make_beamng_continuous,
    metadata={"n_states": 13, "n_actions": 3, "state_type": "continuous"},
)


def _make_beamng_camera(vehicle_id="taxi", map_name="gridmap_v2", **_kwargs):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGCameraEnv

    return BeamNGCameraEnv(
        beamng_home=BEAMNG_HOME, beamng_user=BEAMNG_USER, headless=HEADLESS,
        vehicle_id=vehicle_id, map_name=map_name,
    )


registry.register_environment(
    "beamng_camera",
    factory=_make_beamng_camera,
    metadata={"n_states": 261, "n_actions": 3, "state_type": "continuous"},
)


def _make_beamng_radar(vehicle_id="taxi", map_name="gridmap_v2", **_kwargs):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGRadarEnv

    return BeamNGRadarEnv(
        beamng_home=BEAMNG_HOME, beamng_user=BEAMNG_USER, headless=HEADLESS,
        vehicle_id=vehicle_id, map_name=map_name,
    )


registry.register_environment(
    "beamng_radar",
    factory=_make_beamng_radar,
    metadata={"n_states": 41, "n_actions": 3, "state_type": "continuous"},
)
