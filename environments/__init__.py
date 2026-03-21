"""Register all environments with the pipeline registry."""

from core.registry import registry
from environments.taxi import make_taxi

registry.register_environment(
    "taxi",
    factory=make_taxi,
    metadata={"n_states": 500, "n_actions": 6, "state_type": "discrete"},
)


def _make_beamng(headless=True):
    from config import BEAMNG_HOME, BEAMNG_USER
    from environments.beamng import BeamNGDrivingEnv

    return BeamNGDrivingEnv(
        beamng_home=BEAMNG_HOME, beamng_user=BEAMNG_USER, headless=headless
    )


registry.register_environment(
    "beamng",
    factory=_make_beamng,
    metadata={"n_states": 7, "n_actions": 7, "state_type": "continuous"},
)
