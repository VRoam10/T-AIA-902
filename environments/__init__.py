"""Register the BeamNG environment with the pipeline registry.

There is one environment. The two axes that used to be baked into four registered
names (``beamng``, ``beamng_lidar``, ``beamng_continuous``, ``beamng_camera``) are
now constructor arguments — ``sensor`` and ``output`` — because they are
independent: ``beamng`` and ``beamng_continuous`` produced identical observations
and differed only in how ``step()`` read the action.

Observation and action sizes therefore depend on the *request*, not on the
registered name, so they are no longer stored in the metadata. Callers size agents
with ``beamng_spec.obs_size`` / ``beamng_spec.action_size`` (see
``core.pipeline_actions.build_agent``).
"""

from core.registry import registry
from environments import beamng_spec


def _make_beamng(
    sensor: str = beamng_spec.DEFAULT_SENSOR,
    output: str = "fixed",
    map_name: str = "gridmap_v2",
    trajectory_hints: int = 0,
    body_orientation: bool = False,
    road_info: bool = False,
    random_path: bool = False,
    dense_episodes: int = 0,
    track: str | None = None,
    # Tolerates keys this factory does not model, so an unrelated caller passing
    # extras does not crash a run. Note the hazard: every env option MUST be named
    # explicitly above AND forwarded below, or this sink swallows it and the run
    # silently uses the default. That is exactly how `track` was lost once.
    **_kwargs,
):
    from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
    from environments.beamng import BeamNGDrivingEnv

    return BeamNGDrivingEnv(
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        sensor=sensor,
        output=output,
        map_name=map_name,
        trajectory_hints=trajectory_hints,
        body_orientation=body_orientation,
        road_info=road_info,
        random_path=random_path,
        dense_episodes=dense_episodes,
        track=track or None,
    )


registry.register_environment(
    "beamng",
    factory=_make_beamng,
    metadata={"state_type": "continuous"},
)
