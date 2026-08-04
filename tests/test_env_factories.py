"""The BeamNG env factory must forward every axis and flag to the env.

Regression this guards: the CLI collected `random_path` but the registry factory
wrapper dropped it, so single-agent training always ran with random_path=False.
The sensor/output axes reach the env the same way, so they are covered here too —
a dropped `output` would silently give a continuous agent a discrete env.
"""

import pytest

from environments import beamng_spec


def _factory():
    import environments  # noqa: F401  (importing registers the env)
    from core.registry import registry

    return registry.get_environment("beamng")["factory"]


def test_beamng_factory_forwards_random_path():
    env = _factory()(map_name="italy", random_path=True)
    assert env.random_path is True


def test_beamng_factory_random_path_defaults_false():
    env = _factory()(map_name="italy")
    assert env.random_path is False


@pytest.mark.parametrize("sensor", beamng_spec.SENSORS)
def test_beamng_factory_forwards_sensor(sensor):
    env = _factory()(map_name="italy", sensor=sensor)
    assert env.sensor == sensor
    assert env.n_states == beamng_spec.obs_size(sensor)


@pytest.mark.parametrize("output", beamng_spec.OUTPUTS)
def test_beamng_factory_forwards_output(output):
    env = _factory()(map_name="italy", output=output)
    assert env.output == output
    assert env.n_actions == beamng_spec.action_size(output)


def test_beamng_factory_defaults_to_lidar_and_fixed():
    env = _factory()(map_name="italy")
    assert (env.sensor, env.output) == (beamng_spec.DEFAULT_SENSOR, "fixed")


def test_beamng_factory_forwards_observation_flags():
    env = _factory()(
        map_name="italy", sensor="adv_lidar", trajectory_hints=2, body_orientation=True
    )
    assert env.n_states == beamng_spec.obs_size("adv_lidar", 2, True, False)


def test_unknown_sensor_is_rejected_at_construction():
    # Fail loudly here rather than build an agent whose input size cannot match.
    with pytest.raises(ValueError, match="unknown sensor"):
        _factory()(map_name="italy", sensor="radar")


def test_unknown_output_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown output"):
        _factory()(map_name="italy", output="discrete")


def test_beamng_factory_forwards_track():
    # Same regression as random_path above, second occurrence: the factory's
    # **_kwargs sink swallowed `track`, so choosing a game track in the TUI ran the
    # generated paths instead — with no error to show why.
    env = _factory()(map_name="italy", track="mixedCircuit1")
    assert env.track == "mixedCircuit1"


def test_beamng_factory_track_defaults_to_none():
    assert _factory()(map_name="italy").track is None


def test_beamng_factory_treats_an_empty_track_as_none():
    # The TUI/request layer uses "" for "generated paths"; the env expects None.
    assert _factory()(map_name="italy", track="").track is None


def test_factory_forwards_every_beamng_option_it_accepts():
    """Guards the bug *class*, not just `track`.

    The factory takes **_kwargs, so a request option it forgets to name is dropped
    in silence and the run uses the default. This asserts that every field of
    BeamNGOptions the env also has an attribute for actually arrives, so the next
    option added cannot repeat the same disappearance.
    """
    from dataclasses import fields

    from core.pipeline_actions import BeamNGOptions

    probes = {
        "map_name": "italy",
        "sensor": "adv_lidar",
        "trajectory_hints": 3,
        "body_orientation": True,
        "road_info": True,
        "wheel_info": True,
        "random_path": True,
        "dense_episodes": 7,
        "track": "mixedCircuit1",
    }
    # Every option must have a probe value, or this test is silently incomplete.
    assert {f.name for f in fields(BeamNGOptions)} == set(probes)

    env = _factory()(**probes)
    for name, expected in probes.items():
        assert getattr(env, name) == expected, f"factory dropped {name!r}"
