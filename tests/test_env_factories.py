"""The BeamNG env factories must forward the random_path flag to the env.

Regression: the CLI collected `random_path` but the registry factory wrappers
dropped it, so single-agent training always ran with random_path=False.
"""

import pytest

ENV_NAMES = ["beamng", "beamng_lidar", "beamng_continuous", "beamng_camera"]


def _factory(env_name):
    import environments  # noqa: F401  (importing registers the envs)
    from core.registry import registry

    return registry.get_environment(env_name)["factory"]


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_beamng_factory_forwards_random_path(env_name):
    env = _factory(env_name)(map_name="italy", random_path=True)
    assert env.random_path is True


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_beamng_factory_random_path_defaults_false(env_name):
    env = _factory(env_name)(map_name="italy")
    assert env.random_path is False
