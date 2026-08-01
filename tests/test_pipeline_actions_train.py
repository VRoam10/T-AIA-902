"""Tests for run_train's output paths.

The training plot must always land beside the checkpoint it describes. The model
name is the user's (the TUI derives it from algo + sensor + options), so deriving
the plot name from anything else lets the two drift apart — and a plot named after
a stem the user never chose overwrites whatever model already owned that name.
"""

from unittest.mock import MagicMock

import pytest

import core.pipeline_actions as pipeline_actions
from core.pipeline_actions import BeamNGOptions, TrainRequest


@pytest.fixture
def fake_beamng_registry(monkeypatch):
    """A minimal continuous env registered as 'beamng'."""

    def _factory(**kwargs):
        env = MagicMock()
        env.reset.return_value = ([0.0] * 4, {})
        env.step.return_value = ([0.0] * 4, 1.0, False, {})
        return env

    info = {
        "factory": _factory,
        "metadata": {"n_states": 4, "n_actions": 2, "state_type": "continuous"},
    }
    monkeypatch.setattr(pipeline_actions.registry, "get_environment", lambda n: info)
    return info


@pytest.fixture
def fake_algorithms(monkeypatch):
    """Return a tiny stand-in agent class for the algos under test."""

    def _get(algo_name):
        class FakeDDPG:
            def __init__(self, n_states, n_actions, state_type="continuous", **kw):
                pass

        return {"class": FakeDDPG, "default_config": {"actor_lr": 3e-4, "n_actions": 2}}

    monkeypatch.setattr(pipeline_actions.registry, "get_algorithm", _get)


@pytest.fixture
def captured_runner(monkeypatch):
    """Replace PipelineRunner so the paths are asserted without a training loop."""
    calls = {}

    class FakeRunner:
        def train(self, agent, env, **kwargs):
            calls.update(kwargs)
            return {"rewards": [1.0]}

    monkeypatch.setattr(pipeline_actions, "PipelineRunner", FakeRunner)
    return calls


def _request(save_path: str) -> TrainRequest:
    return TrainRequest(
        algo_name="ddpg",
        env_name="beamng",
        n_episodes=1,
        save_path=save_path,
        agent_params={},
        beamng=BeamNGOptions(sensor="lidar"),
    )


def test_plot_lands_beside_the_model(
    fake_beamng_registry, fake_algorithms, captured_runner, tmp_path
):
    model = tmp_path / "ddpg_lidar.pth"
    pipeline_actions.run_train(_request(str(model)))
    assert captured_runner["plot_path"] == str(tmp_path / "ddpg_lidar_training.png")


def test_a_custom_model_name_keeps_its_own_plot(
    fake_beamng_registry, fake_algorithms, captured_runner, tmp_path
):
    """The plot must not fall back to an algo/env stem another model already owns."""
    model = tmp_path / "my_experiment.pth"
    pipeline_actions.run_train(_request(str(model)))
    assert captured_runner["plot_path"] == str(tmp_path / "my_experiment_training.png")
    assert "ddpg_beamng_training.png" not in captured_runner["plot_path"]


def test_an_extensionless_model_path_still_gets_a_plot(
    fake_beamng_registry, fake_algorithms, captured_runner, tmp_path
):
    model = tmp_path / "ddpg_lidar"
    pipeline_actions.run_train(_request(str(model)))
    assert captured_runner["plot_path"] == str(tmp_path / "ddpg_lidar_training.png")


def test_the_plot_path_is_reported_back_to_the_caller(
    fake_beamng_registry, fake_algorithms, captured_runner, tmp_path
):
    """The TUI shows where the run's artefacts went, so the result carries both."""
    model = tmp_path / "ddpg_lidar.pth"
    result = pipeline_actions.run_train(_request(str(model)))
    assert result["plot_path"] == str(tmp_path / "ddpg_lidar_training.png")
