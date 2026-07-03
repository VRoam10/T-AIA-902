"""Tests for core.pipeline_actions benchmark helpers.

Covers the BeamNG env threading, agent parameter wiring, and the
comparison/single/run_benchmark entry points.
"""

from unittest.mock import MagicMock

import pytest

import core.pipeline_actions as pipeline_actions
from core.pipeline_actions import (
    BeamNGOptions,
    BenchmarkRequest,
    _benchmark_agent_params,
    _benchmark_env,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_taxi_registry(monkeypatch):
    """A minimal discrete env registered as 'taxi'."""

    def _factory(**kwargs):
        env = MagicMock()
        env.reset.return_value = (0, {})
        env.step.return_value = (0, 1.0, True, {})
        return env

    info = {
        "factory": _factory,
        "metadata": {"n_states": 500, "n_actions": 6, "state_type": "discrete"},
    }
    monkeypatch.setattr(pipeline_actions.registry, "get_environment", lambda n: info)
    return info


@pytest.fixture
def fake_beamng_registry(monkeypatch):
    """A minimal continuous env registered as 'beamng'."""
    captured = {}

    def _factory(**kwargs):
        captured.update(kwargs)
        env = MagicMock()
        env.reset.return_value = ([0.0] * 4, {})
        env.step.return_value = ([0.0] * 4, 1.0, False, {})
        return env

    info = {
        "factory": _factory,
        "metadata": {"n_states": 4, "n_actions": 2, "state_type": "continuous"},
    }
    monkeypatch.setattr(pipeline_actions.registry, "get_environment", lambda n: info)
    return info, captured


@pytest.fixture
def fake_algorithms(monkeypatch):
    """Make get_algorithm return small fake defaults for the algos we test."""

    def _get(algo_name):
        # DDPG/TD3 accept state_type; DQN/Q-learning do not.
        class FakeDDPG:
            def __init__(self, n_states, n_actions, state_type="continuous", **kw):
                pass

        class FakeTD3:
            def __init__(self, n_states, n_actions, state_type="continuous", **kw):
                pass

        class FakeDQN:
            def __init__(self, n_states, n_actions, **kw):
                pass

        class FakeQL:
            def __init__(self, n_states, n_actions, **kw):
                pass

        sigs = {
            "ddpg": FakeDDPG,
            "td3": FakeTD3,
            "dqn": FakeDQN,
            "q_learning": FakeQL,
        }
        defaults = {
            "dqn": {"lr": 1e-3, "batch_size": 64},
            "ddpg": {"actor_lr": 3e-4, "batch_size": 64, "n_actions": 2},
            "td3": {"actor_lr": 3e-4, "batch_size": 64, "n_actions": 2},
            "q_learning": {"learning_rate": 0.85},
        }
        return {
            "class": sigs[algo_name],
            "default_config": dict(defaults[algo_name]),
        }

    monkeypatch.setattr(pipeline_actions.registry, "get_algorithm", _get)


# ---------------------------------------------------------------------------
# _benchmark_env
# ---------------------------------------------------------------------------
def test_benchmark_env_non_beamng_returns_factory_and_metadata(fake_taxi_registry):
    req = BenchmarkRequest(
        benchmark_name="convergence",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        algo_name="q_learning",
        env_name="taxi",
    )
    factory, metadata = _benchmark_env(req)
    assert factory is fake_taxi_registry["factory"]
    assert metadata == fake_taxi_registry["metadata"]


def test_benchmark_env_beamng_forwards_options(fake_beamng_registry):
    info, captured = fake_beamng_registry
    req = BenchmarkRequest(
        benchmark_name="convergence",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        algo_name="ddpg",
        env_name="beamng",
        beamng=BeamNGOptions(
            map_name="italy",
            vehicle_id="super",
            trajectory_hints=3,
            body_orientation=True,
            wheel_terrain=True,
            random_path=True,
            dense_episodes=5,
        ),
    )
    factory, metadata = _benchmark_env(req, algo_name="ddpg")
    factory()
    assert captured["map_name"] == "italy"
    assert captured["vehicle_id"] == "super"
    assert captured["trajectory_hints"] == 3
    assert captured["body_orientation"] is True
    assert captured["wheel_terrain"] is True


def test_benchmark_env_beamng_excludes_training_only_options(fake_beamng_registry):
    info, captured = fake_beamng_registry
    req = BenchmarkRequest(
        benchmark_name="convergence",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        algo_name="ddpg",
        env_name="beamng",
        beamng=BeamNGOptions(random_path=True, dense_episodes=5),
    )
    _benchmark_env(req, algo_name="ddpg")
    assert "random_path" not in captured
    assert "dense_episodes" not in captured


def test_benchmark_env_beamng_includes_reward_mode_for_ddpg_td3(fake_beamng_registry):
    info, captured = fake_beamng_registry
    req = BenchmarkRequest(
        benchmark_name="convergence",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        algo_name="ddpg",
        env_name="beamng",
        beamng=BeamNGOptions(),
    )
    factory, _ = _benchmark_env(req, algo_name="ddpg")
    factory()
    assert captured.get("reward_mode") == "ddpg"
    captured.clear()
    factory2, _ = _benchmark_env(req, algo_name="td3")
    factory2()
    assert captured.get("reward_mode") == "td3"


def test_benchmark_env_beamng_omits_reward_mode_for_dqn_and_none(fake_beamng_registry):
    info, captured = fake_beamng_registry
    req = BenchmarkRequest(
        benchmark_name="convergence",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        algo_name="dqn",
        env_name="beamng",
        beamng=BeamNGOptions(),
    )
    factory, _ = _benchmark_env(req, algo_name="dqn")
    factory()
    assert "reward_mode" not in captured
    captured.clear()
    factory2, _ = _benchmark_env(req, algo_name=None)
    factory2()
    assert "reward_mode" not in captured


def test_benchmark_env_beamng_widens_n_states(fake_beamng_registry):
    info, captured = fake_beamng_registry
    req = BenchmarkRequest(
        benchmark_name="convergence",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        algo_name="ddpg",
        env_name="beamng",
        beamng=BeamNGOptions(trajectory_hints=2, body_orientation=True, wheel_terrain=True),
    )
    factory, metadata = _benchmark_env(req, algo_name="ddpg")
    # 4 + 2*2 + 2*1 + 2*1 = 4 + 4 + 2 + 2 = 12
    assert metadata["n_states"] == 12


def test_benchmark_env_beamng_defaults_when_none(fake_beamng_registry):
    info, captured = fake_beamng_registry
    req = BenchmarkRequest(
        benchmark_name="convergence",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        algo_name="ddpg",
        env_name="beamng",
        beamng=None,
    )
    factory, metadata = _benchmark_env(req, algo_name="ddpg")
    factory()
    # defaults: trajectory_hints=0, no flags → no widening
    assert metadata["n_states"] == info["metadata"]["n_states"]
    assert captured["map_name"] == "gridmap_v2"


def test_benchmark_env_does_not_mutate_registry_metadata(fake_beamng_registry, monkeypatch):
    info = fake_beamng_registry[0]
    original_n_states = info["metadata"]["n_states"]
    req = BenchmarkRequest(
        benchmark_name="convergence",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        algo_name="ddpg",
        env_name="beamng",
        beamng=BeamNGOptions(trajectory_hints=5),
    )
    _benchmark_env(req, algo_name="ddpg")
    assert info["metadata"]["n_states"] == original_n_states


# ---------------------------------------------------------------------------
# _benchmark_agent_params
# ---------------------------------------------------------------------------
def test_agent_params_returns_copy(fake_algorithms):
    params = _benchmark_agent_params("ddpg", {"state_type": "continuous", "n_actions": 2})
    params2 = _benchmark_agent_params("ddpg", {"state_type": "continuous", "n_actions": 2})
    assert params is not params2
    assert params == params2


def test_agent_params_adds_state_type_for_ddpg_td3(fake_algorithms):
    meta = {"state_type": "continuous", "n_actions": 2}
    params = _benchmark_agent_params("ddpg", meta)
    assert params.get("state_type") == "continuous"
    params = _benchmark_agent_params("td3", meta)
    assert params.get("state_type") == "continuous"


def test_agent_params_omits_state_type_for_dqn_q_learning(fake_algorithms):
    meta = {"state_type": "discrete", "n_actions": 6}
    params = _benchmark_agent_params("dqn", meta)
    assert "state_type" not in params
    params = _benchmark_agent_params("q_learning", meta)
    assert "state_type" not in params


def test_agent_params_discrete_overrides_n_actions(fake_algorithms):
    meta = {"state_type": "discrete", "n_actions": 6}
    # ddpg default has n_actions=2
    params = _benchmark_agent_params("ddpg", meta)
    assert params["n_actions"] == 6
    # td3 default has n_actions=2
    params = _benchmark_agent_params("td3", meta)
    assert params["n_actions"] == 6


def test_agent_params_continuous_leaves_n_actions_defaults(fake_algorithms):
    meta = {"state_type": "continuous", "n_actions": 2}
    params = _benchmark_agent_params("ddpg", meta)
    # default n_actions=2 should remain (not forced to None or different)
    assert params["n_actions"] == 2
    params = _benchmark_agent_params("td3", meta)
    assert params["n_actions"] == 2


# ---------------------------------------------------------------------------
# _run_comparison
# ---------------------------------------------------------------------------
def test_run_comparison_builds_variants_and_shares_factory(
    monkeypatch, fake_beamng_registry, fake_algorithms
):
    info, captured = fake_beamng_registry
    # Need a comparison benchmark class with a run() we can spy on.
    bench = MagicMock()
    bench.run.return_value = {"variants": {}}

    req = BenchmarkRequest(
        benchmark_name="comparison",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        env_name="beamng",
        algos=["dqn", "ddpg"],
    )
    from core.pipeline_actions import _run_comparison

    _run_comparison(
        bench,
        req,
        {
            "seeds": [0],
            "eval_episodes": 2,
            "success_threshold": 0,
            "max_episodes": 3,
            "reward_threshold": 7,
        },
    )

    # bench.run called with env_factory from _benchmark_env(request) – no reward_mode
    call_args = bench.run.call_args
    # bench.run(None, env_factory, config)  → positional args
    config = call_args[0][2]
    variants = config["variants"]
    assert len(variants) == 2
    assert variants[0]["name"] == "dqn"
    assert variants[0]["algo"] == "dqn"
    assert "agent_params" in variants[0]
    assert variants[1]["name"] == "ddpg"


# ---------------------------------------------------------------------------
# run_benchmark smoke test (taxi guard)
# ---------------------------------------------------------------------------
def test_run_benchmark_taxi_returns_ok(monkeypatch, fake_taxi_registry):
    # Provide a real-looking fake benchmark that can execute quickly.
    class FakeBench:
        def run(self, agent_cls, env_factory, config):
            return {"mean_reward": 1.0}

        def report(self, results):
            return "ok"

        def export(self, *a, **k):
            pass

    monkeypatch.setattr(pipeline_actions.registry, "get_benchmark", lambda n: {"class": FakeBench})
    monkeypatch.setattr(FakeBench, "run", lambda *a, **k: {"mean_reward": 1.0})
    monkeypatch.setattr(FakeBench, "report", lambda self, results: "report")
    monkeypatch.setattr(FakeBench, "export", lambda *a, **k: None)

    req = BenchmarkRequest(
        benchmark_name="convergence",
        seeds=[0],
        eval_episodes=2,
        success_threshold=0,
        max_episodes=3,
        algo_name="q_learning",
        env_name="taxi",
        reward_threshold=7,
    )
    result = run_benchmark(req)
    assert result["status"] == "ok"
    assert "report" in result
