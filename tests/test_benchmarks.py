"""Smoke and determinism tests for the benchmark suite (Taxi + Q-Learning)."""

import algorithms  # noqa: F401 — triggers registry auto-registration
import benchmarks  # noqa: F401 — triggers registry auto-registration
import environments  # noqa: F401 — triggers registry auto-registration
from core.registry import registry


def _qlearning_and_taxi():
    return registry.get_algorithm("q_learning"), registry.get_environment("taxi")


def test_convergence_reports_eval_metrics():
    algo, env = _qlearning_and_taxi()
    bench = registry.get_benchmark("convergence")["class"]()
    config = {
        "agent_params": algo["default_config"],
        "env_metadata": env["metadata"],
        "max_episodes": 40,
        "window": 20,
        "seed": 0,
        "eval_episodes": 10,
    }
    result = bench.run(algo["class"], env["factory"], config)
    for key in ("converged", "eval_mean_reward", "eval_success_rate", "mean_steps"):
        assert key in result
    assert result["mean_steps"] > 0


def test_convergence_is_deterministic_with_seed():
    algo, env = _qlearning_and_taxi()
    bench = registry.get_benchmark("convergence")["class"]()
    config = {
        "agent_params": algo["default_config"],
        "env_metadata": env["metadata"],
        "max_episodes": 40,
        "window": 20,
        "seed": 0,
        "eval_episodes": 10,
    }
    first = bench.run(algo["class"], env["factory"], dict(config))
    second = bench.run(algo["class"], env["factory"], dict(config))
    assert first["eval_mean_reward"] == second["eval_mean_reward"]


def test_run_multi_aggregates_over_seeds():
    algo, env = _qlearning_and_taxi()
    bench = registry.get_benchmark("convergence")["class"]()
    config = {
        "agent_params": algo["default_config"],
        "env_metadata": env["metadata"],
        "max_episodes": 30,
        "window": 20,
        "seeds": [0, 1],
        "eval_episodes": 10,
    }
    multi = bench.run_multi(algo["class"], env["factory"], config)
    assert multi["n_seeds"] == 2
    assert len(multi["per_seed"]) == 2
    assert "eval_mean_reward" in multi["aggregate"]
    assert multi["aggregate"]["eval_mean_reward"]["n"] == 2


def test_comparison_is_agnostic_and_aggregates():
    _, env = _qlearning_and_taxi()
    bench = registry.get_benchmark("comparison")["class"]()
    config = {
        "env_metadata": env["metadata"],
        "max_episodes": 30,
        "window": 20,
        "threshold": 7.0,
        "seeds": [0, 1],
        "eval_episodes": 10,
        "variants": [{"name": "Q-Learning", "algo": "q_learning"}],
    }
    result = bench.run(None, env["factory"], config)
    variant = result["variants"]["Q-Learning"]
    assert variant["n_seeds"] == 2
    assert "eval_mean_reward" in variant["aggregate"]
    assert 0.0 <= variant["converged_rate"] <= 1.0


def test_gridsearch_ranks_configurations():
    algo, env = _qlearning_and_taxi()
    bench = registry.get_benchmark("gridsearch")["class"]()
    config = {
        "agent_params": algo["default_config"],
        "env_metadata": env["metadata"],
        "max_episodes": 30,
        "eval_episodes": 10,
        "seeds": [0],
        "param_grid": {"learning_rate": [0.3, 0.8], "discount_factor": [0.9, 0.99]},
    }
    result = bench.run(algo["class"], env["factory"], config)
    assert result["n_combinations"] == 4
    rewards = [entry["eval_mean_reward"] for entry in result["entries"]]
    assert rewards == sorted(rewards, reverse=True)
    assert result["best"] is result["entries"][0]
