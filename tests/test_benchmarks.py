"""Smoke and determinism tests for the benchmark suite.

These exercise the benchmark *machinery* — seeded determinism, multi-seed
aggregation, variant comparison, gridsearch ranking — so they run against
``LineWorld`` (see conftest) rather than BeamNG, which needs a live simulator.
DQN is the agent under test because it is the cheapest registered algorithm.
"""

import algorithms  # noqa: F401 — triggers registry auto-registration
import benchmarks  # noqa: F401 — triggers registry auto-registration
import environments  # noqa: F401 — triggers registry auto-registration
from core.registry import registry
from tests.conftest import LINEWORLD_METADATA, LineWorld

# Small nets and a tiny replay buffer: these tests must be fast, and none of them
# assert anything about how well the agent learns.
AGENT_PARAMS = {
    "lr": 1e-2,
    "gamma": 0.9,
    "epsilon": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.9,
    "batch_size": 8,
    "memory_size": 500,
    "target_update_freq": 20,
    "hidden": 16,
}


def _dqn():
    return registry.get_algorithm("dqn")


def _config(**overrides):
    config = {
        "agent_params": dict(AGENT_PARAMS),
        "env_metadata": dict(LINEWORLD_METADATA),
        "max_episodes": 12,
        "window": 5,
        "eval_episodes": 3,
    }
    config.update(overrides)
    return config


def test_convergence_reports_eval_metrics():
    bench = registry.get_benchmark("convergence")["class"]()
    result = bench.run(_dqn()["class"], LineWorld, _config(seed=0))
    for key in ("converged", "eval_mean_reward", "eval_success_rate", "mean_steps"):
        assert key in result
    assert result["mean_steps"] > 0


def test_convergence_is_deterministic_with_seed():
    bench = registry.get_benchmark("convergence")["class"]()
    first = bench.run(_dqn()["class"], LineWorld, _config(seed=0))
    second = bench.run(_dqn()["class"], LineWorld, _config(seed=0))
    assert first["eval_mean_reward"] == second["eval_mean_reward"]


def test_run_multi_aggregates_over_seeds():
    bench = registry.get_benchmark("convergence")["class"]()
    multi = bench.run_multi(_dqn()["class"], LineWorld, _config(seeds=[0, 1]))
    assert multi["n_seeds"] == 2
    assert len(multi["per_seed"]) == 2
    assert "eval_mean_reward" in multi["aggregate"]
    assert multi["aggregate"]["eval_mean_reward"]["n"] == 2


def test_comparison_is_agnostic_and_aggregates():
    bench = registry.get_benchmark("comparison")["class"]()
    config = _config(
        seeds=[0, 1],
        threshold=1.0,
        variants=[{"name": "DQN", "algo": "dqn", "agent_params": dict(AGENT_PARAMS)}],
    )
    result = bench.run(None, LineWorld, config)
    variant = result["variants"]["DQN"]
    assert variant["n_seeds"] == 2
    assert "eval_mean_reward" in variant["aggregate"]
    assert 0.0 <= variant["converged_rate"] <= 1.0


def test_gridsearch_ranks_configurations():
    bench = registry.get_benchmark("gridsearch")["class"]()
    config = _config(seeds=[0], param_grid={"lr": [1e-2, 5e-3], "gamma": [0.9, 0.99]})
    result = bench.run(_dqn()["class"], LineWorld, config)
    assert result["n_combinations"] == 4
    rewards = [entry["eval_mean_reward"] for entry in result["entries"]]
    assert rewards == sorted(rewards, reverse=True)
    assert result["best"] is result["entries"][0]
