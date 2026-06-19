"""Validate the benchmark pipeline on the continuous (DDPG/TD3) path.

BeamNG cannot run in CI, so the continuous agents are exercised through a
lightweight stub environment that follows the same contract: an N-dimensional
continuous observation and a 2-dimensional continuous action in [-1, 1].
This guards the DDPG/TD3 path that BeamNG actually uses.
"""

import numpy as np
import pytest

import algorithms  # noqa: F401 — registers ddpg/td3 for registry-name resolution
from algorithms.ddpg import DDPGAgent
from algorithms.td3 import TD3Agent
from benchmarks.comparison import ComparisonBenchmark
from benchmarks.convergence import ConvergenceBenchmark

_FAST = {"warmup_steps": 4, "batch_size": 8, "memory_size": 500}

N_STATES = 6
N_ACTIONS = 2
EP_LEN = 10


class _ActionSpace:
    """Minimal action space exposing a seedable interface."""

    def __init__(self, dim: int):
        self.shape = (dim,)
        self._seed = None

    def seed(self, seed: int):
        self._seed = seed


class FakeContinuousEnv:
    """Minimal Gymnasium-style continuous environment (BeamNG-like contract)."""

    def __init__(self, n_states: int = N_STATES, n_actions: int = N_ACTIONS, ep_len: int = EP_LEN):
        self.n_states = n_states
        self.n_actions = n_actions
        self.ep_len = ep_len
        self.action_space = _ActionSpace(n_actions)
        self._t = 0
        self._state = np.zeros(n_states, dtype=np.float32)

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        self._t = 0
        self._state = np.random.uniform(-1.0, 1.0, self.n_states).astype(np.float32)
        return self._state, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).ravel()
        self._t += 1
        self._state = np.random.uniform(-1.0, 1.0, self.n_states).astype(np.float32)
        reward = float(-np.sum(np.abs(action)))
        done = self._t >= self.ep_len
        return self._state, reward, done, False, {"steps": self._t}

    def close(self):
        pass


def _factory():
    return FakeContinuousEnv()


@pytest.mark.parametrize(
    "agent_cls,params",
    [
        (
            DDPGAgent,
            {"warmup_steps": 4, "batch_size": 8, "memory_size": 500, "updates_per_step": 1},
        ),
        (
            TD3Agent,
            {"n_actions": N_ACTIONS, "warmup_steps": 4, "batch_size": 8, "memory_size": 500},
        ),
    ],
)
def test_continuous_agent_runs_through_convergence(agent_cls, params):
    bench = ConvergenceBenchmark()
    config = {
        "agent_params": params,
        "env_metadata": {"n_states": N_STATES, "n_actions": N_ACTIONS},
        "max_episodes": 8,
        "window": 3,
        "threshold": -1.0,
        "seed": 0,
        "eval_episodes": 3,
        "success_threshold": -100.0,
    }
    result = bench.run(agent_cls, _factory, config)
    assert result["total_episodes"] == 8
    assert "eval_mean_reward" in result
    assert "eval_success_rate" in result
    assert result["mean_steps"] > 0


def test_continuous_multiseed_aggregates():
    bench = ConvergenceBenchmark()
    config = {
        "agent_params": {
            "n_actions": N_ACTIONS,
            "warmup_steps": 4,
            "batch_size": 8,
            "memory_size": 500,
        },
        "env_metadata": {"n_states": N_STATES, "n_actions": N_ACTIONS},
        "max_episodes": 6,
        "window": 3,
        "threshold": -1.0,
        "seeds": [0, 1],
        "eval_episodes": 3,
        "success_threshold": -100.0,
    }
    multi = bench.run_multi(TD3Agent, _factory, config)
    assert multi["n_seeds"] == 2
    assert "eval_mean_reward" in multi["aggregate"]
    assert multi["aggregate"]["eval_mean_reward"]["n"] == 2


def test_continuous_comparison_ddpg_vs_td3():
    bench = ComparisonBenchmark()
    config = {
        "env_metadata": {"n_states": N_STATES, "n_actions": N_ACTIONS},
        "max_episodes": 6,
        "window": 3,
        "threshold": -1.0,
        "seeds": [0],
        "eval_episodes": 3,
        "success_threshold": -100.0,
        "variants": [
            {"name": "DDPG", "algo": "ddpg", "agent_params": {**_FAST, "updates_per_step": 1}},
            {"name": "TD3", "algo": "td3", "agent_params": _FAST},
        ],
    }
    result = bench.run(None, _factory, config)
    assert set(result["variants"]) == {"DDPG", "TD3"}
    for variant in result["variants"].values():
        assert "eval_mean_reward" in variant["aggregate"]
