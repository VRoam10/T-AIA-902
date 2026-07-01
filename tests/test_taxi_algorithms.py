"""Regression tests: every algorithm the pipeline offers for Taxi-v3 must run on it.

Taxi-v3 emits a *discrete integer* observation. Neural-network agents (DQN)
need a fixed-length feature vector, so the integer has to be one-hot encoded;
before this was fixed, DQN / dqn_per crashed on Taxi with a matmul shape
mismatch. Tabular Q-learning consumes the integer directly and, given enough
episodes, must solve the task. Continuous-control agents (DDPG / TD3) output
continuous actions and cannot drive Taxi's ``Discrete(6)`` space, so they must
not be advertised as Taxi-compatible.
"""

import gymnasium as gym
import numpy as np
import pytest

import algorithms  # noqa: F401 — importing registers the algorithms
import environments  # noqa: F401 — importing registers the environments
from algorithms.dqn import DQNAgent
from core.pipeline_actions import build_agent
from core.registry import registry
from core.runner import PipelineRunner

TAXI_STATES = 500
TAXI_ACTIONS = 6


# ---------------------------------------------------------------------------
# Registry: which algorithms are offered for Taxi
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ["q_learning", "dqn", "dqn_per"])
def test_discrete_algo_is_taxi_compatible(algo):
    assert "taxi" in registry.compatible_environments(algo)


@pytest.mark.parametrize("algo", ["ddpg", "td3"])
def test_continuous_algo_is_not_taxi_compatible(algo):
    # DDPG / TD3 emit continuous actions and cannot act in Taxi's Discrete(6).
    assert "taxi" not in registry.compatible_environments(algo)


# ---------------------------------------------------------------------------
# DQN: discrete-state encoding
# ---------------------------------------------------------------------------


class TestDQNDiscreteEncoding:
    def test_scalar_state_is_one_hot_encoded(self):
        agent = DQNAgent(TAXI_STATES, TAXI_ACTIONS)
        vec = agent._encode(350)
        assert vec.shape == (TAXI_STATES,)
        assert vec[350] == 1.0
        assert vec.sum() == pytest.approx(1.0)

    def test_vector_state_passes_through_unchanged(self):
        agent = DQNAgent(TAXI_STATES, TAXI_ACTIONS)
        obs = np.arange(TAXI_STATES, dtype=np.float32)
        vec = agent._encode(obs)
        assert np.array_equal(vec, obs)

    def test_select_action_accepts_scalar_state(self):
        agent = DQNAgent(TAXI_STATES, TAXI_ACTIONS)
        agent.epsilon = 0.0
        action = agent.select_action(499)
        assert 0 <= action < TAXI_ACTIONS


# ---------------------------------------------------------------------------
# Integration: agents actually run on the real Taxi-v3 environment
# ---------------------------------------------------------------------------


def _train_on_taxi(algo, n_episodes, agent_params=None, seed=0):
    agent = build_agent(algo, "taxi", agent_params)
    env = gym.make("Taxi-v3")
    runner = PipelineRunner()
    history = runner.train(agent, env, n_episodes=n_episodes, seed=seed)
    env.close()
    return agent, history


@pytest.mark.parametrize("algo", ["dqn", "dqn_per"])
def test_dqn_family_trains_on_taxi_without_crashing(algo):
    # Regression: the discrete Taxi state used to blow up the network with a
    # matmul shape mismatch. A couple of episodes with a small replay buffer
    # must train cleanly and keep selecting valid actions.
    agent, history = _train_on_taxi(
        algo, n_episodes=2, agent_params={"batch_size": 16, "memory_size": 1000}
    )
    assert len(history["rewards"]) == 2
    assert agent.train_steps > 0
    agent.epsilon = 0.0
    assert 0 <= agent.select_action(0) < TAXI_ACTIONS


def test_q_learning_solves_taxi():
    # Given enough episodes, tabular Q-learning must reach a near-optimal
    # greedy policy on Taxi-v3 (optimal average return is ~+8).
    agent, _ = _train_on_taxi("q_learning", n_episodes=1200, seed=0)
    env = gym.make("Taxi-v3")
    runner = PipelineRunner()
    metrics = runner.evaluate(agent, env, n_episodes=30, seed=10_000)
    env.close()
    assert metrics["avg_reward"] >= 6.0
