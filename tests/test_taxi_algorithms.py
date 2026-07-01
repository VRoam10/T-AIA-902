"""Regression tests: every algorithm the pipeline offers for Taxi-v3 must run on it.

Taxi-v3 emits a *discrete integer* observation and expects a discrete action in
``Discrete(6)``. Neural-network agents therefore have to one-hot encode the
integer state; the continuous-control agents (DDPG / TD3) additionally have to
bridge their continuous actor to a discrete action (score-per-action + argmax).
Before this was fixed, dqn / dqn_per crashed with a matmul shape mismatch and
ddpg / td3 crashed feeding a continuous vector into Taxi's discrete step.
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
ALL_TAXI_ALGOS = ["q_learning", "dqn", "dqn_per", "ddpg", "td3"]


# ---------------------------------------------------------------------------
# Registry: every algorithm is offered for Taxi
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algo", ALL_TAXI_ALGOS)
def test_algo_is_taxi_compatible(algo):
    assert "taxi" in registry.compatible_environments(algo)


@pytest.mark.parametrize("algo", ALL_TAXI_ALGOS)
def test_agent_action_space_matches_taxi(algo):
    # A discrete env fixes the action count; an algorithm's continuous-control
    # default (e.g. TD3's n_actions=2) must not shrink Taxi's Discrete(6).
    agent = build_agent(algo, "taxi")
    assert agent.n_actions == TAXI_ACTIONS


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


@pytest.mark.parametrize("algo", ["ddpg", "td3"])
def test_continuous_algo_trains_on_taxi_without_crashing(algo):
    # Regression: DDPG/TD3 are continuous-control algorithms; on Taxi their
    # actor now scores each discrete action and argmax picks a valid Discrete(6)
    # action. They must train without crashing and emit integer actions.
    agent, history = _train_on_taxi(
        algo,
        n_episodes=2,
        agent_params={"batch_size": 16, "memory_size": 1000, "warmup_steps": 32},
    )
    assert len(history["rewards"]) == 2
    assert agent.train_steps > 0
    agent.epsilon = 0.0
    action = agent.select_action(0)
    assert isinstance(action, int)
    assert 0 <= action < TAXI_ACTIONS


def test_q_learning_solves_taxi():
    # Given enough episodes, tabular Q-learning must reach a near-optimal
    # greedy policy on Taxi-v3 (optimal average return is ~+8).
    agent, _ = _train_on_taxi("q_learning", n_episodes=1200, seed=0)
    env = gym.make("Taxi-v3")
    runner = PipelineRunner()
    metrics = runner.evaluate(agent, env, n_episodes=30, seed=10_000)
    env.close()
    assert metrics["avg_reward"] >= 6.0
