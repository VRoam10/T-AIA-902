"""Tests for reproducible seeding (core.seeding) and seeded training runs."""

import numpy as np

from algorithms.q_learning import QLearningAgent
from core.runner import PipelineRunner
from core.seeding import seed_action_space, set_global_seed
from environments.taxi import make_taxi


def test_set_global_seed_makes_numpy_deterministic():
    set_global_seed(123)
    a = np.random.rand(5)
    set_global_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_different_seeds_differ():
    set_global_seed(1)
    a = np.random.rand(5)
    set_global_seed(2)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_seed_action_space_is_safe_without_action_space():
    class Dummy:
        pass

    # Should not raise even though there is no action_space.
    seed_action_space(Dummy(), 0)


def _train_qlearning(seed, n_episodes=30):
    env = make_taxi()
    agent = QLearningAgent(n_states=500, n_actions=6, epsilon_decay=0.99)
    runner = PipelineRunner()
    runner.train(agent, env, n_episodes=n_episodes, seed=seed)
    env.close()
    return agent.q_table.copy()


def test_same_seed_reproduces_q_table():
    q1 = _train_qlearning(seed=42)
    q2 = _train_qlearning(seed=42)
    assert np.array_equal(q1, q2)


def test_different_seed_changes_q_table():
    q1 = _train_qlearning(seed=42)
    q2 = _train_qlearning(seed=7)
    assert not np.array_equal(q1, q2)
