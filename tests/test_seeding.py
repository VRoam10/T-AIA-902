"""Tests for reproducible seeding (core.seeding) and seeded training runs.

The seeded-run tests use ``LineWorld`` (see conftest) rather than BeamNG: the
property under test is that one seed reproduces one trajectory, which needs a
fast in-process env with a seedable action space.
"""

import numpy as np

from algorithms.dqn import DQNAgent
from core.runner import PipelineRunner
from core.seeding import seed_action_space, set_global_seed
from tests.conftest import LineWorld


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


def test_seed_action_space_seeds_a_real_action_space():
    env = LineWorld()
    seed_action_space(env, 7)
    first = [env.action_space.sample() for _ in range(10)]
    seed_action_space(env, 7)
    assert [env.action_space.sample() for _ in range(10)] == first


def _train_dqn(seed, n_episodes=6):
    # Seed BEFORE constructing the agent. A neural agent draws its initial weights
    # from the global torch RNG at construction time, so seeding only inside
    # runner.train() (which is what the benchmarks do) leaves weight init at the
    # mercy of whatever ran previously — two "identically seeded" runs then differ.
    set_global_seed(seed)
    env = LineWorld()
    agent = DQNAgent(
        n_states=env.n_states,
        n_actions=3,
        hidden=16,
        batch_size=8,
        memory_size=200,
        epsilon_decay=0.9,
        # Pinned to CPU: several CUDA kernels are non-deterministic, so on GPU the
        # same seed can still produce bitwise-different weights. The property under
        # test is that the seed fixes the *run*, not that CUDA is reproducible.
        device="cpu",
    )
    runner = PipelineRunner()
    runner.train(agent, env, n_episodes=n_episodes, seed=seed)
    env.close()
    return [p.detach().clone() for p in agent.q_net.parameters()]


def _same_weights(a, b) -> bool:
    return all(x.equal(y) for x, y in zip(a, b, strict=True))


def test_same_seed_reproduces_weights():
    assert _same_weights(_train_dqn(seed=42), _train_dqn(seed=42))


def test_different_seed_changes_weights():
    assert not _same_weights(_train_dqn(seed=42), _train_dqn(seed=7))
