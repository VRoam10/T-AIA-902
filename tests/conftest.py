"""Shared test configuration and fixtures.

Forces a non-interactive matplotlib backend so the benchmark/runner plotting code
never reaches for a GUI toolkit (Tk) during the test run — keeping the suite
headless and order-independent on CI and on machines without Tk.

Also provides :class:`LineWorld`, the tiny deterministic environment the runner
and benchmark tests train against. It replaces Gymnasium's Taxi, which those
tests previously used as a cheap stand-in: BeamNG needs a running simulator, so a
fast in-process env is required to exercise the generic train/evaluate/benchmark
machinery at all. LineWorld is deliberately trivial (a policy can solve it in a
handful of episodes) because these tests check the *plumbing* — determinism under
a seed, metric aggregation, ranking — not learning quality.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


class _ActionSpace:
    """Minimal Gymnasium-like discrete action space.

    ``seed()`` exists so ``core.seeding.seed_action_space`` has something real to
    act on — that is the behaviour the determinism tests depend on.
    """

    def __init__(self, n: int):
        self.n = n
        self._rng = np.random.default_rng(0)

    def seed(self, seed):
        self._rng = np.random.default_rng(seed)

    def sample(self) -> int:
        return int(self._rng.integers(self.n))


class LineWorld:
    """A 1-D corridor: walk right to the goal, in continuous observations.

    Observation is a length-``n_states`` float vector holding the normalized
    position (remaining entries are zero padding, so it can stand in for any
    observation width an agent expects). Actions are ``left / stay / right``.
    Reaching the goal ends the episode with a large bonus; every step costs a
    little, so a faster solve scores higher and the reward ordering is meaningful.

    Deterministic given a seed: ``reset(seed=...)`` fixes the start position, so
    two identically seeded runs produce identical trajectories.
    """

    LENGTH = 8
    STEP_COST = 0.05
    GOAL_REWARD = 10.0

    def __init__(self, n_states: int = 4, max_steps: int = 40):
        self.n_states = n_states
        self.max_steps = max_steps
        self.action_space = _ActionSpace(3)
        self._pos = 0
        self._steps = 0
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        self._pos = int(self._rng.integers(0, max(1, self.LENGTH // 2)))
        self._steps = 0
        return self._obs()

    def step(self, action):
        action = int(np.asarray(action).ravel()[0]) if np.ndim(action) else int(action)
        self._pos = int(np.clip(self._pos + (action - 1), 0, self.LENGTH))
        self._steps += 1

        reward = -self.STEP_COST
        done = False
        if self._pos >= self.LENGTH:
            reward += self.GOAL_REWARD
            done = True
        elif self._steps >= self.max_steps:
            done = True
        return self._obs(), reward, done, {"steps": self._steps}

    def close(self):
        pass

    def _obs(self) -> np.ndarray:
        obs = np.zeros(self.n_states, dtype=np.float32)
        obs[0] = self._pos / self.LENGTH
        return obs


LINEWORLD_METADATA = {"n_states": 4, "n_actions": 3, "state_type": "continuous"}


@pytest.fixture
def lineworld_factory():
    """Factory + metadata pair shaped like a registry environment entry."""
    return LineWorld, dict(LINEWORLD_METADATA)
