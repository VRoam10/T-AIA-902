"""Tests for PipelineRunner train/evaluate tqdm postfix logic.

Covers the live "checkpoints passed" counter for envs that report
``waypoint_idx`` in their step info, and the absence of the key for
other envs (e.g. Taxi).
"""

import io
from contextlib import redirect_stderr

import numpy as np

from core.runner import PipelineRunner


class FakeAgent:
    """Minimal agent with the BaseAgent subset the runner touches."""

    def __init__(self):
        self.epsilon = 1.0
        self.episode = 0

    def select_action(self, state):
        return 0

    def update(self, s, a, r, ns, done):
        return 0.01

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)


class FakeCheckpointsEnv:
    """Gymnasium-like env that reports waypoint_idx in info."""

    def __init__(self, max_steps=6):
        self.max_steps = max_steps
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return np.array([0.5, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        reward = 1.0
        done = self.t >= self.max_steps
        info = {"steps": self.t, "waypoint_idx": self.t // 2}
        return np.array([0.5, 0.0], dtype=np.float32), reward, done, info


class FakeTaxiEnv:
    """Gymnasium-like env without waypoint_idx (like Taxi)."""

    def __init__(self, max_steps=4):
        self.max_steps = max_steps
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return 0, {}

    def step(self, action):
        self.t += 1
        reward = 1.0
        done = self.t >= self.max_steps
        return 0, reward, done, {}


class TestRunnerPostfix:
    def test_train_postfix_contains_checkpoints_for_waypoint_env(self):
        agent = FakeAgent()
        env = FakeCheckpointsEnv(max_steps=6)
        runner = PipelineRunner()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            runner.train(agent, env, n_episodes=1)
        text = stderr.getvalue()
        # checkpoints should appear in postfix at some point
        assert "checkpoints" in text

    def test_train_postfix_contains_reward_and_checkpoints_in_final_frame(self):
        agent = FakeAgent()
        env = FakeCheckpointsEnv(max_steps=6)
        runner = PipelineRunner()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            runner.train(agent, env, n_episodes=1)
        text = stderr.getvalue()
        # Look for the final set_postfix call that contains reward=
        # The final frame has both reward= and checkpoints=
        # Use the last occurrence of "reward=" to find the final frame
        last_reward_idx = text.rfind("reward=")
        assert last_reward_idx != -1
        tail = text[last_reward_idx:]
        assert "checkpoints=" in tail

    def test_train_postfix_updates_live_mid_episode(self):
        agent = FakeAgent()
        env = FakeCheckpointsEnv(max_steps=8)
        runner = PipelineRunner()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            runner.train(agent, env, n_episodes=1)
        text = stderr.getvalue()
        # Should see checkpoints change value during the episode
        # (0 -> 1 -> 2 -> 3 as t goes 0..7)
        assert "checkpoints=0" in text or "checkpoints=1" in text or "checkpoints=2" in text

    def test_train_postfix_no_checkpoints_for_taxi(self):
        agent = FakeAgent()
        env = FakeTaxiEnv(max_steps=4)
        runner = PipelineRunner()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            runner.train(agent, env, n_episodes=1)
        text = stderr.getvalue()
        assert "checkpoints" not in text

    def test_evaluate_postfix_contains_checkpoints_for_waypoint_env(self):
        agent = FakeAgent()
        env = FakeCheckpointsEnv(max_steps=6)
        runner = PipelineRunner()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            runner.evaluate(agent, env, n_episodes=1)
        text = stderr.getvalue()
        assert "checkpoints" in text

    def test_evaluate_postfix_final_frame_has_reward_and_checkpoints(self):
        agent = FakeAgent()
        env = FakeCheckpointsEnv(max_steps=6)
        runner = PipelineRunner()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            runner.evaluate(agent, env, n_episodes=1)
        text = stderr.getvalue()
        last_reward_idx = text.rfind("reward=")
        assert last_reward_idx != -1
        tail = text[last_reward_idx:]
        assert "checkpoints=" in tail

    def test_evaluate_postfix_no_checkpoints_for_taxi(self):
        agent = FakeAgent()
        env = FakeTaxiEnv(max_steps=4)
        runner = PipelineRunner()
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            runner.evaluate(agent, env, n_episodes=1)
        text = stderr.getvalue()
        assert "checkpoints" not in text
