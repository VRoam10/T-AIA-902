"""Tests for core.multi_runner — parallel loop with a fake env + agents."""

import numpy as np

from core.multi_runner import MultiAgentRunner
from environments.beamng_multi import VehicleSlot


class FakeAgent:
    def __init__(self, action=0):
        self._action = action
        self.epsilon = 1.0
        self.updates = 0
        self.decays = 0
        self.saved = 0

    def select_action(self, state):
        return self._action

    def update(self, s, a, r, ns, done):
        self.updates += 1
        return 0.1

    def decay_epsilon(self):
        self.decays += 1

    def save(self, path):
        self.saved += 1


class FakeEnv:
    """Each slot finishes its episode after `episode_len` ticks."""

    def __init__(self, n_slots=2, episode_len=3, n_states=4):
        self.n_states = n_states
        self.slots = []
        for i in range(n_slots):
            agent = FakeAgent(action=i)
            self.slots.append(
                VehicleSlot(
                    name=f"ego_{i}", color="Red", vehicle_id="taxi", agent=agent,
                    reward_mode="default", action_space="discrete",
                    save_path=f"outputs/a{i}.pth",
                )
            )
            self.slots[-1].last_obs = np.zeros(n_states, dtype=np.float32)
        self.episode_len = episode_len
        self.reset_all_calls = 0
        self.step_calls = 0
        self.reset_vehicle_calls = 0

    def reset_all(self):
        self.reset_all_calls += 1
        for s in self.slots:
            s.last_obs = np.zeros(self.n_states, dtype=np.float32)

    def observe(self, slot):
        return np.zeros(self.n_states, dtype=np.float32)

    def apply_action(self, slot, action):
        pass

    def step_physics(self):
        self.step_calls += 1

    def compute_reward(self, slot, obs):
        # FakeEnv tracks ticks separately so the runner remains the only writer
        # of slot.steps.
        slot._fake_ticks = getattr(slot, "_fake_ticks", 0) + 1
        done = slot._fake_ticks >= self.episode_len
        if done:
            slot._fake_ticks = 0
        return 1.0, done

    def reset_vehicle(self, slot):
        self.reset_vehicle_calls += 1
        slot.last_obs = np.zeros(self.n_states, dtype=np.float32)

    def close(self):
        pass


class TestMultiAgentRunner:
    def test_runs_until_each_agent_completes_n_episodes(self):
        env = FakeEnv(n_slots=2, episode_len=3)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=2, time_limit=None, save_every=999)
        for s in env.slots:
            assert s.episode == 2
            assert len(s.reward_history) == 2

    def test_steps_physics_once_per_tick(self):
        env = FakeEnv(n_slots=2, episode_len=3)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=1, time_limit=None, save_every=999)
        assert env.step_calls == 3

    def test_updates_every_agent_each_tick(self):
        env = FakeEnv(n_slots=2, episode_len=3)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=1, time_limit=None, save_every=999)
        for s in env.slots:
            assert s.agent.updates == 3
            assert s.agent.decays == 1

    def test_finished_vehicle_is_reset(self):
        env = FakeEnv(n_slots=1, episode_len=2)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=2, time_limit=None, save_every=999)
        assert env.reset_vehicle_calls >= 1

    def test_time_limit_zero_stops_immediately_after_reset(self):
        env = FakeEnv(n_slots=1, episode_len=3)
        runner = MultiAgentRunner()
        runner.train(env, n_episodes=100, time_limit=0.0, save_every=999)
        assert env.slots[0].episode == 0
