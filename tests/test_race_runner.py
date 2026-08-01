"""Tests for core.race_runner — the exhibition / race-training loop (no simulator).

A fake race env stands in for BeamNGRaceEnv so the loop's control flow is testable:
one advance per tick for the whole field, rewards then a single gap snapshot, the
human never receiving controls, and exhibition mode leaving the policies untouched.
"""

import numpy as np

from core.race_runner import RaceRunner


class _FakeAgent:
    def __init__(self, action=1):
        self.action = action
        self.epsilon = 0.7
        self.updates = 0
        self.decays = 0
        self.saved_to = None
        self.episode = 0

    def select_action(self, _state):
        return self.action

    def update(self, *_args):
        self.updates += 1
        return 0.25

    def decay_epsilon(self):
        self.decays += 1

    def save(self, path):
        self.saved_to = path


class _FakeSlot:
    def __init__(self, name, *, human=False, save_path=""):
        self.name = name
        self.human = human
        self.agent = None if human else _FakeAgent()
        self.save_path = save_path
        self.last_obs = np.zeros(14, dtype=np.float32)
        self.waypoints = [(0.0, 0.0, 0.0)] * 5
        self.waypoint_idx = 0
        self.steps = 0
        self.done = False
        self.finished = False
        self.ep_reward = 0.0
        self.ep_losses = []
        self.ep_speeds = []
        self.episode = 0
        self.reward_history = []
        self.steps_history = []


class _FakeRaceEnv:
    """Minimal stand-in: finishes slot 0 after `finish_after` ticks."""

    MAX_STEPS = 20

    def __init__(self, *, human=False, finish_after=3, realtime=False, laps=1):
        self.slots = [_FakeSlot("racer_0", save_path="outputs/a.pth")]
        if human:
            self.slots.append(_FakeSlot("human_1", human=True))
        else:
            self.slots.append(_FakeSlot("racer_1", save_path="outputs/b.pth"))
        self.realtime = realtime
        self.laps = laps
        self.finish_after = finish_after
        self.advances = 0
        self.snapshots = 0
        self.resets = 0
        self.applied = []
        self.tick = 0

    # --- the surface RaceRunner uses -------------------------------------
    def agent_slots(self):
        return [s for s in self.slots if not s.human]

    def human_slots(self):
        return [s for s in self.slots if s.human]

    def reset_race(self):
        self.resets += 1
        self.tick = 0
        for s in self.slots:
            s.done = s.finished = False
            s.steps = 0
            s.waypoint_idx = 0
            s.ep_reward = 0.0

    def apply_action(self, slot, action):
        self.applied.append((slot.name, action))

    def advance(self):
        self.advances += 1
        self.tick += 1

    def observe_all(self):
        return {s.name: np.zeros(14, dtype=np.float32) for s in self.agent_slots()}

    def compute_race_reward_for(self, slot, _obs):
        if slot is self.slots[0] and self.tick >= self.finish_after:
            slot.finished = True
            return 100.0, True
        return 1.0, False

    def snapshot_progress(self):
        self.snapshots += 1

    def race_over(self):
        return any(s.finished for s in self.slots) or all(s.done for s in self.slots)

    def result(self):
        return {"winner": self.slots[0].name, "margin_m": 12.5, "entrants": []}


class TestRaceLoop:
    def test_runs_the_requested_number_of_races(self):
        env = _FakeRaceEnv()
        out = RaceRunner().run(env, races=3)
        assert out["races"] == 3
        assert env.resets == 3

    def test_one_advance_per_tick_for_the_whole_field(self):
        """Contact must be symmetric: both cars move on the same advance."""
        env = _FakeRaceEnv(finish_after=4)
        RaceRunner().run(env, races=1)
        assert env.advances == 4

    def test_gap_baseline_is_snapshotted_exactly_once_per_tick(self):
        """Once per tick, after all rewards — snapshotting per slot would let one
        car's update move another car's telescoping baseline mid-tick."""
        env = _FakeRaceEnv(finish_after=4)
        RaceRunner().run(env, races=1)
        assert env.snapshots == env.advances

    def test_stops_when_someone_finishes(self):
        env = _FakeRaceEnv(finish_after=2)
        out = RaceRunner().run(env, races=1)
        assert out["results"][0]["winner"] == "racer_0"

    def test_records_the_winner_tally_across_races(self):
        env = _FakeRaceEnv()
        out = RaceRunner().run(env, races=2)
        assert out["wins"] == {"racer_0": 2}

    def test_a_stalled_field_times_out_instead_of_looping_forever(self):
        env = _FakeRaceEnv(finish_after=10_000)  # nobody ever finishes
        out = RaceRunner().run(env, races=1)
        assert out["results"][0]["timed_out"] is True
        assert env.advances <= env.MAX_STEPS * RaceRunner.MAX_TICKS_PER_RACE_FACTOR


class TestHumanEntrant:
    def test_the_human_never_receives_controls(self):
        env = _FakeRaceEnv(human=True, finish_after=3)
        RaceRunner().run(env, races=1)
        assert all(name != "human_1" for name, _ in env.applied)

    def test_the_humans_steps_are_still_counted(self):
        env = _FakeRaceEnv(human=True, finish_after=3)
        RaceRunner().run(env, races=1)
        assert env.slots[1].steps == env.advances

    def test_the_human_can_finish_by_clearing_the_checkpoints(self):
        env = _FakeRaceEnv(human=True, finish_after=10_000)
        human = env.slots[1]

        original = env.advance

        def advance_and_progress():
            original()
            human.waypoint_idx = len(human.waypoints)  # player crosses the line

        env.advance = advance_and_progress
        out = RaceRunner().run(env, races=1)
        assert human.finished is True
        assert out["results"][0]["timed_out"] is False


class TestExhibitionMode:
    def test_learning_off_does_not_update_the_agents(self):
        env = _FakeRaceEnv()
        RaceRunner().run(env, races=1, learning=False)
        assert all(s.agent.updates == 0 for s in env.agent_slots())

    def test_learning_off_does_not_save_checkpoints(self):
        env = _FakeRaceEnv()
        RaceRunner().run(env, races=1, learning=False)
        assert all(s.agent.saved_to is None for s in env.agent_slots())

    def test_exploration_noise_is_zeroed_during_the_race(self):
        """An exhibition must show the learned policy, not one still exploring."""
        env = _FakeRaceEnv()
        seen = []
        agent = env.slots[0].agent
        original = agent.select_action

        def record(state):
            seen.append(agent.epsilon)
            return original(state)

        agent.select_action = record
        RaceRunner().run(env, races=1, learning=False)
        assert seen and all(e == 0.0 for e in seen)

    def test_exploration_noise_is_restored_afterwards(self):
        env = _FakeRaceEnv()
        before = env.slots[0].agent.epsilon
        RaceRunner().run(env, races=1, learning=False)
        assert env.slots[0].agent.epsilon == before


class TestRaceTrainingMode:
    def test_learning_on_updates_every_driven_agent(self):
        env = _FakeRaceEnv(finish_after=3)
        RaceRunner().run(env, races=1, learning=True)
        assert all(s.agent.updates > 0 for s in env.agent_slots())

    def test_learning_on_saves_checkpoints(self):
        env = _FakeRaceEnv()
        RaceRunner().run(env, races=1, learning=True)
        assert env.slots[0].agent.saved_to == "outputs/a.pth"

    def test_learning_on_decays_epsilon_once_per_race(self):
        env = _FakeRaceEnv()
        RaceRunner().run(env, races=2, learning=True)
        assert env.slots[0].agent.decays == 2

    def test_learning_on_keeps_exploration_noise(self):
        env = _FakeRaceEnv()
        agent = env.slots[0].agent
        RaceRunner().run(env, races=1, learning=True)
        assert agent.epsilon > 0.0

    def test_reward_history_grows_per_race(self):
        env = _FakeRaceEnv()
        RaceRunner().run(env, races=2, learning=True)
        assert len(env.slots[0].reward_history) == 2


class TestPacing:
    def test_realtime_paces_the_loop(self):
        env = _FakeRaceEnv(finish_after=2, realtime=True)
        runner = RaceRunner()
        runner.REALTIME_TICK_S = 0.01
        RaceRunner.run(runner, env, races=1)
        assert env.advances == 2

    def test_pace_never_sleeps_negative(self):
        import time

        runner = RaceRunner()
        runner.REALTIME_TICK_S = 0.0
        start = time.time()
        runner._pace(start - 5.0)  # tick already overran its budget
        assert time.time() - start < 0.5
