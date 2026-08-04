"""Tests for environments.beamng_race — the shared-track race env (no simulator)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from environments.beamng_race import BeamNGRaceEnv, build_race_slots

STRAIGHT = [(float(i) * 25.0, 0.0, 0.0) for i in range(1, 6)]  # 25..125 m


class _FakeAgent:
    def __init__(self, action=1):
        self.action = action
        self.epsilon = 0.7
        self.updates = 0
        self.saved_to = None

    def select_action(self, _state):
        return self.action

    def update(self, *_args):
        self.updates += 1
        return 0.1

    def decay_epsilon(self):
        self.epsilon *= 0.9

    def save(self, path):
        self.saved_to = path


def _path(spawn=(0.0, 0.0, 10.0)):
    return SimpleNamespace(
        sparse_waypoints=list(STRAIGHT),
        dense_waypoints=list(STRAIGHT),
        spawn_pos=spawn,
        spawn_rot=(0.0, 0.0, 0.0, 1.0),
    )


def _specs(n=2, human_last=False):
    specs = []
    for i in range(n):
        if human_last and i == n - 1:
            specs.append({"human": True, "color": "Blue"})
        else:
            specs.append(
                {
                    "algo": "dqn",
                    "agent": _FakeAgent(),
                    "color": ["Yellow", "Red", "Green"][i % 3],
                    "save_path": f"outputs/r{i}.pth",
                    "sensor": "lidar",
                }
            )
    return specs


def _env(n=2, human_last=False, **kwargs):
    env = BeamNGRaceEnv(
        slots=build_race_slots(_specs(n, human_last)), beamng_home="unused", **kwargs
    )
    env.trajectories = SimpleNamespace(paths=[_path()])
    return env


def _wire(slot, *, pos, waypoint_idx=0, damage=0.0, speed=10.0):
    slot.vehicle = MagicMock()
    slot.vehicle.state = {"pos": pos, "vel": (1.0, 0.0, 0.0), "dir": (1.0, 0.0, 0.0)}
    slot.electrics = MagicMock()
    slot.electrics.data = {"wheelspeed": speed, "steering": 0.0}
    slot.damage_sensor = MagicMock()
    slot.damage_sensor.data = {"damage": damage}
    slot.lidar = MagicMock()
    slot.lidar.poll.return_value = {"pointCloud": None}
    slot.ego_local_extents = None
    slot.waypoints = list(STRAIGHT)
    slot.waypoint_idx = waypoint_idx
    slot.current_pos = pos


class TestBuildRaceSlots:
    def test_agent_slots_are_sized_from_their_sensor(self):
        slots = build_race_slots(_specs(2))
        assert all(s.n_states == 14 for s in slots)
        assert all(s.output == "fixed" for s in slots)

    def test_human_slot_needs_no_algo_or_agent(self):
        slots = build_race_slots([{"human": True, "color": "Blue"}])
        assert slots[0].human is True
        assert slots[0].agent is None

    def test_names_distinguish_humans_from_racers(self):
        slots = build_race_slots(_specs(2, human_last=True))
        assert slots[0].name.startswith("racer")
        assert slots[1].name.startswith("human")


class TestLapsGuard:
    def test_more_than_one_lap_is_rejected(self):
        """Generated paths are open roads, so lap 2 would mean driving back to the
        start. Fail loudly rather than silently race one lap."""
        with pytest.raises(ValueError, match="laps=2 is not supported"):
            BeamNGRaceEnv(slots=[], beamng_home="unused", laps=2)

    def test_one_lap_is_accepted(self):
        assert BeamNGRaceEnv(slots=[], beamng_home="unused", laps=1).laps == 1


class TestSharedPathAndGrid:
    def test_every_entrant_races_the_same_waypoints(self):
        env = _env(3)
        env._assign_paths()
        first = env.slots[0].waypoints
        assert all(s.waypoints == first for s in env.slots)

    def test_spawns_are_offset_so_cars_do_not_overlap(self):
        env = _env(3)
        env._assign_paths()
        positions = [s.spawn_pos for s in env.slots]
        for i, a in enumerate(positions):
            for b in positions[i + 1 :]:
                assert np.hypot(a[0] - b[0], a[1] - b[1]) > 2.0

    def test_all_entrants_face_the_same_way(self):
        env = _env(2)
        env._assign_paths()
        assert env.slots[0].spawn_rot == env.slots[1].spawn_rot

    def test_more_entrants_than_paths_is_fine_unlike_training(self):
        # Training raises here; a race shares one path by design.
        env = _env(3)
        env.trajectories = SimpleNamespace(paths=[_path()])
        env._assign_paths()  # must not raise
        assert len({s.path_idx for s in env.slots}) == 1

    def test_no_drivable_path_is_an_error(self):
        env = _env(2)
        env.trajectories = SimpleNamespace(paths=[])
        with pytest.raises(ValueError, match="no drivable path"):
            env._assign_paths()

    def test_random_path_is_forced_off(self):
        # Racers must all run the same path, so per-episode randomisation is invalid.
        assert _env(2).random_path is False


class TestProgressAndStandings:
    def test_leader_is_the_car_furthest_along(self):
        env = _env(2)
        _wire(env.slots[0], pos=(30.0, 0.0, 0.0), waypoint_idx=1)
        _wire(env.slots[1], pos=(80.0, 0.0, 0.0), waypoint_idx=3)
        assert env.leader() is env.slots[1]

    def test_standings_are_ordered_leader_first(self):
        env = _env(2)
        _wire(env.slots[0], pos=(90.0, 0.0, 0.0), waypoint_idx=3)
        _wire(env.slots[1], pos=(20.0, 0.0, 0.0), waypoint_idx=0)
        names = [n for n, _ in env.standings()]
        assert names == [env.slots[0].name, env.slots[1].name]

    def test_leader_of_an_empty_field_is_none(self):
        env = BeamNGRaceEnv(slots=[], beamng_home="unused")
        assert env.leader() is None


class TestGapReward:
    def _two_car_env(self, mine, theirs):
        env = _env(2)
        _wire(env.slots[0], pos=(mine, 0.0, 0.0), waypoint_idx=1)
        _wire(env.slots[1], pos=(theirs, 0.0, 0.0), waypoint_idx=1)
        for s in env.slots:
            s.last_dist = s.current_dist = 10.0
        env.snapshot_progress()
        return env

    def test_pulling_away_scores_better_than_falling_back(self):
        obs = np.zeros(14, dtype=np.float32)
        obs[0] = 0.5
        obs[6:] = 1.0
        # Advancing 20 m past x=50 also clears the waypoint at x=75's predecessor,
        # so the index moves with the position — the invariant the sim maintains.
        moved_up = dict(pos=(70.0, 0.0, 0.0), waypoint_idx=2)

        ahead = self._two_car_env(50.0, 50.0)
        _wire(ahead.slots[0], **moved_up)  # I gained ground
        gained, _ = ahead.compute_race_reward_for(ahead.slots[0], obs)

        behind = self._two_car_env(50.0, 50.0)
        _wire(behind.slots[1], **moved_up)  # the rival gained ground
        lost, _ = behind.compute_race_reward_for(behind.slots[0], obs)

        assert gained > lost
        # And the difference is symmetric: +20 m of gap one way, -20 m the other.
        assert gained - lost == pytest.approx(2 * 20.0 * 5.0, abs=1.0)

    def test_a_lone_entrant_gets_no_gap_term(self):
        env = _env(1)
        _wire(env.slots[0], pos=(10.0, 0.0, 0.0), waypoint_idx=1)
        env.slots[0].last_dist = env.slots[0].current_dist = 10.0
        obs = np.zeros(14, dtype=np.float32)
        obs[0] = 0.5
        obs[6:] = 1.0
        reward, _ = env.compute_race_reward_for(env.slots[0], obs)
        # speed*3 - step penalty, with no gap contribution.
        assert reward == pytest.approx(0.5 * 3.0 - 0.5)

    def test_snapshot_uses_one_consistent_baseline_for_all_slots(self):
        env = self._two_car_env(40.0, 60.0)
        assert env.slots[0].last_rival_progress_m == pytest.approx(env.progress_of(env.slots[1]))
        assert env.slots[1].last_rival_progress_m == pytest.approx(env.progress_of(env.slots[0]))


class TestRaceOverAndResult:
    def test_race_is_over_once_someone_finishes(self):
        env = _env(2)
        assert env.race_over() is False
        env.slots[1].finished = True
        assert env.race_over() is True

    def test_race_is_over_when_everyone_is_done(self):
        env = _env(2)
        for s in env.slots:
            s.done = True
        assert env.race_over() is True

    def test_winner_is_the_finisher_not_merely_the_leader(self):
        env = _env(2)
        _wire(env.slots[0], pos=(10.0, 0.0, 0.0), waypoint_idx=0)
        _wire(env.slots[1], pos=(120.0, 0.0, 0.0), waypoint_idx=4)
        env.slots[0].finished = True
        assert env.winner() is env.slots[0]

    def test_winner_falls_back_to_the_leader_when_nobody_finished(self):
        env = _env(2)
        _wire(env.slots[0], pos=(10.0, 0.0, 0.0), waypoint_idx=0)
        _wire(env.slots[1], pos=(120.0, 0.0, 0.0), waypoint_idx=4)
        assert env.winner() is env.slots[1]

    def test_result_reports_margin_and_ordered_entrants(self):
        env = _env(2)
        _wire(env.slots[0], pos=(100.0, 0.0, 0.0), waypoint_idx=4)
        _wire(env.slots[1], pos=(60.0, 0.0, 0.0), waypoint_idx=2)
        result = env.result()
        assert result["winner"] == env.slots[0].name
        assert result["margin_m"] > 0
        assert [e["name"] for e in result["entrants"]] == [
            env.slots[0].name,
            env.slots[1].name,
        ]


class TestHumanEntrant:
    def test_agent_slots_exclude_the_human(self):
        env = _env(2, human_last=True)
        assert [s.name for s in env.agent_slots()] == [env.slots[0].name]
        assert [s.name for s in env.human_slots()] == [env.slots[1].name]

    def test_human_gets_no_perception_sensor(self):
        env = _env(2, human_last=True)
        env.bng = MagicMock()
        human = env.slots[1]
        human.vehicle = MagicMock()
        env._create_slot_sensor(human)
        assert human.lidar is None
        assert human.camera is None

    def test_observe_all_polls_the_human_but_returns_no_obs_for_them(self):
        """The human must still be polled: observe() is what advances their waypoint
        index, which their rival's gap term reads."""
        env = _env(2, human_last=True)
        _wire(env.slots[0], pos=(10.0, 0.0, 0.0))
        _wire(env.slots[1], pos=(20.0, 0.0, 0.0))
        obs = env.observe_all()
        assert set(obs) == {env.slots[0].name}
        env.slots[1].vehicle.poll_sensors.assert_called_once()

    def test_focus_is_best_effort_when_the_api_is_missing(self, capsys):
        env = _env(2, human_last=True)
        env.bng = SimpleNamespace()  # no switch_vehicle in any spelling
        env.slots[1].vehicle = MagicMock()
        env._focus_human()  # must not raise
        assert "switch_vehicle" in capsys.readouterr().out

    def test_focus_prefers_the_vehicles_namespace(self):
        env = _env(2, human_last=True)
        switcher = MagicMock()
        env.bng = SimpleNamespace(vehicles=SimpleNamespace(switch_vehicle=switcher))
        env.slots[1].vehicle = MagicMock()
        env._focus_human()
        switcher.assert_called_once_with(env.slots[1].vehicle)

    def test_focus_is_a_noop_without_a_human(self):
        env = _env(2)
        env.bng = SimpleNamespace()
        env._focus_human()  # must not raise


class TestMarkers:
    def test_per_slot_markers_are_suppressed(self):
        """Both cars aim at the same waypoint, so per-slot spheres would stack. The
        scenario's checkpoint rings already show the line."""
        env = _env(2)
        env.bng = MagicMock()
        env._update_slot_marker(env.slots[0])
        env.bng.debug.add_spheres.assert_not_called()


class TestAdvance:
    def test_lockstep_steps_physics(self):
        env = _env(2, realtime=False)
        env.bng = MagicMock()
        env.advance()
        env.bng.step.assert_called_once_with(10)

    def test_realtime_resumes_once_and_never_steps(self):
        env = _env(2, realtime=True)
        env.bng = MagicMock()
        env.advance()
        env.advance()
        env.bng.resume.assert_called_once()
        env.bng.step.assert_not_called()

    def test_realtime_resume_opens_the_road_gate(self):
        # Realtime races never call step_physics(), so nothing else would ever
        # reopen the gate (docs/romain.md, seventh issue) — resume() must.
        env = _env(2, realtime=True)
        env.bng = MagicMock()
        env._road_pollable = False
        env.advance()
        assert env._road_pollable is True

    def test_lockstep_advance_opens_the_road_gate(self):
        env = _env(2, realtime=False)
        env.bng = MagicMock()
        env._road_pollable = False
        env.advance()
        assert env._road_pollable is True
