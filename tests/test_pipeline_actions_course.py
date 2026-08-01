"""Tests for the course-mode action layer: validation, session building, run_course."""

import json
from unittest.mock import MagicMock, patch

import pytest

import core.pipeline_actions as pipeline_actions
from core.pipeline_actions import CourseRequest, RacerSpec, build_course_session, run_course


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """Two real checkpoints, shaped exactly as build_course_session will size them.

    They must be genuine torch saves: ``build_course_session`` calls ``agent.load``,
    so a placeholder file would fail on deserialisation rather than on anything the
    tests are about. Sized via the same spec helpers as production, so a sizing
    change breaks the assertion rather than silently loading a mismatched net.
    """
    import algorithms  # noqa: F401 — registers the algorithms
    from core.registry import registry
    from environments import beamng_spec

    tmp = tmp_path_factory.mktemp("checkpoints")
    paths = []
    for algo, sensor, name in (("dqn", "lidar", "a.pth"), ("td3", "adv_lidar", "b.pth")):
        info = registry.get_algorithm(algo)
        cfg = dict(info["default_config"])
        cfg["n_states"] = beamng_spec.obs_size(sensor)
        cfg["n_actions"] = beamng_spec.action_size(beamng_spec.output_for_algo(algo))
        cfg.pop("state_type", None)
        path = tmp / name
        info["class"](**cfg).save(str(path))
        paths.append(str(path))
    return tuple(paths)


def _request(checkpoints, **overrides):
    a, b = checkpoints
    req = dict(
        map_name="gridmap_v2",
        racers=[
            RacerSpec(algo="dqn", sensor="lidar", model_path=a, color="Red"),
            RacerSpec(algo="td3", sensor="adv_lidar", model_path=b, color="Blue"),
        ],
    )
    req.update(overrides)
    return CourseRequest(**req)


class TestValidation:
    def test_more_than_one_lap_is_rejected_with_a_reason(self, checkpoints):
        with pytest.raises(ValueError, match="laps=3 is not supported"):
            run_course(_request(checkpoints, laps=3))

    def test_a_solo_entrant_is_not_a_race(self, checkpoints):
        a, _ = checkpoints
        req = CourseRequest(
            map_name="gridmap_v2", racers=[RacerSpec(algo="dqn", model_path=a)]
        )
        with pytest.raises(ValueError, match="at least two entrants"):
            run_course(req)

    def test_two_humans_are_rejected(self, checkpoints):
        req = CourseRequest(
            map_name="gridmap_v2",
            racers=[RacerSpec(human=True), RacerSpec(human=True)],
        )
        with pytest.raises(ValueError, match="one keyboard"):
            run_course(req)

    def test_a_missing_checkpoint_is_reported(self, checkpoints):
        a, _ = checkpoints
        req = CourseRequest(
            map_name="gridmap_v2",
            racers=[
                RacerSpec(algo="dqn", model_path=a),
                RacerSpec(algo="td3", model_path="outputs/does-not-exist.pth"),
            ],
        )
        with pytest.raises(FileNotFoundError, match="does-not-exist"):
            run_course(req)

    def test_an_entrant_without_a_checkpoint_is_rejected(self, checkpoints):
        a, _ = checkpoints
        req = CourseRequest(
            map_name="gridmap_v2",
            racers=[RacerSpec(algo="dqn", model_path=a), RacerSpec(algo="td3")],
        )
        with pytest.raises(ValueError, match="needs a checkpoint"):
            run_course(req)

    def test_a_human_needs_no_algo_or_checkpoint(self, checkpoints):
        a, _ = checkpoints
        req = CourseRequest(
            map_name="gridmap_v2",
            racers=[RacerSpec(algo="dqn", model_path=a), RacerSpec(human=True)],
        )
        pipeline_actions._validate_course(req)  # must not raise


class TestBuildCourseSession:
    def _build(self, request):
        with patch("environments.beamng_race.BeamNGRaceEnv") as EnvCls:
            EnvCls.return_value = MagicMock()
            return build_course_session(request), EnvCls

    def test_builds_one_slot_per_entrant(self, checkpoints):
        (env, slots), _ = self._build(_request(checkpoints))
        assert len(slots) == 2

    def test_a_chosen_game_track_reaches_the_race_env(self, checkpoints):
        # Regression: the option existed at every layer but was dropped on the way
        # in, so picking a track raced the generated paths with nothing to show why.
        (_, _), EnvCls = self._build(_request(checkpoints, track="race_track"))
        assert EnvCls.call_args.kwargs["track"] == "race_track"

    def test_no_track_races_the_generated_paths(self, checkpoints):
        (_, _), EnvCls = self._build(_request(checkpoints))
        assert EnvCls.call_args.kwargs["track"] is None

    def test_each_agent_is_sized_to_its_own_sensor(self, checkpoints):
        (env, slots), _ = self._build(_request(checkpoints))
        assert slots[0].n_states == 14  # lidar
        assert slots[1].n_states == 38  # adv_lidar

    def test_output_is_derived_per_entrant(self, checkpoints):
        (env, slots), _ = self._build(_request(checkpoints))
        assert slots[0].output == "fixed"  # dqn
        assert slots[1].output == "continuous"  # td3

    def test_checkpoints_are_loaded_into_the_agents(self, checkpoints):
        a, b = checkpoints
        with patch("environments.beamng_race.BeamNGRaceEnv") as EnvCls:
            EnvCls.return_value = MagicMock()
            with patch.object(pipeline_actions.registry, "get_algorithm") as get_algo:
                agent = MagicMock()
                get_algo.return_value = {"class": MagicMock(return_value=agent), "default_config": {}}
                build_course_session(_request(checkpoints))
        assert agent.load.call_count == 2

    def test_race_training_writes_to_its_own_file(self, checkpoints):
        """An exhibition race must never overwrite the checkpoint handed to it."""
        a, b = checkpoints
        (env, slots), _ = self._build(_request(checkpoints))
        assert slots[0].save_path != a
        assert "races" in slots[0].save_path

    def test_a_human_field_forces_realtime(self, checkpoints):
        a, _ = checkpoints
        req = CourseRequest(
            map_name="gridmap_v2",
            racers=[RacerSpec(algo="dqn", model_path=a), RacerSpec(human=True)],
        )
        _, EnvCls = self._build(req)
        assert EnvCls.call_args.kwargs["realtime"] is True

    def test_an_all_agent_field_runs_lockstep(self, checkpoints):
        _, EnvCls = self._build(_request(checkpoints))
        assert EnvCls.call_args.kwargs["realtime"] is False

    def test_colours_default_to_distinct_values(self, checkpoints):
        a, b = checkpoints
        req = CourseRequest(
            map_name="gridmap_v2",
            racers=[
                RacerSpec(algo="dqn", sensor="lidar", model_path=a, color=""),
                RacerSpec(algo="td3", sensor="adv_lidar", model_path=b, color=""),
            ],
        )
        (env, slots), _ = self._build(req)
        assert slots[0].color != slots[1].color

    def test_map_and_path_reach_the_env(self, checkpoints):
        _, EnvCls = self._build(_request(checkpoints, path_idx=2))
        assert EnvCls.call_args.kwargs["map_name"] == "gridmap_v2"
        assert EnvCls.call_args.kwargs["path_idx"] == 2


class TestRunCourse:
    def test_returns_the_race_outcome_and_closes_the_env(self, checkpoints):
        fake_env = MagicMock()
        with (
            patch.object(pipeline_actions, "build_course_session") as build,
            patch("core.race_runner.RaceRunner.run") as run,
        ):
            slot_a, slot_b = MagicMock(), MagicMock()
            slot_a.name, slot_b.name = "racer_0", "racer_1"
            build.return_value = (fake_env, [slot_a, slot_b])
            run.return_value = {"races": 2, "results": [], "wins": {"racer_0": 2}}
            out = run_course(_request(checkpoints, races=2))

        assert out["status"] == "ok"
        assert out["wins"] == {"racer_0": 2}
        assert out["entrants"] == ["racer_0", "racer_1"]
        fake_env.close.assert_called_once()

    def test_closes_the_env_even_when_the_race_raises(self, checkpoints):
        fake_env = MagicMock()
        with (
            patch.object(pipeline_actions, "build_course_session") as build,
            patch("core.race_runner.RaceRunner.run", side_effect=RuntimeError("boom")),
        ):
            build.return_value = (fake_env, [])
            with pytest.raises(RuntimeError, match="boom"):
                run_course(_request(checkpoints))
        # A leaked simulator would block the next launch on the same port.
        fake_env.close.assert_called_once()

    def test_learning_flag_is_forwarded(self, checkpoints):
        fake_env = MagicMock()
        with (
            patch.object(pipeline_actions, "build_course_session") as build,
            patch("core.race_runner.RaceRunner.run") as run,
        ):
            build.return_value = (fake_env, [])
            run.return_value = {"races": 1, "results": [], "wins": {}}
            out = run_course(_request(checkpoints, learning=True))
        assert run.call_args.kwargs["learning"] is True
        assert out["learning"] is True


class TestBackendCommand:
    def test_course_payload_maps_onto_the_request(self, monkeypatch):
        import core.tui_backend as tui_mod

        captured = {}

        def fake_run_course(req):
            captured["request"] = req
            return {"status": "ok"}

        monkeypatch.setattr(tui_mod, "run_course", fake_run_course)
        payload = {
            "map_name": "italy",
            "races": 3,
            "learning": True,
            "racers": [
                {"algo": "dqn", "sensor": "adv_lidar", "model_path": "a.pth", "color": "Red"},
                {"human": True, "color": "Blue"},
            ],
        }
        assert tui_mod.main(["course", "--config-json", json.dumps(payload)]) == 0

        req = captured["request"]
        assert req.map_name == "italy"
        assert req.races == 3
        assert req.learning is True
        assert req.racers[0].sensor == "adv_lidar"
        assert req.racers[1].human is True

    def test_course_is_a_registered_command(self):
        import core.tui_backend as tui_mod

        assert "course" in tui_mod._COMMANDS
