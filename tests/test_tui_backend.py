"""Tests for the TUI backend bridge and trajectory cache helpers."""

import io
import json
from contextlib import redirect_stdout
from unittest.mock import MagicMock

import core.pipeline_actions as pipeline_actions
from core.pipeline_actions import (
    HumanPlayRequest,
    TrajectoryRequest,
    run_human_play,
    run_trajectory,
    trajectory_cache_path,
)
from core.tui_backend import main


def test_catalog_command_emits_expected_keys():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["catalog"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    for key in ("algorithms", "environments", "benchmarks", "beamng_maps", "beamng_vehicles"):
        assert key in payload
    assert payload["algorithms"]
    assert payload["beamng_maps"] == ["gridmap_v2", "italy", "west_coast_usa"]
    assert all("id" in v and "label" in v for v in payload["beamng_vehicles"])


def test_trajectory_cache_path_builds_under_output_dir(tmp_path):
    path = trajectory_cache_path("gridmap_v2", tmp_path)
    assert path == str(tmp_path / "gridmap_v2.json")


def test_run_trajectory_skips_when_cache_exists(tmp_path):
    cache = tmp_path / "gridmap_v2.json"
    cache.write_text("{}")
    result = run_trajectory(TrajectoryRequest("gridmap_v2", overwrite=False), output_dir=tmp_path)
    assert result["status"] == "skipped"
    assert result["path"] == str(cache)
    # Cache untouched.
    assert cache.exists()


def test_run_human_play_forwards_random_path(monkeypatch):
    # The randomize-path option must reach the env so human play deals a new
    # random path on crash / completion.
    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        pipeline_actions.registry, "get_environment", lambda name: {"factory": fake_factory}
    )
    run_human_play(HumanPlayRequest(map_name="italy", vehicle_id="taxi", random_path=True))
    assert captured["random_path"] is True


def test_run_human_play_defaults_random_path_off(monkeypatch):
    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        pipeline_actions.registry, "get_environment", lambda name: {"factory": fake_factory}
    )
    run_human_play(HumanPlayRequest(map_name="italy", vehicle_id="taxi"))
    assert captured["random_path"] is False


def test_cmd_benchmark_parses_beamng_into_request(monkeypatch):
    captured = {}
    import core.tui_backend as tui_mod

    def fake_run_benchmark(req):
        captured["request"] = req
        return {"status": "ok"}

    monkeypatch.setattr(tui_mod, "run_benchmark", fake_run_benchmark)
    payload = {
        "benchmark_name": "convergence",
        "seeds": [0],
        "eval_episodes": 2,
        "success_threshold": 0,
        "max_episodes": 3,
        "reward_threshold": 7.0,
        "algo_name": "dqn",
        "env_name": "beamng_lidar",
        "beamng": {
            "map_name": "italy",
            "vehicle_id": "super",
            "trajectory_hints": 2,
            "body_orientation": True,
            "wheel_terrain": True,
            "random_path": True,
            "dense_episodes": 5,
        },
    }
    rc = main(["benchmark", "--config-json", json.dumps(payload)])
    assert rc == 0
    req = captured["request"]
    assert req.beamng is not None
    assert req.beamng.map_name == "italy"
    assert req.beamng.vehicle_id == "super"
    assert req.beamng.trajectory_hints == 2
    assert req.beamng.body_orientation is True
    assert req.beamng.wheel_terrain is True
    assert req.beamng.random_path is True
    assert req.beamng.dense_episodes == 5
