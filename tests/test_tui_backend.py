"""Tests for the TUI backend bridge and trajectory cache helpers."""

import dataclasses
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

import core.pipeline_actions as pipeline_actions
from core.pipeline_actions import (
    BeamNGOptions,
    HumanPlayRequest,
    RacerSpec,
    TrajectoryRequest,
    run_human_play,
    run_trajectory,
    trajectory_cache_path,
)
from core.tui_backend import main
from environments import beamng_spec


def test_catalog_command_emits_expected_keys():
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["catalog"])
    assert rc == 0
    payload = json.loads(buf.getvalue())
    for key in ("algorithms", "environments", "benchmarks", "beamng_maps", "beamng_sensors"):
        assert key in payload
    assert payload["algorithms"]
    assert payload["beamng_maps"] == list(beamng_spec.AVAILABLE_MAPS)
    assert payload["beamng_sensors"] == list(beamng_spec.SENSORS)
    # One car now, so the catalog no longer offers a vehicle list.
    assert "beamng_vehicles" not in payload
    # One env, and Taxi is gone.
    assert [e["name"] for e in payload["environments"]] == ["beamng"]
    assert "q_learning" not in [a["name"] for a in payload["algorithms"]]


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
    run_human_play(
        HumanPlayRequest(
            map_name="italy",
            sensor="adv_lidar",
            random_path=True,
            road_info=True,
            wheel_info=True,
        )
    )
    assert captured["random_path"] is True
    assert captured["road_info"] is True
    assert captured["wheel_info"] is True


def test_run_human_play_defaults_random_path_off(monkeypatch):
    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(
        pipeline_actions.registry, "get_environment", lambda name: {"factory": fake_factory}
    )
    run_human_play(HumanPlayRequest(map_name="italy"))
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
        "env_name": "beamng",
        "beamng": {
            "map_name": "italy",
            "sensor": "adv_lidar",
            "trajectory_hints": 2,
            "body_orientation": True,
            "road_info": True,
            "random_path": True,
            "dense_episodes": 5,
        },
    }
    rc = main(["benchmark", "--config-json", json.dumps(payload)])
    assert rc == 0
    req = captured["request"]
    assert req.beamng is not None
    assert req.beamng.map_name == "italy"
    assert req.beamng.sensor == "adv_lidar"
    assert req.beamng.trajectory_hints == 2
    assert req.beamng.body_orientation is True
    assert req.beamng.road_info is True
    assert req.beamng.random_path is True
    assert req.beamng.dense_episodes == 5


# --------------------------------------------------------------------------- #
# TS <-> Python payload key contract
# --------------------------------------------------------------------------- #
_WORKFLOWS_TS = (Path(__file__).resolve().parents[1] / "tui" / "src" / "workflows.ts").read_text(
    encoding="utf-8"
)


def _keys_after(marker: str) -> set[str]:
    """Key names of the object literal that opens right after ``marker``.

    A regex over the literal's own ``key:`` lines, not a real TS parser -- this
    is a guard against a stray rename, not a general-purpose tool.
    """
    start = _WORKFLOWS_TS.index("{", _WORKFLOWS_TS.index(marker))
    end = _WORKFLOWS_TS.index("};", start)
    return set(re.findall(r"^\s*(\w+):", _WORKFLOWS_TS[start:end], re.MULTILINE))


def _keys_around(marker: str) -> set[str]:
    """Key names of the object literal enclosing the line containing ``marker``."""
    idx = _WORKFLOWS_TS.index(marker)
    start = _WORKFLOWS_TS.rindex("{", 0, idx)
    end = _WORKFLOWS_TS.index("};", idx)
    return set(re.findall(r"^\s*(\w+):", _WORKFLOWS_TS[start:end], re.MULTILINE))


def test_beamng_options_ts_keys_match_python_dataclass_fields():
    """Guard the boundary a Python-side rename actually breaks.

    ``core/tui_backend.py`` does ``BeamNGOptions(**raw)`` on whatever JSON the
    TUI sends, so a field renamed on either side is a ``TypeError`` at run
    launch -- not hypothetical: mid-plan the TypeScript kept sending
    ``wheel_terrain`` after Python dropped the field, and every TUI-launched
    BeamNG run raised until the UI task landed.
    ``tui/src/__tests__/workflows.test.ts`` already pins the TS-side key set
    against a literal, which catches a stray key added in TypeScript, but not a
    Python-side rename. This test closes the loop from the side that actually
    crashes, by comparing ``tui/src/workflows.ts``'s object-literal keys against
    ``dataclasses.fields`` on the Python side.
    """
    beamng_defaults_keys = _keys_after("BEAMNG_DEFAULTS: BeamNGFields = ")
    python_beamng_keys = {f.name for f in dataclasses.fields(BeamNGOptions)}

    # Legitimately Python-only: not part of BEAMNG_DEFAULTS because each is set
    # directly by its own form field rather than defaulted here (forms.ts sets
    # `random_path`/`dense_episodes`; buildTrainPayload also overrides
    # `random_path` explicitly before spreading `state.beamng` over it).
    beamng_python_only = {"random_path", "dense_episodes"}

    missing_in_ts = python_beamng_keys - beamng_defaults_keys - beamng_python_only
    extra_in_ts = beamng_defaults_keys - python_beamng_keys
    assert not missing_in_ts and not extra_in_ts, (
        "BeamNGOptions fields vs tui/src/workflows.ts BEAMNG_DEFAULTS keys diverged "
        f"(fix whichever side is wrong): in BeamNGOptions but missing from "
        f"BEAMNG_DEFAULTS and not exempted={sorted(missing_in_ts)}; in BEAMNG_DEFAULTS "
        f"but not a BeamNGOptions field={sorted(extra_in_ts)}"
    )


def test_racer_spec_ts_keys_match_python_dataclass_fields():
    """Same guard as above, for the course-racer payload / ``RacerSpec``.

    ``core/tui_backend.py``'s ``_cmd_course`` does ``RacerSpec(**raw)`` per
    racer, so this boundary can break the same way.
    """
    racer_keys = _keys_around("algo: r.algo,")
    python_racer_keys = {f.name for f in dataclasses.fields(RacerSpec)}

    # Legitimately Python-only: a human entrant takes buildCoursePayload's other
    # branch (`{ human: true, color }`), never the full racer object this reads.
    racer_python_only = {"human"}

    missing_in_ts = python_racer_keys - racer_keys - racer_python_only
    extra_in_ts = racer_keys - python_racer_keys
    assert not missing_in_ts and not extra_in_ts, (
        "RacerSpec fields vs tui/src/workflows.ts's course-racer payload keys "
        f"diverged (fix whichever side is wrong): in RacerSpec but missing from "
        f"the TS racer object and not exempted={sorted(missing_in_ts)}; in the TS "
        f"racer object but not a RacerSpec field={sorted(extra_in_ts)}"
    )
