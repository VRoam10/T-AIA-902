"""Tests for the TUI backend bridge and trajectory cache helpers."""

import io
import json
from contextlib import redirect_stdout

from core.pipeline_actions import TrajectoryRequest, run_trajectory, trajectory_cache_path
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
