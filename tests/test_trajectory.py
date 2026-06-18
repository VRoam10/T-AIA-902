"""Unit tests for core.trajectory."""

import json
import math
from unittest.mock import MagicMock

import pytest

from core.trajectory import (
    MIN_PATH_SEPARATION_M,
    MapTrajectories,
    TrajectoryData,
    _edge_center,
    _extract_longest_road,
    _nearest_road,
    _quat_to_forward,
    _road_centerlines,
    _road_path_from_teleport,
    _square_loop_fallback,
    _teleport_points,
    generate,
    heading_to_quat,
    load_or_generate,
    resample,
)


def test_trajectorydata_json_roundtrip():
    data = TrajectoryData(
        spawn_pos=(1.0, 2.0, 3.0),
        spawn_rot=(0.0, 0.0, 0.707, 0.707),
        sparse_waypoints=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        dense_waypoints=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        map_name="gridmap_v2",
        generated_at="2026-05-23T12:00:00Z",
        source="road_network:road_42",
    )

    payload = data.to_json()
    parsed = json.loads(payload)
    assert parsed["map_name"] == "gridmap_v2"
    assert parsed["spawn_pos"] == [1.0, 2.0, 3.0]
    assert parsed["sparse_waypoints"][1] == [10.0, 0.0, 0.0]

    restored = TrajectoryData.from_json(payload)
    assert restored == data


def _sample_traj(map_name="italy", source="road_network:r1"):
    return TrajectoryData(
        spawn_pos=(1.0, 2.0, 3.0),
        spawn_rot=(0.0, 0.0, 0.707, 0.707),
        sparse_waypoints=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        dense_waypoints=[(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        map_name=map_name,
        generated_at="2026-06-18T12:00:00+00:00",
        source=source,
    )


def test_maptrajectories_json_roundtrip():
    mt = MapTrajectories(
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        paths=[_sample_traj(source="teleport:r1"), _sample_traj(source="teleport:r2")],
    )
    restored = MapTrajectories.from_json(mt.to_json())
    assert restored == mt
    assert len(restored.paths) == 2
    assert restored.paths[1].source == "teleport:r2"


def test_maptrajectories_from_json_accepts_old_single_object_format():
    # Old caches stored a single TrajectoryData object at the top level.
    old_payload = _sample_traj(map_name="gridmap_v2").to_json()
    mt = MapTrajectories.from_json(old_payload)
    assert mt.map_name == "gridmap_v2"
    assert len(mt.paths) == 1
    assert mt.paths[0] == _sample_traj(map_name="gridmap_v2")


def test_resample_straight_line_uniform_spacing():
    # A 30 m straight line on the X axis, resampled at 10 m → 4 points (0, 10, 20, 30)
    path = [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)]
    out = resample(path, spacing=10.0)
    assert len(out) == 4
    assert out[0] == (0.0, 0.0, 0.0)
    assert out[-1] == (30.0, 0.0, 0.0)
    for i in range(len(out) - 1):
        d = math.hypot(out[i + 1][0] - out[i][0], out[i + 1][1] - out[i][1])
        assert d == pytest.approx(10.0, abs=1e-6)


def test_resample_preserves_endpoints_with_remainder():
    # 25 m line, spacing 10 m → samples at 0, 10, 20, and 25 (last point preserved)
    path = [(0.0, 0.0, 0.0), (25.0, 0.0, 0.0)]
    out = resample(path, spacing=10.0)
    assert out[0] == (0.0, 0.0, 0.0)
    assert out[-1] == (25.0, 0.0, 0.0)
    # Inner points spaced 10 m
    assert out[1] == pytest.approx((10.0, 0.0, 0.0), abs=1e-6)
    assert out[2] == pytest.approx((20.0, 0.0, 0.0), abs=1e-6)


def test_resample_two_segment_polyline():
    # L-shape: (0,0)→(10,0)→(10,10), 20 m total, spacing 5 m → 5 points
    path = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)]
    out = resample(path, spacing=5.0)
    assert len(out) == 5
    assert out[0] == (0.0, 0.0, 0.0)
    assert out[2] == pytest.approx((10.0, 0.0, 0.0), abs=1e-6)
    assert out[-1] == (10.0, 10.0, 0.0)


def test_resample_rejects_short_path():
    with pytest.raises(ValueError):
        resample([(0.0, 0.0, 0.0)], spacing=5.0)


def test_heading_to_quat_north_is_identity():
    # +Y direction = North in BeamNG = identity quaternion (0,0,0,1)
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (0.0, 10.0, 0.0))
    assert (qx, qy) == (0.0, 0.0)
    assert qz == pytest.approx(0.0, abs=1e-6)
    assert qw == pytest.approx(1.0, abs=1e-6)


def test_heading_to_quat_east():
    # +X direction = East. BeamNGpy yaw is CCW around +Z, so East = yaw -π/2.
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert (qx, qy) == (0.0, 0.0)
    assert qz == pytest.approx(-math.sin(math.pi / 4), abs=1e-6)
    assert qw == pytest.approx(math.cos(math.pi / 4), abs=1e-6)


def test_heading_to_quat_south():
    # -Y direction = South. atan2(-0.0, -10.0) = -π so (qz, qw) = (-1, 0),
    # which represents the same physical rotation as (1, 0) (q and -q are
    # equivalent rotations).
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (0.0, -10.0, 0.0))
    assert (qx, qy) == (0.0, 0.0)
    assert abs(qz) == pytest.approx(1.0, abs=1e-6)
    assert qw == pytest.approx(0.0, abs=1e-6)


def test_heading_to_quat_west():
    # -X direction = West. BeamNGpy yaw is CCW around +Z, so West = yaw +π/2.
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (-10.0, 0.0, 0.0))
    assert (qx, qy) == (0.0, 0.0)
    assert qz == pytest.approx(math.sin(math.pi / 4), abs=1e-6)
    assert qw == pytest.approx(math.cos(math.pi / 4), abs=1e-6)


def test_heading_to_quat_rejects_zero_delta():
    with pytest.raises(ValueError):
        heading_to_quat((1.0, 1.0, 0.0), (1.0, 1.0, 0.0))


def test_square_loop_fallback_topology():
    traj = _square_loop_fallback(map_name="smallgrid")
    # 80 m square, perimeter 320 m
    # Sparse 25 m → ~13 samples; dense 8 m → ~40 samples
    assert traj.map_name == "smallgrid"
    assert traj.source == "fallback:square_loop"
    assert len(traj.sparse_waypoints) >= 11
    assert len(traj.sparse_waypoints) <= 15
    assert len(traj.dense_waypoints) >= 35
    # Spawn coincides with the first square corner (40, -40)
    assert traj.spawn_pos[:2] == (40.0, -40.0)
    # Spawn is above the road (z offset)
    assert traj.spawn_pos[2] > 1.0
    # First waypoint is now the next sample along the loop, NOT the spawn
    assert traj.sparse_waypoints[0][:2] != traj.spawn_pos[:2]


def test_square_loop_corners_are_at_expected_positions():
    traj = _square_loop_fallback(map_name="smallgrid")
    # Corners of an 80 m square around origin → expect points near (40,-40), (40,40), (-40,40), (-40,-40)
    xy = [(p[0], p[1]) for p in traj.sparse_waypoints]
    # The (40,-40) corner is the spawn point itself and is no longer a
    # waypoint; check only the 3 remaining corners.
    for cx, cy in [(40.0, 40.0), (-40.0, 40.0), (-40.0, -40.0)]:
        assert any(math.hypot(x - cx, y - cy) < 1e-3 for x, y in xy), (
            f"missing corner near ({cx}, {cy})"
        )


def test_edge_center_prefers_middle_key():
    edge = {"middle": (5.0, 5.0, 1.0), "left": (0.0, 0.0, 1.0), "right": (10.0, 10.0, 1.0)}
    assert _edge_center(edge) == (5.0, 5.0, 1.0)


def test_edge_center_falls_back_to_left_right_midpoint():
    edge = {"left": (0.0, 0.0, 1.0), "right": (10.0, 4.0, 1.0)}
    assert _edge_center(edge) == (5.0, 2.0, 1.0)


def test_extract_longest_road_picks_longest():
    network = {
        "short_road": {
            "edges": [
                {"middle": (0.0, 0.0, 0.0)},
                {"middle": (5.0, 0.0, 0.0)},
            ],
        },
        "long_road": {
            "edges": [
                {"middle": (0.0, 0.0, 0.0)},
                {"middle": (50.0, 0.0, 0.0)},
                {"middle": (100.0, 0.0, 0.0)},
            ],
        },
    }
    road_id, centerline = _extract_longest_road(network)
    assert road_id == "long_road"
    assert centerline[0] == (0.0, 0.0, 0.0)
    assert centerline[-1] == (100.0, 0.0, 0.0)


def test_extract_longest_road_returns_none_for_empty_network():
    assert _extract_longest_road({}) == (None, None)


def test_extract_longest_road_skips_single_edge_roads():
    network = {"degenerate": {"edges": [{"middle": (0.0, 0.0, 0.0)}]}}
    assert _extract_longest_road(network) == (None, None)


def _spawn_obj(pos, rot=(0.0, 0.0, 0.0, 1.0)):
    obj = MagicMock()
    obj.pos = pos
    obj.rot_quat = rot
    return obj


def _two_road_network():
    return {
        "north": {
            "edges": [
                {"middle": (0.0, 0.0, 0.0)},
                {"middle": (0.0, 50.0, 0.0)},
                {"middle": (0.0, 120.0, 0.0)},
            ],
        },
        "east": {
            "edges": [
                {"middle": (200.0, 0.0, 0.0)},
                {"middle": (250.0, 0.0, 0.0)},
            ],
        },
    }


def test_teleport_points_reads_spawnspheres():
    bng = MagicMock()
    bng.scenario.find_objects_class.return_value = [
        _spawn_obj((1.0, 2.0, 3.0)),
        _spawn_obj((4.0, 5.0, 6.0)),
    ]
    pts = _teleport_points(bng)
    bng.scenario.find_objects_class.assert_called_once_with("SpawnSphere")
    assert pts[0][0] == (1.0, 2.0, 3.0)
    assert len(pts) == 2


def test_teleport_points_empty_on_error():
    bng = MagicMock()
    bng.scenario.find_objects_class.side_effect = RuntimeError("boom")
    bng.scenario.find_waypoints.side_effect = RuntimeError("boom")
    assert _teleport_points(bng) == []


def test_generate_builds_one_path_per_teleport():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_objects_class.return_value = [
        _spawn_obj((0.0, 0.0, 0.0)),     # snaps to "north"
        _spawn_obj((201.0, 0.0, 0.0)),   # snaps to "east"
    ]
    mt = generate(bng, map_name="italy")
    assert isinstance(mt, MapTrajectories)
    assert mt.map_name == "italy"
    assert len(mt.paths) == 2
    # Sorted longest-road-first: the "north" road (120 m) precedes "east" (50 m).
    assert mt.paths[0].source == "teleport:north"
    assert mt.paths[1].source == "teleport:east"
    assert all(len(p.sparse_waypoints) >= 1 for p in mt.paths)


def test_generate_dedupes_nearby_teleports():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    # Second spawn is 1 m away — well within MIN_PATH_SEPARATION_M (30 m).
    assert MIN_PATH_SEPARATION_M > 1.0
    bng.scenario.find_objects_class.return_value = [
        _spawn_obj((0.0, 0.0, 0.0)),
        _spawn_obj((1.0, 0.0, 0.0)),  # within MIN_PATH_SEPARATION_M of the first
    ]
    mt = generate(bng, map_name="italy")
    assert len(mt.paths) == 1


def test_generate_falls_back_to_longest_road_without_teleports():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_objects_class.return_value = []
    bng.scenario.find_waypoints.return_value = []
    mt = generate(bng, map_name="italy")
    assert len(mt.paths) == 1
    assert mt.paths[0].source.startswith("road_network:")


def test_generate_falls_back_to_square_loop_without_roads():
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = {}
    bng.scenario.find_objects_class.return_value = []
    bng.scenario.find_waypoints.return_value = []
    mt = generate(bng, map_name="smallgrid")
    assert len(mt.paths) == 1
    assert mt.paths[0].source == "fallback:square_loop"


def test_load_or_generate_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    mt = MapTrajectories(
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        paths=[_sample_traj(source="teleport:cached")],
    )
    (tmp_path / "italy.json").write_text(mt.to_json())
    bng = MagicMock()
    out = load_or_generate("italy", bng)
    bng.scenario.get_road_network.assert_not_called()
    assert out == mt


def test_load_or_generate_reads_old_single_object_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    (tmp_path / "italy.json").write_text(_sample_traj().to_json())  # old format
    out = load_or_generate("italy", MagicMock())
    assert isinstance(out, MapTrajectories)
    assert len(out.paths) == 1


def test_load_or_generate_generates_and_writes_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_objects_class.return_value = [_spawn_obj((0.0, 0.0, 0.0))]
    out = load_or_generate("italy", bng)
    assert (tmp_path / "italy.json").exists()
    assert MapTrajectories.from_json((tmp_path / "italy.json").read_text()) == out


def test_load_or_generate_raises_when_no_cache_and_no_bng(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="No cached trajectory"):
        load_or_generate("italy", bng=None)


def test_load_or_generate_regenerates_on_corrupt_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    (tmp_path / "italy.json").write_text("{not valid json")
    bng = MagicMock()
    bng.scenario.get_road_network.return_value = _two_road_network()
    bng.scenario.find_objects_class.return_value = [_spawn_obj((0.0, 0.0, 0.0))]
    out = load_or_generate("italy", bng)
    assert out.map_name == "italy"
    MapTrajectories.from_json((tmp_path / "italy.json").read_text())


def test_road_centerlines_lists_all_multi_edge_roads():
    network = {
        "a": {"edges": [{"middle": (0.0, 0.0, 0.0)}, {"middle": (10.0, 0.0, 0.0)}]},
        "b": {"edges": [{"middle": (0.0, 0.0, 0.0)}]},  # single edge -> skipped
        "c": {"edges": [{"middle": (0.0, 5.0, 0.0)}, {"middle": (0.0, 15.0, 0.0)}]},
    }
    roads = _road_centerlines(network)
    ids = {rid for rid, _ in roads}
    assert ids == {"a", "c"}


def test_quat_to_forward_identity_is_north():
    fx, fy = _quat_to_forward((0.0, 0.0, 0.0, 1.0))
    assert fx == pytest.approx(0.0, abs=1e-6)
    assert fy == pytest.approx(1.0, abs=1e-6)


def test_quat_to_forward_east():
    # yaw -pi/2 faces +X (East): forward = (1, 0)
    rot = (0.0, 0.0, math.sin(-math.pi / 4), math.cos(-math.pi / 4))
    fx, fy = _quat_to_forward(rot)
    assert fx == pytest.approx(1.0, abs=1e-6)
    assert fy == pytest.approx(0.0, abs=1e-6)


def test_nearest_road_picks_closest():
    roads = [
        ("far", [(100.0, 100.0, 0.0), (200.0, 100.0, 0.0)]),
        ("near", [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]),
    ]
    rid, centerline = _nearest_road((1.0, 1.0, 0.0), roads)
    assert rid == "near"
    assert centerline[0] == (0.0, 0.0, 0.0)


def test_nearest_road_none_when_empty():
    assert _nearest_road((0.0, 0.0, 0.0), []) is None


def test_road_path_from_teleport_walks_in_heading_direction():
    # Straight east-west road; teleport mid-road facing East -> path heads +X.
    centerline = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0)]
    path = _road_path_from_teleport(centerline, (10.0, 0.0, 0.0), (1.0, 0.0))
    assert path[0] == (10.0, 0.0, 0.0)
    assert path[-1] == (30.0, 0.0, 0.0)


def test_road_path_from_teleport_reverses_when_facing_back():
    centerline = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (30.0, 0.0, 0.0)]
    # Same snap vertex but facing West -> path heads -X.
    path = _road_path_from_teleport(centerline, (20.0, 0.0, 0.0), (-1.0, 0.0))
    assert path[0] == (20.0, 0.0, 0.0)
    assert path[-1] == (0.0, 0.0, 0.0)


def test_single_env_resolve_trajectory_takes_first_path(tmp_path, monkeypatch):
    monkeypatch.setattr("core.trajectory.CACHE_DIR", tmp_path)
    mt = MapTrajectories(
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        paths=[_sample_traj(source="teleport:first"), _sample_traj(source="teleport:second")],
    )
    (tmp_path / "italy.json").write_text(mt.to_json())

    from environments.beamng import BeamNGDrivingEnv

    env = BeamNGDrivingEnv(beamng_home="unused", map_name="italy")
    traj = env._resolve_trajectory()
    assert traj.source == "teleport:first"
