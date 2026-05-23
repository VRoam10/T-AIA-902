"""Unit tests for core.trajectory."""
import json

from core.trajectory import TrajectoryData


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


import math

import pytest

from core.trajectory import resample


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


from core.trajectory import heading_to_quat


def test_heading_to_quat_north_is_identity():
    # +Y direction = North in BeamNG = identity quaternion (0,0,0,1)
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (0.0, 10.0, 0.0))
    assert (qx, qy) == (0.0, 0.0)
    assert qz == pytest.approx(0.0, abs=1e-6)
    assert qw == pytest.approx(1.0, abs=1e-6)


def test_heading_to_quat_east():
    # +X direction = East = (0, 0, 0.707, 0.707)
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    assert (qx, qy) == (0.0, 0.0)
    assert qz == pytest.approx(math.sin(math.pi / 4), abs=1e-6)
    assert qw == pytest.approx(math.cos(math.pi / 4), abs=1e-6)


def test_heading_to_quat_south():
    # -Y direction = South = (0, 0, 1, 0)
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (0.0, -10.0, 0.0))
    assert (qx, qy) == (0.0, 0.0)
    assert qz == pytest.approx(1.0, abs=1e-6)
    assert qw == pytest.approx(0.0, abs=1e-6)


def test_heading_to_quat_west():
    # -X direction = West = (0, 0, -0.707, 0.707)
    qx, qy, qz, qw = heading_to_quat((0.0, 0.0, 0.0), (-10.0, 0.0, 0.0))
    assert (qx, qy) == (0.0, 0.0)
    assert qz == pytest.approx(-math.sin(math.pi / 4), abs=1e-6)
    assert qw == pytest.approx(math.cos(math.pi / 4), abs=1e-6)


def test_heading_to_quat_rejects_zero_delta():
    with pytest.raises(ValueError):
        heading_to_quat((1.0, 1.0, 0.0), (1.0, 1.0, 0.0))


from core.trajectory import _square_loop_fallback


def test_square_loop_fallback_topology():
    traj = _square_loop_fallback(map_name="smallgrid")
    # 80 m square, perimeter 320 m
    # Sparse 25 m → ~13 samples; dense 8 m → ~40 samples
    assert traj.map_name == "smallgrid"
    assert traj.source == "fallback:square_loop"
    assert len(traj.sparse_waypoints) >= 12
    assert len(traj.sparse_waypoints) <= 16
    assert len(traj.dense_waypoints) >= 35
    # First waypoint is the spawn point
    assert traj.spawn_pos[:2] == traj.sparse_waypoints[0][:2]
    # Spawn is above the road (z offset)
    assert traj.spawn_pos[2] > traj.sparse_waypoints[0][2]


def test_square_loop_corners_are_at_expected_positions():
    traj = _square_loop_fallback(map_name="smallgrid")
    # Corners of an 80 m square around origin → expect points near (40,-40), (40,40), (-40,40), (-40,-40)
    xy = [(p[0], p[1]) for p in traj.sparse_waypoints]
    for cx, cy in [(40.0, -40.0), (40.0, 40.0), (-40.0, 40.0), (-40.0, -40.0)]:
        assert any(
            math.hypot(x - cx, y - cy) < 1e-3 for x, y in xy
        ), f"missing corner near ({cx}, {cy})"


from core.trajectory import _extract_longest_road, _edge_center


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
