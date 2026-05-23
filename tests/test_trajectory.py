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
