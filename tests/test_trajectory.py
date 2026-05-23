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
