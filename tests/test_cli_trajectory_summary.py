from core.cli import format_trajectory_summary
from core.trajectory import MapTrajectories, TrajectoryData


def _traj(source):
    return TrajectoryData(
        spawn_pos=(0.0, 0.0, 1.0),
        spawn_rot=(0.0, 0.0, 0.0, 1.0),
        sparse_waypoints=[(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        dense_waypoints=[(0.0, 0.0, 0.0)],
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        source=source,
    )


def test_summary_reports_path_count():
    mt = MapTrajectories(
        map_name="italy",
        generated_at="2026-06-18T12:00:00+00:00",
        paths=[_traj("teleport:r1"), _traj("teleport:r2")],
    )
    summary = format_trajectory_summary(mt)
    assert "2 path" in summary
    assert "italy" in summary
