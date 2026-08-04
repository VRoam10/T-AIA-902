"""The envs' guide polyline: spawn-first, and the source of cross-track + progress."""

import numpy as np
import pytest

from core.trajectory import TrajectoryData
from environments.beamng import BeamNGDrivingEnv
from environments.beamng_multi import BeamNGMultiEnv, VehicleSlot

SPAWN = (0.0, 0.0, 0.0)
WAYPOINTS = [(100.0, 0.0, 0.0), (100.0, 100.0, 0.0)]


def _traj():
    return TrajectoryData(
        spawn_pos=SPAWN,
        spawn_rot=(0.0, 0.0, 0.0, 1.0),
        sparse_waypoints=list(WAYPOINTS),
        dense_waypoints=list(WAYPOINTS),
        map_name="gridmap_v2",
        generated_at="2026-08-04T00:00:00+00:00",
        source="test",
    )


def _env():
    env = BeamNGDrivingEnv(beamng_home="unused")
    env.trajectory = _traj()
    env.waypoints = list(WAYPOINTS)
    env._rebuild_guide_line()
    return env


class TestGuideLine:
    def test_starts_at_the_spawn_so_the_first_segment_counts(self):
        # Projecting onto the waypoints alone would clamp progress to 0 until the
        # car had passed checkpoint 0 — the whole first segment would be invisible.
        env = _env()
        assert env._guide_line[0] == SPAWN
        assert len(env._guide_line) == len(WAYPOINTS) + 1

    def test_progress_grows_along_the_first_segment(self):
        env = _env()
        assert env._project((40.0, 0.0, 0.0)).progress_m == pytest.approx(40.0)

    def test_progress_keeps_growing_round_the_corner(self):
        env = _env()
        assert env._project((100.0, 60.0, 0.0)).progress_m == pytest.approx(160.0)

    def test_no_trajectory_means_a_neutral_projection(self):
        bare = BeamNGDrivingEnv(beamng_home="unused")
        assert bare._project((5.0, 5.0, 0.0)).progress_m == 0.0


class TestPathErrorsReturnsTwoValues:
    def test_heading_and_dist_only(self):
        env = _env()
        heading_err, dist = env._path_errors((0.0, 0.0, 0.0), {"vel": (1.0, 0.0, 0.0)})
        assert dist == pytest.approx(100.0)
        assert heading_err == pytest.approx(0.0)


class TestMultiEnvProgress:
    def test_progress_of_uses_the_slot_guide_line(self):
        slot = VehicleSlot(name="ego_0", color="White", agent=None, save_path="")
        slot.waypoints = list(WAYPOINTS)
        slot.spawn_pos = SPAWN
        slot.guide_line = [SPAWN, *WAYPOINTS]
        slot.current_pos = (40.0, 0.0, 0.0)
        env = BeamNGMultiEnv(slots=[slot], beamng_home="unused")
        assert env.progress_of(slot) == pytest.approx(40.0)

    def test_progress_of_is_zero_without_a_guide_line(self):
        slot = VehicleSlot(name="ego_0", color="White", agent=None, save_path="")
        env = BeamNGMultiEnv(slots=[slot], beamng_home="unused")
        assert env.progress_of(slot) == 0.0


class TestTrackProgressIsGone:
    def test_the_superseded_helper_is_deleted(self):
        import environments.beamng_geometry as geometry

        assert not hasattr(geometry, "track_progress_m")
