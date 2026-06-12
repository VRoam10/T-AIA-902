"""Tests for environments.beamng — refactored LiDAR delegation (no sim)."""

import numpy as np
import pytest

from environments.beamng import BeamNGDrivingEnv


def _bare_env():
    # __init__ only stores config; no BeamNG connection is opened.
    return BeamNGDrivingEnv(beamng_home="unused")


class TestProcessLidarDelegation:
    def test_empty_cloud_all_clear(self):
        env = _bare_env()
        env._ego_local_extents = None
        out = env._process_lidar(None, (0.0, 0.0, 0.0), 0.0)
        assert out.shape == (BeamNGDrivingEnv.LIDAR_RAYS,)
        assert np.all(out == 1.0)

    def test_single_obstacle_normalized_into_one_bin(self):
        env = _bare_env()
        env._ego_local_extents = None
        cloud = np.array([[25.0, 0.0, 1.0]], dtype=np.float32)
        out = env._process_lidar(cloud, (0.0, 0.0, 0.0), 0.0)
        assert out.min() == pytest.approx(0.5, abs=1e-3)
        assert (out == 1.0).sum() == BeamNGDrivingEnv.LIDAR_RAYS - 1
        # debug populated as a side effect, as before
        assert env._lidar_debug["fov"] == 1


class TestNoNpcApi:
    def test_npc_helpers_removed(self):
        assert not hasattr(BeamNGDrivingEnv, "_spawn_npc_vehicles")
        assert not hasattr(BeamNGDrivingEnv, "NPC_COUNT")
