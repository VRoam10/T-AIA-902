"""Unit tests for environments.beamng_geometry — pure LiDAR geometry."""

import numpy as np
import pytest

from environments.beamng_geometry import (
    LidarConfig,
    ego_local_extents_from_bbox,
    lidar_keep_mask,
    process_lidar,
    world_to_local,
)

CFG = LidarConfig(
    rays=8,
    v_bins=1,
    channels=1,
    fov_deg=120.0,
    vert_angle=6.0,
    max_dist=50.0,
    self_margin=0.30,
    ground_clearance=0.30,
)

CFG_360 = LidarConfig(
    rays=8,
    v_bins=1,
    channels=1,
    fov_deg=360.0,
    vert_angle=6.0,
    max_dist=50.0,
    self_margin=0.30,
    ground_clearance=0.30,
)


class TestEgoLocalExtents:
    def test_returns_none_without_bbox(self):
        assert ego_local_extents_from_bbox({}, {"pos": (0, 0, 0)}, 0.3) is None

    def test_returns_none_without_pos(self):
        bbox = {"a": (1.0, 1.0, 1.0)}
        assert ego_local_extents_from_bbox(bbox, {}, 0.3) is None

    def test_axis_aligned_box_extents_include_margin(self):
        # Vehicle at origin, heading +X (dir=(1,0,0)). A unit box's local extents
        # should equal world extents (no rotation) expanded by the margin.
        bbox = {
            "c0": (-1.0, -0.5, 0.0),
            "c1": (1.0, 0.5, 1.5),
        }
        state = {"pos": (0.0, 0.0, 0.0), "dir": (1.0, 0.0, 0.0)}
        ext = ego_local_extents_from_bbox(bbox, state, margin=0.3)
        x_min, x_max, y_min, y_max, z_min, z_max = ext
        assert x_min == pytest.approx(-1.3)
        assert x_max == pytest.approx(1.3)
        assert y_min == pytest.approx(-0.8)
        assert y_max == pytest.approx(0.8)
        assert z_min == pytest.approx(-0.3)
        assert z_max == pytest.approx(1.8)

    def test_rejects_world_scale_garbage_box(self):
        # A bad pose leaves corners at world scale; the de-rotated box exceeds the
        # plausible half-extent and must be discarded (None) rather than disabling
        # the LiDAR with a giant self-filter box.
        bbox = {"c0": (-500.0, -500.0, 0.0), "c1": (500.0, 500.0, 2.0)}
        state = {"pos": (0.0, 0.0, 0.0), "dir": (1.0, 0.0, 0.0)}
        assert ego_local_extents_from_bbox(bbox, state, 0.3, max_half_extent=10.0) is None

    def test_keeps_plausible_box_within_bound(self):
        bbox = {"c0": (-2.0, -1.0, 0.0), "c1": (2.0, 1.0, 1.5)}
        state = {"pos": (0.0, 0.0, 0.0), "dir": (1.0, 0.0, 0.0)}
        assert ego_local_extents_from_bbox(bbox, state, 0.3, max_half_extent=10.0) is not None


class TestWorldToLocal:
    def test_identity_heading_translates_only(self):
        pts = np.array([[5.0, 0.0, 1.0]], dtype=np.float32)
        lx, ly, lz = world_to_local(pts, (1.0, 0.0, 0.0), heading=0.0)
        assert lx[0] == pytest.approx(4.0)
        assert ly[0] == pytest.approx(0.0)
        assert lz[0] == pytest.approx(1.0)

    def test_heading_90deg_rotates_into_local(self):
        # Heading +90deg (facing +Y). A point straight ahead in world +Y maps to local +X.
        pts = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)
        lx, ly, lz = world_to_local(pts, (0.0, 0.0, 0.0), heading=np.pi / 2)
        assert lx[0] == pytest.approx(10.0, abs=1e-5)
        assert ly[0] == pytest.approx(0.0, abs=1e-5)


class TestLidarKeepMask:
    def test_rejects_points_inside_ego_box(self):
        lx = np.array([0.0, 5.0], dtype=np.float32)
        ly = np.array([0.0, 0.0], dtype=np.float32)
        lz = np.array([1.0, 1.0], dtype=np.float32)
        ext = (-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
        keep, dbg = lidar_keep_mask(lx, ly, lz, ext, self_margin=0.3, ground_clearance=0.3)
        assert keep.tolist() == [False, True]
        assert dbg["self"] == 1
        assert dbg["kept"] == 1

    def test_rejects_ground_points(self):
        lx = np.array([5.0, 5.0], dtype=np.float32)
        ly = np.array([0.0, 0.0], dtype=np.float32)
        lz = np.array([-0.5, 2.0], dtype=np.float32)  # first is below ground
        ext = (-1.0, 1.0, -1.0, 1.0, 0.0, 2.0)
        keep, dbg = lidar_keep_mask(lx, ly, lz, ext, self_margin=0.3, ground_clearance=0.3)
        assert keep.tolist() == [False, True]
        assert dbg["ground"] == 1

    def test_no_extents_uses_flat_ground_threshold(self):
        lx = np.array([5.0], dtype=np.float32)
        ly = np.array([0.0], dtype=np.float32)
        lz = np.array([1.0], dtype=np.float32)
        keep, dbg = lidar_keep_mask(lx, ly, lz, None, self_margin=0.3, ground_clearance=0.3)
        assert keep.tolist() == [True]
        assert dbg["extents_none"] is True


class TestProcessLidar:
    def test_empty_cloud_returns_all_clear(self):
        out, dbg = process_lidar(None, (0, 0, 0), 0.0, None, CFG)
        assert out.shape == (8,)
        assert np.all(out == 1.0)

    def test_single_obstacle_lands_in_one_bin_and_is_normalized(self):
        # One point 25 m straight ahead (local +X), above ground, no ego box.
        # Distance 25 / max 50 = 0.5 in the centre bin; others stay clear (1.0).
        cloud = np.array([[25.0, 0.0, 1.0]], dtype=np.float32)
        out, dbg = process_lidar(cloud, (0, 0, 0), 0.0, None, CFG)
        assert out.min() == pytest.approx(0.5, abs=1e-3)
        assert (out == 1.0).sum() == 7
        assert dbg["fov"] == 1

    def test_point_outside_fov_is_ignored(self):
        # Point directly behind (local -X) is outside the 120deg forward FOV.
        cloud = np.array([[-25.0, 0.0, 1.0]], dtype=np.float32)
        out, dbg = process_lidar(cloud, (0, 0, 0), 0.0, None, CFG)
        assert np.all(out == 1.0)

    def test_full_360_keeps_rear_obstacle(self):
        # Point directly behind (local -X) is kept in full 360 mode.
        cloud = np.array([[-25.0, 0.0, 1.0]], dtype=np.float32)
        out, dbg = process_lidar(cloud, (0, 0, 0), 0.0, None, CFG_360)
        assert out.min() == pytest.approx(0.5, abs=1e-3)
        assert (out == 1.0).sum() == 7
        assert dbg["fov"] == 1

    def test_full_360_rejects_ego_body_before_binning(self):
        # Near point inside the ego OBB is rejected; rear obstacle still binned.
        cloud = np.array([[-1.0, 0.0, 1.0], [-25.0, 0.0, 1.0]], dtype=np.float32)
        ego_extents = (-2.0, 2.0, -1.0, 1.0, 0.0, 2.0)
        out, dbg = process_lidar(cloud, (0, 0, 0), 0.0, ego_extents, CFG_360)
        assert dbg["self"] == 1
        assert dbg["kept"] == 1
        assert out.min() == pytest.approx(0.5, abs=1e-3)


class TestBodyOrientationFeatures:
    def test_flat_vehicle_reads_zero(self):
        from environments.beamng_geometry import body_orientation_features

        out = body_orientation_features((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)

    def test_nose_up_is_positive_pitch_zero_roll(self):
        from environments.beamng_geometry import body_orientation_features

        # facing +Y, body tilted nose-up: up vector leans backward (-Y)
        pitch, roll = body_orientation_features((0.0, 1.0, 0.0), (0.0, -0.3, 0.95))
        assert pitch > 0.0
        assert abs(roll) < 1e-6

    def test_lean_right_is_positive_roll(self):
        from environments.beamng_geometry import body_orientation_features

        # facing +Y (lateral axis = +X), up vector leans right (+X) -> roll > 0
        pitch, roll = body_orientation_features((0.0, 1.0, 0.0), (0.3, 0.0, 0.95))
        assert roll > 0.0
        assert abs(pitch) < 1e-6

    def test_saturates_at_one(self):
        from environments.beamng_geometry import body_orientation_features

        pitch, _ = body_orientation_features((0.0, 1.0, 0.0), (0.0, -5.0, 0.1))
        assert pitch == pytest.approx(1.0)
