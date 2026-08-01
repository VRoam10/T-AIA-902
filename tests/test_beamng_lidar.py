"""Unit tests for BeamNG LiDAR geometry helpers."""

import pytest

from environments.beamng import BeamNGDrivingEnv


def _env_with_extents(extents, sensor="lidar"):
    # __new__ bypasses __init__, so set the handles close() and the mount/creation
    # helpers touch (real construction always sets these in __init__).
    env = BeamNGDrivingEnv.__new__(BeamNGDrivingEnv)
    env._ego_local_extents = extents
    env.sensor = sensor
    env.lidar = None
    env.camera = None
    env.roads_sensor = None
    return env


class _FakeLidar:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class _FakeBeamNG:
    def __init__(self):
        self.closed = False
        self.disconnected = False

    def close(self):
        self.closed = True

    def disconnect(self):
        self.disconnected = True


def test_lidar_mount_starts_above_cached_vehicle_bbox_for_beamng_snapping():
    env = _env_with_extents((-4.0, 6.0, -1.4, 1.4, 0.2, 3.8))

    mount_x, mount_y, mount_z = env._resolve_lidar_mount_pos()

    assert mount_x == pytest.approx(0.0)
    assert mount_y == pytest.approx(0.0)
    assert mount_z > 3.8


def test_lidar_mount_scales_roof_seed_with_small_vehicle_bbox():
    env = _env_with_extents((-1.1, 1.0, -0.45, 0.45, 0.1, 1.3))

    mount_x, mount_y, mount_z = env._resolve_lidar_mount_pos()

    assert mount_x == pytest.approx(0.0)
    assert mount_y == pytest.approx(0.0)
    assert mount_z == pytest.approx(1.3 + BeamNGDrivingEnv.LIDAR_SELF_MARGIN)


def test_lidar_mount_keeps_configured_fallback_without_cached_bbox():
    env = _env_with_extents(None)

    assert env._resolve_lidar_mount_pos() == BeamNGDrivingEnv.LIDAR_MOUNT_POS


def test_lidar_creation_uses_full_360_without_surface_snapping():
    env = _env_with_extents((-4.0, 6.0, -1.4, 1.4, 0.2, 3.8))

    kwargs = env._lidar_creation_kwargs()

    assert kwargs["pos"] == env._resolve_lidar_mount_pos()
    assert kwargs["is_360_mode"] is True
    assert kwargs["is_rotate_mode"] is False
    assert kwargs["horizontal_angle"] == 360.0
    assert kwargs["is_snapping_desired"] is False
    assert kwargs["is_force_inside_triangle"] is False


def test_lidar_creation_follows_the_sensor_axis():
    """adv_lidar must reach the physical sensor, not just the observation size: it
    trades vertical resolution for a narrower FOV so its 4 rows span useful
    elevations instead of mostly sky and mostly road."""
    plain = _env_with_extents(None, sensor="lidar")._lidar_creation_kwargs()
    adv = _env_with_extents(None, sensor="adv_lidar")._lidar_creation_kwargs()

    assert (plain["vertical_resolution"], plain["vertical_angle"]) == (32, 26.9)
    assert (adv["vertical_resolution"], adv["vertical_angle"]) == (16, 20.0)


def test_remove_lidar_detaches_current_sensor_before_vehicle_replacement():
    env = _env_with_extents(None)
    lidar = _FakeLidar()
    env.lidar = lidar

    env._remove_lidar()

    assert lidar.removed is True
    assert env.lidar is None


def test_close_can_disconnect_without_killing_human_play_simulator():
    env = _env_with_extents(None)
    lidar = _FakeLidar()
    env.lidar = lidar
    bng = _FakeBeamNG()
    env.bng = bng
    env.vehicle = object()

    env.close(kill_sim=False)
    assert lidar.removed is True
    assert env.lidar is None

    assert bng.disconnected is True
    assert bng.closed is False
    assert env.bng is None
    assert env.vehicle is None
