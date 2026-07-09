"""Tests for environments.beamng — refactored LiDAR delegation (no sim)."""

import socket
import time
from unittest.mock import MagicMock

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

    def test_rear_obstacle_visible_with_default_360_fov(self):
        env = _bare_env()
        env._ego_local_extents = None
        cloud = np.array([[-25.0, 0.0, 1.0]], dtype=np.float32)
        out = env._process_lidar(cloud, (0.0, 0.0, 0.0), 0.0)
        assert out.min() == pytest.approx(0.5, abs=1e-3)
        assert (out == 1.0).sum() == BeamNGDrivingEnv.LIDAR_RAYS - 1


class TestNoNpcApi:
    def test_npc_helpers_removed(self):
        assert not hasattr(BeamNGDrivingEnv, "_spawn_npc_vehicles")
        assert not hasattr(BeamNGDrivingEnv, "NPC_COUNT")


class TestExtraFeatures:
    def test_flags_default_off_no_extra(self):
        env = BeamNGDrivingEnv(beamng_home="unused")
        assert env.body_orientation is False
        assert env.wheel_terrain is False
        assert env._extra_features({}).shape == (0,)
        assert env.n_states == BeamNGDrivingEnv.N_STATES  # 14

    def test_n_states_accounts_for_flags(self):
        base = BeamNGDrivingEnv.N_STATES
        assert BeamNGDrivingEnv(beamng_home="x", body_orientation=True).n_states == base + 2
        assert BeamNGDrivingEnv(beamng_home="x", wheel_terrain=True).n_states == base + 2
        both = BeamNGDrivingEnv(beamng_home="x", body_orientation=True, wheel_terrain=True)
        assert both.n_states == base + 4

    def test_n_states_combines_flags_and_hints(self):
        env = BeamNGDrivingEnv(
            beamng_home="x", trajectory_hints=2, body_orientation=True, wheel_terrain=True
        )
        assert env.n_states == BeamNGDrivingEnv.N_STATES + 4 + 2 + 2

    def test_extra_features_order_is_orientation_then_terrain(self):
        env = BeamNGDrivingEnv(beamng_home="x", body_orientation=True, wheel_terrain=True)
        env.roads_sensor = None
        state = {"dir": (0.0, 1.0, 0.0), "up": (0.0, -0.3, 0.95)}
        out = env._extra_features(state)
        assert out.shape == (4,)
        assert out[0] > 0.0  # pitch (nose up) first
        assert out[2] == pytest.approx(0.0, abs=1e-6)  # left terrain (neutral) after

    def test_wheel_terrain_wrapper_reads_sensor(self):
        env = BeamNGDrivingEnv(beamng_home="x", wheel_terrain=True)
        env.roads_sensor = MagicMock()
        env.roads_sensor.poll.return_value = {"halfWidth": 3.0, "dist2Left": 3.7, "dist2Right": 0.7}
        left, right = env._wheel_terrain_features()
        assert left == pytest.approx(1.0, abs=1e-6)
        assert right == pytest.approx(0.0, abs=1e-6)


class TestCloseWaitsForSimShutdown:
    """close(kill_sim=True) must not return while the sim port still accepts
    connections. BeamNGpy open(launch=True) connects to any instance still
    listening before it launches a new one, so returning early lets the next
    env (benchmark eval right after training) latch onto the dying simulator
    and die with BNGDisconnectedError instead of relaunching the game."""

    def _fake_sim_env(self, close_delay: float):
        """Env pointed at a local listening socket standing in for the sim.

        The mocked BeamNGpy close() sleeps past the join timeout before
        releasing the port, mimicking the real shutdown (scenario close,
        Quit ack, process kill) outliving the daemon-thread join.
        """
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        env = BeamNGDrivingEnv(beamng_home="unused", host="127.0.0.1", port=port)
        env.CLOSE_JOIN_TIMEOUT = 0.1
        env.KILL_WAIT_POLL = 0.05
        env.KILL_WAIT_TIMEOUT = 10.0

        env.bng = MagicMock()

        def slow_close():
            time.sleep(close_delay)
            listener.close()

        env.bng.close.side_effect = slow_close
        env.bng.disconnect.side_effect = lambda: None
        return env, port

    def test_kill_sim_blocks_until_port_refuses_connections(self):
        env, port = self._fake_sim_env(close_delay=1.0)
        env.close(kill_sim=True)
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()

    def test_disconnect_leaves_sim_running(self):
        env, port = self._fake_sim_env(close_delay=1.0)
        start = time.time()
        env.close(kill_sim=False)
        assert time.time() - start < 1.0  # no port wait on disconnect
        # The sim (listener) is still accepting: disconnect must not kill it.
        socket.create_connection(("127.0.0.1", port), timeout=0.5).close()


class TestRoadsSensorLifecycle:
    def test_attach_roads_sensor_noop_when_flag_off(self):
        env = BeamNGDrivingEnv(beamng_home="x", wheel_terrain=False)
        env.bng = MagicMock()
        env.vehicle = MagicMock()
        env._attach_roads_sensor()
        assert env.roads_sensor is None

    def test_remove_roads_sensor_clears_handle(self):
        env = BeamNGDrivingEnv(beamng_home="x", wheel_terrain=True)
        env.roads_sensor = MagicMock()
        env._remove_roads_sensor()
        assert env.roads_sensor is None


class TestContinuousRollDeleted:
    def test_class_is_gone(self):
        import environments.beamng as m

        assert not hasattr(m, "BeamNGContinuousRollEnv")


class TestRegistry:
    def test_continuous_roll_not_registered(self):
        import environments  # noqa: F401  (triggers registration)
        from core.registry import registry

        assert "beamng_continuous_roll" not in registry.list_environments()

    def test_beamng_factory_forwards_flags(self):
        from environments import _make_beamng

        env = _make_beamng(body_orientation=True, wheel_terrain=True)
        assert env.body_orientation is True
        assert env.wheel_terrain is True

    def test_all_beamng_factories_forward_flags(self):
        # Every registered beamng factory must accept and forward both flags —
        # the subclasses override __init__, so a base-only change misses them.
        from environments import (
            _make_beamng,
            _make_beamng_camera,
            _make_beamng_continuous,
            _make_beamng_lidar,
        )

        for factory in (
            _make_beamng,
            _make_beamng_lidar,
            _make_beamng_continuous,
            _make_beamng_camera,
        ):
            env = factory(body_orientation=True, wheel_terrain=True)
            assert env.body_orientation is True, factory.__name__
            assert env.wheel_terrain is True, factory.__name__
            assert env.n_states == env.N_STATES + 4, factory.__name__
