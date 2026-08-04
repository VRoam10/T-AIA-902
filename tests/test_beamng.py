"""Tests for environments.beamng — refactored LiDAR delegation (no sim)."""

import socket
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.trajectory import TrajectoryData
from environments import beamng_spec
from environments.beamng import BeamNGDrivingEnv


def _bare_env(**kwargs):
    # __init__ only stores config; no BeamNG connection is opened.
    return BeamNGDrivingEnv(beamng_home="unused", **kwargs)


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


def _traj(spawn_z=51.92):
    return TrajectoryData(
        spawn_pos=(700.0, -6.7, spawn_z),
        spawn_rot=(0.0, 0.0, 0.0, 1.0),
        sparse_waypoints=[(710.0, -6.7, 51.9), (730.0, -6.7, 51.9)],
        dense_waypoints=[(705.0, -6.7, 51.9)],
        map_name="east_coast_usa",
        generated_at="2026-07-16T08:54:17+00:00",
        source="teleport:30079.0",
    )


class TestTrackSelection:
    """A chosen game track must replace the generated paths, end to end.

    The regression this guards is the whole point: `track` existed at every layer
    but the registry factory's **kwargs sink dropped it, so a run silently drove
    the generated road-network paths and nothing said why.
    """

    def _home(self, tmp_path):
        """A BeamNG install holding one two-node sprint on 'italy'."""
        import json
        import zipfile

        levels = tmp_path / "content" / "levels"
        levels.mkdir(parents=True)
        spec = {
            "name": "Test Sprint",
            "classification": {"closed": False},
            "defaultStartPosition": 1,
            "startPositions": [{"oldId": 1, "pos": [0.0, 0.0, 5.0]}],
            "startNode": 1,
            "pathnodes": [
                {"oldId": 1, "pos": [50.0, 0.0, 5.0], "radius": 8},
                {"oldId": 2, "pos": [150.0, 0.0, 5.0], "radius": 8},
            ],
        }
        with zipfile.ZipFile(levels / "italy.zip", "w") as z:
            z.writestr("levels/italy/quickrace/t.race.json", json.dumps(spec))
        return tmp_path

    def test_track_branch_needs_no_cache_and_no_simulator(self, tmp_path):
        # Reading level files is all it takes, so this branch skips both the cache
        # and the probe scenario the generated path needs.
        env = BeamNGDrivingEnv(beamng_home=str(self._home(tmp_path)), map_name="italy", track="t")
        traj = env._resolve_trajectory()
        assert traj.source == "quickrace:t:sprint"
        assert traj.spawn_pos == (0.0, 0.0, 5.0)
        assert traj.sparse_waypoints == [(50.0, 0.0, 5.0), (150.0, 0.0, 5.0)]

    def test_the_track_becomes_the_only_path(self, tmp_path):
        env = BeamNGDrivingEnv(beamng_home=str(self._home(tmp_path)), map_name="italy", track="t")
        env._resolve_trajectory()
        assert len(env._paths) == 1
        assert env.trajectory is env._paths[0]

    def test_an_unknown_track_fails_loudly(self, tmp_path):
        # Better than silently driving something else, which is what it used to do.
        env = BeamNGDrivingEnv(beamng_home=str(self._home(tmp_path)), map_name="italy", track="nope")
        with pytest.raises(ValueError, match="not a usable race track"):
            env._resolve_trajectory()


class TestSpawnHeight:
    """A teleport must land the car where it rests, not above or below it.

    The cache stores the road surface (SPAWN_Z_OFFSET_M is 0); Vehicle.teleport has
    no cling and places the reference point exactly, so the car's ride height has to
    be added or every reset drops it onto race suspension and damages the engine.
    """

    def test_spawn_target_applies_the_measured_correction(self):
        env = _bare_env()
        env.trajectory = _traj(51.92)
        env._spawn_z_correction = 0.36
        assert env._spawn_target() == (700.0, -6.7, pytest.approx(52.28))

    def test_spawn_target_defaults_to_the_cached_height(self):
        # Nothing measured yet (or measurement rejected): use the cache as-is.
        env = _bare_env()
        env.trajectory = _traj(51.92)
        assert env._spawn_target() == (700.0, -6.7, 51.92)

    def test_human_respawn_teleports_to_the_corrected_height(self):
        env = _bare_env()
        env.bng = None  # keeps _update_active_marker a no-op
        env.vehicle = MagicMock()
        env.trajectory = _traj(51.92)
        env._paths = [env.trajectory]
        env._spawn_z_correction = 0.36
        env._reset_human_episode()
        pos = env.vehicle.teleport.call_args.args[0]
        assert pos[2] == pytest.approx(52.28)


class TestExtraFeatures:
    def test_flags_default_off_no_extra(self):
        env = BeamNGDrivingEnv(beamng_home="unused")
        assert env.body_orientation is False
        assert env.road_info is False
        assert env._extra_features({}, (0.0, 0.0, 0.0), 0.0).shape == (0,)
        assert env.n_states == beamng_spec.obs_size("lidar")  # 14

    def test_n_states_accounts_for_flags(self):
        base = beamng_spec.obs_size("lidar")
        assert BeamNGDrivingEnv(beamng_home="x", body_orientation=True).n_states == base + 2
        assert BeamNGDrivingEnv(beamng_home="x", road_info=True).n_states == base + 6
        both = BeamNGDrivingEnv(beamng_home="x", body_orientation=True, road_info=True)
        assert both.n_states == base + 8

    def test_n_states_combines_flags_and_hints(self):
        env = BeamNGDrivingEnv(
            beamng_home="x", trajectory_hints=2, body_orientation=True, road_info=True
        )
        assert env.n_states == beamng_spec.obs_size("lidar") + 4 + 2 + 6

    def test_extra_features_order_is_orientation_then_road(self):
        env = _bare_env(body_orientation=True, road_info=True)
        state = {"dir": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0)}
        out = env._extra_features(state, (0.0, 0.0, 0.0), 0.0)
        assert out.shape == (8,)
        # No RoadsSensor attached, so the road block is neutral and identifiable.
        np.testing.assert_allclose(out[2:], [0.0] * 6, atol=1e-6)

    def test_road_block_is_neutral_without_a_sensor(self):
        env = _bare_env(road_info=True)
        out = env._road_info_features((0.0, 0.0, 0.0), 0.0)
        assert out.shape == (6,)
        np.testing.assert_allclose(out, [0.0] * 6, atol=1e-6)

    def test_n_states_counts_the_road_block(self):
        base = _bare_env().n_states
        assert _bare_env(road_info=True).n_states == base + 6


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
        env = BeamNGDrivingEnv(beamng_home="x", road_info=False)
        env.bng = MagicMock()
        env.vehicle = MagicMock()
        env._attach_roads_sensor()
        assert env.roads_sensor is None

    def test_remove_roads_sensor_clears_handle(self):
        env = BeamNGDrivingEnv(beamng_home="x", road_info=True)
        env.roads_sensor = MagicMock()
        env._remove_roads_sensor()
        assert env.roads_sensor is None


class TestSubclassesCollapsed:
    """The per-sensor / per-output subclasses are gone: one env now takes both as
    arguments, so there is no class whose name can disagree with its behaviour."""

    def test_old_env_classes_are_gone(self):
        import environments.beamng as m

        for name in (
            "BeamNGLidarEnv",
            "BeamNGContinuousEnv",
            "BeamNGCameraEnv",
            "BeamNGContinuousRollEnv",
        ):
            assert not hasattr(m, name), name

    def test_only_one_beamng_env_is_registered(self):
        import environments  # noqa: F401  (triggers registration)
        from core.registry import registry

        assert registry.list_environments() == ["beamng"]


class TestObservationLayoutPerSensor:
    """Flag-off observation lengths must equal the pre-refactor per-class lengths,
    or every existing assumption about obs slicing breaks."""

    @pytest.mark.parametrize(
        ("sensor", "expected"), [("lidar", 14), ("adv_lidar", 38), ("camera", 262)]
    )
    def test_n_states_matches_historical_length(self, sensor, expected):
        assert _bare_env(sensor=sensor).n_states == expected

    @pytest.mark.parametrize("sensor", beamng_spec.SENSORS)
    def test_n_perception_matches_the_spec(self, sensor):
        env = _bare_env(sensor=sensor)
        assert env.n_perception == beamng_spec.perception_features(sensor)


class TestControlsFor:
    """The output axis decides how an action becomes (throttle, steering, brake)."""

    def test_fixed_indexes_the_action_table(self):
        env = _bare_env(output="fixed")
        for i, entry in enumerate(BeamNGDrivingEnv.ACTIONS):
            assert env.controls_for(i) == (entry["throttle"], entry["steering"], entry["brake"])

    def test_continuous_three_vector_maps_directly(self):
        env = _bare_env(output="continuous")
        assert env.controls_for(np.array([0.8, -0.5, 0.0])) == pytest.approx((0.8, -0.5, 0.0))

    def test_continuous_negative_throttle_is_clipped_not_reversed(self):
        env = _bare_env(output="continuous")
        throttle, _steer, brake = env.controls_for(np.array([-0.4, 0.0, 0.0]))
        assert (throttle, brake) == (0.0, 0.0)

    def test_continuous_two_vector_splits_accel_into_throttle_and_brake(self):
        env = _bare_env(output="continuous")
        assert env.controls_for(np.array([0.7, 0.2])) == pytest.approx((0.7, 0.2, 0.0))
        assert env.controls_for(np.array([-0.7, 0.2])) == pytest.approx((0.0, 0.2, 0.7))

    def test_continuous_clips_out_of_range_values(self):
        env = _bare_env(output="continuous")
        assert env.controls_for(np.array([5.0, -9.0, 3.0])) == pytest.approx((1.0, -1.0, 1.0))

    def test_int_action_reads_as_a_table_index_even_on_a_continuous_env(self):
        # A discrete agent driving a continuous env must still produce valid
        # controls rather than being indexed as an array.
        env = _bare_env(output="continuous")
        assert env.controls_for(1) == (1.0, 0.0, 0.0)


class TestRaceCar:
    def test_single_car_replaces_the_vehicle_table(self):
        assert not hasattr(BeamNGDrivingEnv, "VEHICLES")
        assert BeamNGDrivingEnv.RACE_CAR["model"] == "vivace"

    def test_env_no_longer_takes_a_vehicle_id(self):
        import inspect

        assert "vehicle_id" not in inspect.signature(BeamNGDrivingEnv.__init__).parameters


class TestActionTableTunedForTheRaceCar:
    """The car is a 682 hp AWD hillclimb build: full throttle mid-corner overwhelms the
    tyres. Steering and throttle must trade off, or the discrete policy has no usable
    cornering action."""

    def test_best_available_throttle_decreases_as_steering_increases(self):
        # Compare the *most* throttle the table offers at each steering magnitude:
        # the table also holds a deliberate coast (0 steering, 0 throttle), so a
        # plain sort over every entry would not express the tradeoff.
        best: dict[float, float] = {}
        for a in BeamNGDrivingEnv.ACTIONS:
            if a["brake"] > 0.0:
                continue
            steer = abs(a["steering"])
            best[steer] = max(best.get(steer, 0.0), a["throttle"])
        by_steer = [best[k] for k in sorted(best)]
        assert by_steer == sorted(by_steer, reverse=True)
        assert len(by_steer) >= 3, "table should offer several steering magnitudes"

    def test_sharp_turns_are_near_lift_off(self):
        sharp = [a for a in BeamNGDrivingEnv.ACTIONS if abs(a["steering"]) > 0.4]
        assert sharp, "no sharp-steering action in the table"
        assert all(a["throttle"] <= 0.2 for a in sharp)

    def test_table_length_matches_the_fixed_action_size(self):
        assert len(BeamNGDrivingEnv.ACTIONS) == beamng_spec.action_size("fixed")
