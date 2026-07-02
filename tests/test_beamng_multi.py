"""Tests for environments.beamng_multi."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from environments.beamng_geometry import body_orientation_features, wheel_terrain_features
from environments.beamng_multi import (
    BeamNGMultiEnv,
    VehicleSlot,
    _color_rgba,
    build_slots,
    env_profile,
    slot_n_states,
)


def _slot(**kw):
    defaults = dict(
        name="ego_0",
        color="Red",
        vehicle_id="taxi",
        agent=object(),
        reward_mode="default",
        action_space="discrete",
        save_path="outputs/dqn.pth",
    )
    defaults.update(kw)
    return VehicleSlot(**defaults)


class TestVehicleSlot:
    def test_episode_state_defaults_to_zero(self):
        s = _slot()
        assert s.waypoint_idx == 0
        assert s.steps == 0
        assert s.last_damage == 0.0
        assert s.checkpoint_hit is False
        assert s.done is False
        assert s.episode == 0
        assert s.reward_history == []

    def test_vehicle_slot_has_waypoints_field(self):
        s = _slot()
        assert s.waypoints == []

    def test_reset_episode_zeros_running_state_but_keeps_episode_count(self):
        s = _slot()
        s.waypoint_idx = 4
        s.steps = 123
        s.last_damage = 50.0
        s.ep_reward = 99.0
        s.checkpoint_hit = True
        s.done = True
        s.episode = 7
        s.reward_history.append(99.0)

        s.reset_episode()

        assert s.waypoint_idx == 0
        assert s.steps == 0
        assert s.last_damage == 0.0
        assert s.ep_reward == 0.0
        assert s.checkpoint_hit is False
        assert s.done is False
        # history + episode counter survive a reset
        assert s.episode == 7
        assert s.reward_history == [99.0]


class _FakeAgent:
    pass


SPECS = [
    {
        "algo": "dqn",
        "env": "beamng",
        "agent": _FakeAgent(),
        "vehicle_id": "taxi",
        "color": "Yellow",
        "save_path": "outputs/dqn.pth",
    },
    {
        "algo": "ddpg",
        "env": "beamng_continuous",
        "agent": _FakeAgent(),
        "vehicle_id": "ibishu_pigeon",
        "color": "Red",
        "save_path": "outputs/ddpg.pth",
    },
    {
        "algo": "td3",
        "env": "beamng_continuous",
        "agent": _FakeAgent(),
        "vehicle_id": "taxi",
        "color": "Blue",
        "save_path": "outputs/td3.pth",
    },
]


class TestEnvProfiles:
    def test_env_profile_returns_perception_type(self):
        assert env_profile("beamng") == "lidar"
        assert env_profile("beamng_lidar") == "lidar_grid"
        assert env_profile("beamng_camera") == "camera"
        assert env_profile("beamng_continuous") == "lidar"
        assert env_profile("unknown_env") == "lidar"  # fallback

    def test_slot_n_states_no_hints(self):
        assert slot_n_states("beamng") == 14  # 6 + 8 lidar
        assert slot_n_states("beamng_lidar") == 38  # 6 + 32 grid
        assert slot_n_states("beamng_camera") == 262  # 6 + 256 pixels

    def test_slot_n_states_with_hints(self):
        assert slot_n_states("beamng", trajectory_hints=1) == 16  # 14 + 2
        assert slot_n_states("beamng", trajectory_hints=2) == 18  # 14 + 4
        assert slot_n_states("beamng_camera", trajectory_hints=1) == 264  # 262 + 2


class TestBuildSlots:
    def test_names_are_unique_and_indexed(self):
        slots = build_slots(SPECS)
        assert [s.name for s in slots] == ["ego_0", "ego_1", "ego_2"]

    def test_reward_mode_and_action_space_derived_from_algo(self):
        slots = build_slots(SPECS)
        assert slots[0].reward_mode == "default"
        assert slots[0].action_space == "discrete"
        assert slots[1].reward_mode == "ddpg"
        assert slots[1].action_space == "continuous"
        assert slots[2].reward_mode == "ddpg"
        assert slots[2].action_space == "continuous"

    def test_perception_and_n_states_from_env(self):
        slots = build_slots(SPECS)
        assert slots[0].perception == "lidar"
        assert slots[0].n_states == 14
        assert slots[1].env_name == "beamng_continuous"
        assert slots[1].perception == "lidar"

    def test_continuous_algo_on_camera_uses_default_reward(self):
        # DDPG on a camera env: no LiDAR bins to reason about -> default reward.
        slots = build_slots(
            [
                {
                    "algo": "ddpg",
                    "env": "beamng_camera",
                    "agent": _FakeAgent(),
                    "vehicle_id": "taxi",
                    "color": "Red",
                    "save_path": "outputs/x.pth",
                }
            ]
        )
        assert slots[0].perception == "camera"
        assert slots[0].reward_mode == "default"
        assert slots[0].action_space == "continuous"
        assert slots[0].n_states == 262

    def test_carries_color_and_save_path(self):
        slots = build_slots(SPECS)
        assert slots[1].color == "Red"
        assert slots[1].save_path == "outputs/ddpg.pth"


def _env(slots=None):
    return BeamNGMultiEnv(slots=slots or build_slots(SPECS), beamng_home="unused")


class TestApplyAction:
    def test_discrete_action_maps_through_actions_table(self):
        env = _env()
        slot = env.slots[0]  # discrete
        slot.vehicle = MagicMock()
        env.apply_action(slot, 1)  # ACTIONS[1] = full throttle straight
        slot.vehicle.control.assert_called_once_with(throttle=1.0, steering=0.0, brake=0.0)

    @staticmethod
    def _assert_control(slot, *, throttle, steering, brake):
        # float32 -> Python float drifts slightly; compare with tolerance.
        slot.vehicle.control.assert_called_once()
        kwargs = slot.vehicle.control.call_args.kwargs
        assert kwargs["throttle"] == pytest.approx(throttle, abs=1e-5)
        assert kwargs["steering"] == pytest.approx(steering, abs=1e-5)
        assert kwargs["brake"] == pytest.approx(brake, abs=1e-5)

    def test_continuous_action_positive_accel_is_throttle(self):
        env = _env()
        slot = env.slots[1]  # continuous
        slot.vehicle = MagicMock()
        env.apply_action(slot, np.array([0.8, -0.5], dtype=np.float32))
        self._assert_control(slot, throttle=0.8, steering=-0.5, brake=0.0)

    def test_continuous_action_negative_accel_is_brake(self):
        env = _env()
        slot = env.slots[1]
        slot.vehicle = MagicMock()
        env.apply_action(slot, np.array([-0.6, 0.2], dtype=np.float32))
        self._assert_control(slot, throttle=0.0, steering=0.2, brake=0.6)

    def test_continuous_three_dim_action_is_throttle_steer_brake(self):
        env = _env()
        slot = env.slots[2]
        slot.vehicle = MagicMock()
        env.apply_action(slot, np.array([0.5, 0.1, 0.3], dtype=np.float32))
        self._assert_control(slot, throttle=0.5, steering=0.1, brake=0.3)


class TestPathErrorsAndReward:
    def test_path_errors_advance_waypoint_when_close(self):
        env = _env()
        slot = env.slots[0]
        slot.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot.waypoint_idx = 0
        state = {"vel": (1.0, 0.0, 0.0)}
        env._path_errors(slot, pos=(0.0, 0.0, 0.0), state=state)
        assert slot.waypoint_idx == 1
        assert slot.checkpoint_hit is True

    def test_default_reward_gives_checkpoint_bonus(self):
        env = _env()
        env.slots[0].waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]  # reward_mode "default"
        slot.checkpoint_hit = True
        slot.waypoint_idx = 1
        slot.checkpoint_dist = 0.0
        obs = np.zeros(slot.n_states, dtype=np.float32)
        obs[0] = 0.5  # moving (speed) so no stationary penalty
        obs[5:] = 1.0  # clear LiDAR bins (far); zeros read as an obstacle at 0 m (-5)
        reward, done = env.compute_reward(slot, obs)
        assert reward >= 100.0
        assert slot.checkpoint_hit is False  # consumed

    def test_default_reward_terminates_on_max_damage(self):
        env = _env()
        env.slots[0].waypoints = [(0.0, 0.0, 0.0)]
        slot = env.slots[0]
        obs = np.zeros(slot.n_states, dtype=np.float32)
        obs[0] = 0.5
        obs[4] = 1.0  # damage_norm = 1.0 -> 1000 damage == MAX_DAMAGE
        reward, done = env.compute_reward(slot, obs)
        assert done is True

    def test_ddpg_reward_rewards_progress(self):
        env = _env()
        env.slots[1].waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[1]  # reward_mode "ddpg"
        slot.last_dist = 50.0
        slot.current_dist = 40.0  # got 10 m closer
        obs = np.zeros(slot.n_states, dtype=np.float32)
        obs[0] = 0.4  # speed
        reward, done = env.compute_reward(slot, obs)
        assert reward > 0.0


class TestObserve:
    def _wire_slot_sensors(self, slot, *, speed, steering, damage, pos, vel, lidar_points):
        slot.vehicle = MagicMock()
        slot.vehicle.state = {"pos": pos, "vel": vel, "dir": vel}
        slot.electrics = MagicMock()
        slot.electrics.data = {"wheelspeed": speed, "steering": steering}
        slot.damage_sensor = MagicMock()
        slot.damage_sensor.data = {"damage": damage}
        slot.lidar = MagicMock()
        slot.lidar.poll.return_value = {"pointCloud": lidar_points}
        slot.ego_local_extents = None

    def test_observe_returns_vector_of_n_states(self):
        env = _env()
        slot = env.slots[0]
        slot.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        self._wire_slot_sensors(
            slot,
            speed=10.0,
            steering=0.0,
            damage=0.0,
            pos=(0.0, 0.0, 0.0),
            vel=(1.0, 0.0, 0.0),
            lidar_points=None,
        )
        obs = env.observe(slot)
        assert obs.shape == (slot.n_states,)
        assert obs[0] == pytest.approx(0.2, abs=1e-3)

    def test_observe_polls_each_slot_sensor(self):
        env = _env()
        slot = env.slots[0]
        slot.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        self._wire_slot_sensors(
            slot,
            speed=0.0,
            steering=0.0,
            damage=0.0,
            pos=(0.0, 0.0, 0.0),
            vel=(1.0, 0.0, 0.0),
            lidar_points=None,
        )
        env.observe(slot)
        slot.vehicle.poll_sensors.assert_called_once()
        slot.lidar.poll.assert_called_once()

    def test_observe_appends_extras_when_flags_on(self):
        env = _env()
        slot = env.slots[0]
        slot.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot.body_orientation = True
        slot.wheel_terrain = True
        slot.n_states = 18
        self._wire_slot_sensors(
            slot,
            speed=10.0,
            steering=0.0,
            damage=0.0,
            pos=(0.0, 0.0, 0.0),
            vel=(1.0, 0.0, 0.0),
            lidar_points=None,
        )
        slot.vehicle.state = {
            "pos": (0.0, 0.0, 0.0),
            "vel": (1.0, 0.0, 0.0),
            "dir": (1.0, 0.0, 0.0),
            "up": (0.0, 0.0, 1.0),
        }
        slot.roads_sensor = MagicMock()
        slot.roads_sensor.poll.return_value = {
            "halfWidth": 3.0,
            "dist2Left": 0.7,
            "dist2Right": 0.7,
        }
        obs = env.observe(slot)
        assert obs.shape == (18,)
        expected_body = body_orientation_features((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        expected_wheel = wheel_terrain_features(
            {"halfWidth": 3.0, "dist2Left": 0.7, "dist2Right": 0.7}, 0.7
        )
        np.testing.assert_allclose(obs[-4:-2], expected_body, atol=1e-6)
        np.testing.assert_allclose(obs[-2:], expected_wheel, atol=1e-6)


class TestCreateSlotSensor:
    def test_lidar_slot_uses_full_360_and_caches_ego_box_first(self):
        env = _env()
        env.bng = MagicMock()
        slot = env.slots[0]  # perception "lidar"
        slot.wheel_terrain = False
        slot.vehicle = MagicMock()
        slot.vehicle.state = {"pos": (0.0, 0.0, 0.0), "dir": (1.0, 0.0, 0.0)}
        slot.vehicle.get_bbox.return_value = {
            "near_bottom_left": (-2.0, -1.0, 0.0),
            "far_top_right": (2.0, 1.0, 1.6),
        }

        with patch("environments.beamng_multi.Lidar") as MockLidar:
            env._create_slot_sensor(slot)

        assert MockLidar.called
        kwargs = MockLidar.call_args.kwargs
        assert kwargs["is_360_mode"] is True
        assert kwargs["is_rotate_mode"] is False
        assert kwargs["is_snapping_desired"] is False
        assert kwargs["is_force_inside_triangle"] is False
        assert kwargs["horizontal_angle"] == 360.0
        assert slot.ego_local_extents is not None


class TestSlotExtraFeatures:
    def test_slot_n_states_with_flags(self):
        assert slot_n_states("beamng", body_orientation=True) == 16  # 14 + 2
        assert slot_n_states("beamng", wheel_terrain=True) == 16  # 14 + 2
        assert slot_n_states("beamng", body_orientation=True, wheel_terrain=True) == 18
        assert (
            slot_n_states("beamng", trajectory_hints=1, body_orientation=True, wheel_terrain=True)
            == 14 + 2 + 2 + 2
        )

    def test_build_slots_reads_flags(self):
        specs = [
            {
                "algo": "dqn",
                "env": "beamng",
                "agent": _FakeAgent(),
                "vehicle_id": "taxi",
                "color": "Yellow",
                "save_path": "outputs/x.pth",
                "body_orientation": True,
                "wheel_terrain": True,
            }
        ]
        slot = build_slots(specs)[0]
        assert slot.body_orientation is True
        assert slot.wheel_terrain is True
        assert slot.n_states == 18

    def test_build_slots_flags_default_off(self):
        slot = build_slots(SPECS)[0]
        assert slot.body_orientation is False
        assert slot.wheel_terrain is False
        assert slot.n_states == 14


class TestLifecycle:
    def test_step_physics_steps_once_for_all(self):
        env = _env()
        env.bng = MagicMock()
        env.step_physics()
        env.bng.step.assert_called_once_with(10)

    def test_reset_vehicle_teleports_to_its_grid_pose_and_resets_state(self):
        env = _env()
        slot = env.slots[0]
        slot.vehicle = MagicMock()
        slot.spawn_pos = (1.0, 2.0, 3.0)  # the slot's assigned grid pose
        slot.spawn_rot = (0.0, 0.0, 0.0, 1.0)
        slot.waypoint_idx = 5
        slot.steps = 99
        env.reset_vehicle(slot)
        slot.vehicle.teleport.assert_called_once_with(
            (1.0, 2.0, 3.0), rot_quat=(0.0, 0.0, 0.0, 1.0), reset=True
        )
        assert slot.waypoint_idx == 0
        assert slot.steps == 0

    def test_close_removes_lidars_and_closes_bng(self):
        env = _env()
        bng = MagicMock()
        env.bng = bng
        for s in env.slots:
            s.lidar = MagicMock()
        env.close()
        for s in env.slots:
            assert s.lidar is None
        bng.close.assert_called_once()
        assert env.bng is None


class TestPathAssignment:
    def _mt(self, n_paths):
        from core.trajectory import MapTrajectories, TrajectoryData

        paths = [
            TrajectoryData(
                spawn_pos=(float(i) * 100.0, 0.0, 1.0),
                spawn_rot=(0.0, 0.0, 0.0, 1.0),
                sparse_waypoints=[(float(i) * 100.0, 10.0, 0.0), (float(i) * 100.0, 20.0, 0.0)],
                dense_waypoints=[(float(i) * 100.0, 10.0, 0.0)],
                map_name="italy",
                generated_at="2026-06-18T12:00:00+00:00",
                source=f"teleport:r{i}",
            )
            for i in range(n_paths)
        ]
        return MapTrajectories(
            map_name="italy", generated_at="2026-06-18T12:00:00+00:00", paths=paths
        )

    def test_each_slot_gets_its_own_path(self):
        env = _env()  # 3 slots
        env.trajectories = self._mt(3)
        env._assign_paths()
        assert env.slots[0].spawn_pos == (0.0, 0.0, 1.0)
        assert env.slots[1].spawn_pos == (100.0, 0.0, 1.0)
        assert env.slots[2].spawn_pos == (200.0, 0.0, 1.0)
        assert env.slots[0].waypoints == [(0.0, 10.0, 0.0), (0.0, 20.0, 0.0)]
        assert env.slots[1].waypoints[0] == (100.0, 10.0, 0.0)
        # Distinct spawns -> no shared start line.
        assert len({s.spawn_pos for s in env.slots}) == 3

    def test_more_vehicles_than_paths_raises(self):
        env = _env()  # 3 slots
        env.trajectories = self._mt(2)
        with pytest.raises(ValueError, match="only 2 distinct path"):
            env._assign_paths()

    def test_random_assign_gives_distinct_paths(self, monkeypatch):
        env = _env()  # 3 slots
        env.random_path = True
        env.trajectories = self._mt(5)
        monkeypatch.setattr(
            "environments.beamng_multi.random.shuffle",
            lambda seq: seq.reverse(),
        )
        env._assign_paths()
        idxs = [s.path_idx for s in env.slots]
        assert len(set(idxs)) == 3  # distinct
        # reversed [0,1,2,3,4] -> [4,3,2,1,0]; first 3 dealt
        assert idxs == [4, 3, 2]

    def test_pick_distinct_path_idx_avoids_other_slots(self):
        env = _env()  # 3 slots
        env.random_path = True
        env.trajectories = self._mt(3)
        env.slots[1].path_idx = 1
        env.slots[2].path_idx = 2
        # only index 0 is free for slot 0
        assert env._pick_distinct_path_idx(env.slots[0]) == 0

    def test_assign_paths_not_random_is_sequential(self):
        env = _env()
        env.trajectories = self._mt(3)
        env._assign_paths()
        assert [s.path_idx for s in env.slots] == [0, 1, 2]


class TestMarkers:
    def test_color_rgba_known_case_insensitive(self):
        assert _color_rgba("Red") == (1.0, 0.0, 0.0, 0.8)
        assert _color_rgba("red") == (1.0, 0.0, 0.0, 0.8)
        assert _color_rgba("Yellow") == (1.0, 1.0, 0.0, 0.8)

    def test_color_rgba_unknown_returns_default(self):
        assert _color_rgba("chartreuse") == (0.0, 1.0, 0.2, 0.8)

    def test_update_slot_marker_adds_sphere_in_slot_color(self):
        env = _env()
        env.bng = MagicMock()
        env.bng.debug.add_spheres.return_value = ["sphere-1"]
        slot = env.slots[0]
        slot.waypoints = [(10.0, 20.0, 1.0), (30.0, 40.0, 1.0)]
        slot.color = "Red"
        slot.waypoint_idx = 0
        env._update_slot_marker(slot)
        env.bng.debug.add_spheres.assert_called_once()
        kwargs = env.bng.debug.add_spheres.call_args.kwargs
        assert kwargs["coordinates"] == [(10.0, 20.0, 3.0)]
        assert kwargs["rgba_colors"] == [(1.0, 0.0, 0.0, 0.8)]
        assert slot.active_marker_id == "sphere-1"

    def test_update_slot_marker_removes_previous(self):
        env = _env()
        env.bng = MagicMock()
        env.bng.debug.add_spheres.return_value = ["new"]
        slot = env.slots[0]
        slot.waypoints = [(0.0, 0.0, 0.0)]
        slot.active_marker_id = "old"
        env._update_slot_marker(slot)
        env.bng.debug.remove_spheres.assert_called_once_with(["old"])
        assert slot.active_marker_id == "new"

    def test_update_slot_marker_noop_without_bng(self):
        env = _env()
        env.bng = None
        slot = env.slots[0]
        slot.waypoints = [(0.0, 0.0, 0.0)]
        env._update_slot_marker(slot)
        assert slot.active_marker_id is None
