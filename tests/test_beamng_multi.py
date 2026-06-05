"""Tests for environments.beamng_multi."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from environments.beamng_multi import BeamNGMultiEnv, VehicleSlot, _color_rgba, build_slots


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
    {"algo": "dqn", "agent": _FakeAgent(), "vehicle_id": "taxi", "color": "Yellow",
     "save_path": "outputs/dqn.pth"},
    {"algo": "ddpg", "agent": _FakeAgent(), "vehicle_id": "ibishu_pigeon", "color": "Red",
     "save_path": "outputs/ddpg.pth"},
    {"algo": "td3", "agent": _FakeAgent(), "vehicle_id": "taxi", "color": "Blue",
     "save_path": "outputs/td3.pth"},
]


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
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]
        slot.waypoint_idx = 0
        state = {"vel": (1.0, 0.0, 0.0)}
        env._path_errors(slot, pos=(0.0, 0.0, 0.0), state=state)
        assert slot.waypoint_idx == 1
        assert slot.checkpoint_hit is True

    def test_default_reward_gives_checkpoint_bonus(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]  # reward_mode "default"
        slot.checkpoint_hit = True
        slot.waypoint_idx = 1
        slot.checkpoint_dist = 0.0
        obs = np.zeros(env.n_states, dtype=np.float32)
        obs[0] = 0.5  # moving (speed) so no stationary penalty
        reward, done = env.compute_reward(slot, obs)
        assert reward >= 100.0
        assert slot.checkpoint_hit is False  # consumed

    def test_default_reward_terminates_on_max_damage(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0)]
        slot = env.slots[0]
        obs = np.zeros(env.n_states, dtype=np.float32)
        obs[0] = 0.5
        obs[4] = 1.0  # damage_norm = 1.0 -> 1000 damage == MAX_DAMAGE
        reward, done = env.compute_reward(slot, obs)
        assert done is True

    def test_ddpg_reward_rewards_progress(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[1]  # reward_mode "ddpg"
        slot.last_dist = 50.0
        slot.current_dist = 40.0  # got 10 m closer
        obs = np.zeros(env.n_states, dtype=np.float32)
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
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]
        self._wire_slot_sensors(
            slot, speed=10.0, steering=0.0, damage=0.0,
            pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), lidar_points=None,
        )
        obs = env.observe(slot)
        assert obs.shape == (env.n_states,)
        assert obs[0] == pytest.approx(0.2, abs=1e-3)

    def test_observe_polls_each_slot_sensor(self):
        env = _env()
        env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
        slot = env.slots[0]
        self._wire_slot_sensors(
            slot, speed=0.0, steering=0.0, damage=0.0,
            pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), lidar_points=None,
        )
        env.observe(slot)
        slot.vehicle.poll_sensors.assert_called_once()
        slot.lidar.poll.assert_called_once()


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


class TestStartingGrid:
    def _env_with_spawn(self, n_slots, spawn_pos, spawn_rot):
        specs = [
            {"algo": "dqn", "agent": _FakeAgent(), "vehicle_id": "taxi",
             "color": "Yellow", "save_path": f"outputs/a{i}.pth"}
            for i in range(n_slots)
        ]
        env = BeamNGMultiEnv(slots=build_slots(specs), beamng_home="unused")
        env.trajectory = MagicMock()
        env.trajectory.spawn_pos = spawn_pos
        env.trajectory.spawn_rot = spawn_rot
        return env

    def test_grid_is_side_by_side_centered_on_spawn(self):
        # Identity spawn_rot -> forward +Y, right +X, so the line fans along X.
        env = self._env_with_spawn(4, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        gap = env.GRID_LANE_OFFSET
        xs = []
        for i in range(4):
            pos, rot = env._grid_pose(i)
            xs.append(pos[0])
            assert pos[1] == pytest.approx(0.0, abs=1e-6)  # all on the same line
            assert pos[2] == pytest.approx(0.0)
            assert rot == (0.0, 0.0, 0.0, 1.0)  # all face the same way
        # Centered: 4 cars at -1.5,-0.5,+0.5,+1.5 lane-widths
        assert xs == pytest.approx([-1.5 * gap, -0.5 * gap, 0.5 * gap, 1.5 * gap])

    def test_grid_poses_are_all_distinct(self):
        env = self._env_with_spawn(3, (10.0, 5.0, 1.0), (0.0, 0.0, 0.0, 1.0))
        poses = [env._grid_pose(i)[0] for i in range(3)]
        assert len({tuple(p) for p in poses}) == 3

    def test_grid_fans_along_right_axis_when_rotated(self):
        # spawn_rot for yaw=-pi/2 (forward +X) -> right axis is -Y, so the line
        # fans along Y instead of X.
        import math

        rot = (0.0, 0.0, math.sin(-math.pi / 4), math.cos(-math.pi / 4))
        env = self._env_with_spawn(2, (0.0, 0.0, 0.0), rot)
        p0, _ = env._grid_pose(0)
        p1, _ = env._grid_pose(1)
        # Both share X (fanned purely along Y), and differ in Y.
        assert p0[0] == pytest.approx(p1[0], abs=1e-6)
        assert p0[1] != pytest.approx(p1[1])


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
        env.waypoints = [(10.0, 20.0, 1.0), (30.0, 40.0, 1.0)]
        slot = env.slots[0]
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
        env.waypoints = [(0.0, 0.0, 0.0)]
        slot = env.slots[0]
        slot.active_marker_id = "old"
        env._update_slot_marker(slot)
        env.bng.debug.remove_spheres.assert_called_once_with(["old"])
        assert slot.active_marker_id == "new"

    def test_update_slot_marker_noop_without_bng(self):
        env = _env()
        env.bng = None
        env.waypoints = [(0.0, 0.0, 0.0)]
        slot = env.slots[0]
        env._update_slot_marker(slot)
        assert slot.active_marker_id is None
