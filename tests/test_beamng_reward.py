"""Tests for environments.beamng_reward — obs slicing via the real env callers.

The observation's kinematics block has 6 entries (speed, steering, heading_err,
lateral_err, damage_norm, dist_norm) and the perception block is followed by
waypoint hints / extra features. The obstacle-proximity penalty must read ONLY
the perception block — not the 6th kin entry, hints, or extras.
"""

import numpy as np
import pytest

import environments.beamng_reward as reward_mod
from environments.beamng import BeamNGDrivingEnv
from environments.beamng_multi import BeamNGMultiEnv, VehicleSlot


def _single_env():
    # __init__ only stores config; no BeamNG connection is opened.
    env = BeamNGDrivingEnv(beamng_home="unused")
    env.waypoints = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]
    env._last_dist = 50.0
    env._current_dist = 50.0
    return env


def _base_obs(n, *, speed=0.5, dist_norm=1.0):
    """Obs with moving vehicle, perfect alignment, clear perception block."""
    obs = np.zeros(n, dtype=np.float32)
    obs[0] = speed
    obs[5] = dist_norm
    obs[6:] = 1.0  # clear lidar bins (and any tail, unless a test overrides)
    return obs


# With speed=0.5, alignment=1 and zero progress/damage, the terms left are the
# speed-alignment reward (0.5 * 1.0 * SPEED_ALIGN_COEF = 1.5) minus the per-step
# time penalty.
CLEAN_REWARD = 1.5 - reward_mod.STEP_PENALTY


class TestSingleEnvRewardSlicing:
    def test_dist_entry_near_checkpoint_is_not_an_obstacle(self):
        env = _single_env()
        obs = _base_obs(env.n_states, dist_norm=0.05)  # 10 m from checkpoint
        reward, done = env._compute_reward(obs)
        assert reward == pytest.approx(CLEAN_REWARD)
        assert done is False

    def test_close_lidar_bin_still_penalised(self):
        env = _single_env()
        obs = _base_obs(env.n_states)
        obs[6] = 0.1  # real obstacle in the first lidar bin
        reward, _done = env._compute_reward(obs)
        assert reward == pytest.approx(CLEAN_REWARD - (1.0 - 0.1) * 5.0)


class _FakeAgent:
    pass


def _multi_slot(n_states):
    return VehicleSlot(
        name="ego_0",
        color="Red",
        agent=_FakeAgent(),
        save_path="outputs/dqn.pth",
        sensor="lidar",
        n_states=n_states,
        waypoints=[(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)],
        last_dist=50.0,
        current_dist=50.0,
    )


class TestMultiEnvRewardSlicing:
    def test_negative_hints_and_extras_are_not_obstacles(self):
        # 6 kin + 8 lidar + 2 hints*2 + body orientation tail = 20 states.
        slot = _multi_slot(20)
        env = BeamNGMultiEnv(slots=[slot], beamng_home="unused")
        obs = _base_obs(20)
        obs[14:] = [-0.5, 0.2, -0.1, 0.0, 0.0, 0.0]  # hints behind/left, flat pitch/roll
        reward, done = env.compute_reward(slot, obs)
        assert reward == pytest.approx(CLEAN_REWARD)
        assert done is False

    def test_close_lidar_bin_still_penalised(self):
        slot = _multi_slot(14)
        env = BeamNGMultiEnv(slots=[slot], beamng_home="unused")
        obs = _base_obs(14)
        obs[6] = 0.1
        reward, _done = env.compute_reward(slot, obs)
        assert reward == pytest.approx(CLEAN_REWARD - (1.0 - 0.1) * 5.0)
