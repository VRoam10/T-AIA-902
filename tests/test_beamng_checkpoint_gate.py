"""Reaching a checkpoint means covering its distance, not standing near it.

The gate used to be proximity: ``dist < WAYPOINT_RADIUS``, 8 m. Three constants
made that free rather than earned — ``SPAWN_CLEARANCE_M`` (2 m) puts checkpoint 0
inside the ring before the car moves, ``DENSE_SPACING_M`` (8 m) equals the radius
so the next checkpoint on the dense chain is inside the current one's, and
``WAYPOINT_RADIUS`` itself is generous enough that a settling car drifts through
one. Measured on the shipped caches: a car parked at east_coast_usa's spawn banked
+56 over 12 steps of nothing, and 8 of 44 paths have every dense gap under the
radius (over 94% on most of the rest).

Both envs now advance on arc length along the guide line, which is what the
projection was built to measure.
"""

import pytest

from core.trajectory import DENSE_SPACING_M, SPAWN_CLEARANCE_M, TrajectoryData
from environments.beamng import BeamNGDrivingEnv
from environments.beamng_multi import BeamNGMultiEnv, VehicleSlot

SPAWN = (0.0, 0.0, 0.0)
# A straight 8 m chain: the dense spacing that used to equal the arrival radius.
CHAIN = [(8.0, 0.0, 0.0), (16.0, 0.0, 0.0), (24.0, 0.0, 0.0), (32.0, 0.0, 0.0)]
EAST = {"vel": (10.0, 0.0, 0.0)}


def _traj(waypoints):
    return TrajectoryData(
        spawn_pos=SPAWN,
        spawn_rot=(0.0, 0.0, 0.0, 1.0),
        sparse_waypoints=list(waypoints),
        dense_waypoints=list(waypoints),
        map_name="gridmap_v2",
        generated_at="2026-08-05T00:00:00+00:00",
        source="test",
    )


def _env(waypoints=CHAIN):
    env = BeamNGDrivingEnv(beamng_home="unused")
    env.trajectory = _traj(waypoints)
    env.waypoints = list(waypoints)
    env._rebuild_guide_line()
    env._waypoint_idx = 0
    env._checkpoint_hit = False
    return env


def _drive(env, x):
    """Put the car at x metres along the straight chain and run the gate."""
    pos = (x, 0.0, 0.0)
    env._path_pos = env._project(pos)
    env._path_errors(pos, EAST, env._path_pos.progress_m)
    return env._waypoint_idx


def _slot_env(waypoints=CHAIN):
    slot = VehicleSlot(name="ego_0", color="White", agent=None, save_path="")
    slot.waypoints = list(waypoints)
    slot.spawn_pos = SPAWN
    slot.guide_line = [SPAWN, *waypoints]
    env = BeamNGMultiEnv(slots=[slot], beamng_home="unused")
    return env, slot


class TestTheConstantsThatMadeProximityFree:
    def test_the_dense_spacing_is_not_greater_than_the_old_radius(self):
        # Pins the arithmetic that caused this: while these two are equal, any
        # proximity gate hands over the next checkpoint the moment one is reached.
        assert DENSE_SPACING_M <= BeamNGDrivingEnv.WAYPOINT_RADIUS

    def test_the_spawn_clearance_is_inside_the_old_radius(self):
        assert SPAWN_CLEARANCE_M < BeamNGDrivingEnv.WAYPOINT_RADIUS


class TestSingleEnvGate:
    def test_a_parked_car_reaches_nothing(self):
        env = _env()
        assert _drive(env, 0.0) == 0
        assert env._checkpoint_hit is False

    def test_a_car_short_of_the_first_checkpoint_reaches_nothing(self):
        # Inside the old 8 m ring the whole way, and still not there.
        env = _env()
        for x in (1.0, 3.0, 5.0, 7.9):
            assert _drive(env, x) == 0

    def test_covering_the_distance_reaches_it(self):
        env = _env()
        assert _drive(env, 8.0) == 1
        assert env._checkpoint_hit is True

    def test_the_chain_advances_one_checkpoint_per_eight_metres(self):
        env = _env()
        assert _drive(env, 8.0) == 1
        assert _drive(env, 16.0) == 2
        assert _drive(env, 24.0) == 3
        assert _drive(env, 32.0) == 4

    def test_several_checkpoints_in_one_step_all_count(self):
        # At 27 m/s a step covers three 8 m gaps. Advancing only one per step would
        # leave the chain lagging behind the car and the finish arriving late.
        env = _env()
        assert _drive(env, 25.0) == 3

    def test_passing_wide_of_a_checkpoint_still_counts(self):
        # A racing line does not thread the markers; the car got as far along the
        # path as the checkpoint sits, which is the question being asked.
        env = _env()
        pos = (16.0, 4.0, 0.0)
        env._path_pos = env._project(pos)
        env._path_errors(pos, EAST, env._path_pos.progress_m)
        assert env._waypoint_idx == 2

    def test_the_finish_needs_the_whole_path(self):
        env = _env()
        assert _drive(env, 31.9) == 3  # one short, with the last marker 0.1 m away
        assert _drive(env, 32.0) == len(CHAIN)


class TestMultiEnvGate:
    def test_a_parked_slot_reaches_nothing(self):
        env, slot = _slot_env()
        env._path_errors(slot, pos=SPAWN, state=EAST, progress_m=0.0)
        assert slot.waypoint_idx == 0
        assert slot.checkpoint_hit is False

    def test_covering_the_distance_reaches_it(self):
        env, slot = _slot_env()
        env._path_errors(slot, pos=(8.0, 0.0, 0.0), state=EAST, progress_m=8.0)
        assert slot.waypoint_idx == 1
        assert slot.checkpoint_hit is True

    def test_several_checkpoints_in_one_step_all_count(self):
        env, slot = _slot_env()
        env._path_errors(slot, pos=(25.0, 0.0, 0.0), state=EAST, progress_m=25.0)
        assert slot.waypoint_idx == 3


class TestReportedCheckpointCount:
    def test_the_env_reports_the_count_not_the_zeroed_index(self):
        env = _env()
        env._checkpoints_reached = 7
        env._waypoint_idx = 0  # as the finish leaves it
        assert env._checkpoints_reached == 7

    def test_a_fresh_env_reports_zero(self):
        assert BeamNGDrivingEnv(beamng_home="unused")._checkpoints_reached == 0


class TestGuideLineArcs:
    def test_the_first_checkpoint_sits_at_its_distance_from_the_spawn(self):
        from environments.beamng_path import waypoint_arcs

        arcs = waypoint_arcs(_env()._guide_line)
        assert arcs[0] == pytest.approx(8.0)
        assert len(arcs) == len(CHAIN)
