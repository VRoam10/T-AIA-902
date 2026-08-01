"""Unit tests for the racing geometry helpers — pure math, no simulator.

`track_progress_m` orders two cars along a shared path (the race reward's gap
term depends on it); `starting_grid` keeps them from spawning inside each other.
"""

import numpy as np
import pytest

from environments.beamng_geometry import starting_grid, track_progress_m

# A straight 100 m path with checkpoints every 25 m.
STRAIGHT = [(0.0, 0.0, 0.0), (25.0, 0.0, 0.0), (50.0, 0.0, 0.0), (75.0, 0.0, 0.0), (100.0, 0.0, 0.0)]


class TestTrackProgress:
    def test_empty_waypoints_is_zero(self):
        assert track_progress_m([], 0, (0.0, 0.0, 0.0)) == 0.0

    def test_at_the_target_progress_equals_its_arc_length(self):
        assert track_progress_m(STRAIGHT, 2, (50.0, 0.0, 0.0)) == pytest.approx(50.0)

    def test_halfway_to_the_target_is_halfway_along_the_segment(self):
        # Between wp1 (25 m) and wp2 (50 m), sitting at x=37.5.
        assert track_progress_m(STRAIGHT, 2, (37.5, 0.0, 0.0)) == pytest.approx(37.5)

    def test_increases_monotonically_while_driving_forward(self):
        values = [track_progress_m(STRAIGHT, 1, (x, 0.0, 0.0)) for x in range(0, 25)]
        assert values == sorted(values)

    def test_is_continuous_across_a_checkpoint_transition(self):
        """The gap term telescopes, so a discontinuity here would inject a
        spurious one-off gap into the reward at every checkpoint."""
        # Just before reaching wp2, the target is wp2; just after, it becomes wp3.
        before = track_progress_m(STRAIGHT, 2, (49.9, 0.0, 0.0))
        after = track_progress_m(STRAIGHT, 3, (50.1, 0.0, 0.0))
        assert after - before == pytest.approx(0.2, abs=1e-6)

    def test_past_the_last_checkpoint_reports_the_full_length(self):
        assert track_progress_m(STRAIGHT, 5, (999.0, 0.0, 0.0)) == pytest.approx(100.0)

    def test_orders_two_cars_along_a_shared_path(self):
        leader = track_progress_m(STRAIGHT, 3, (70.0, 0.0, 0.0))
        chaser = track_progress_m(STRAIGHT, 2, (40.0, 0.0, 0.0))
        assert leader > chaser

    def test_follows_a_curved_path_by_arc_length_not_straight_line(self):
        # An L-shaped path: 10 m east, then 10 m north. At the corner, arc = 10.
        corner = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)]
        assert track_progress_m(corner, 1, (10.0, 0.0, 0.0)) == pytest.approx(10.0)
        assert track_progress_m(corner, 2, (10.0, 10.0, 0.0)) == pytest.approx(20.0)

    def test_single_waypoint_path_does_not_crash(self):
        assert track_progress_m([(0.0, 0.0, 0.0)], 0, (5.0, 0.0, 0.0)) == pytest.approx(-5.0)


class TestStartingGrid:
    IDENTITY = (0.0, 0.0, 0.0, 1.0)  # faces South in this project's convention

    def test_zero_racers_is_empty(self):
        assert starting_grid((0.0, 0.0, 0.0), self.IDENTITY, 0) == []

    def test_two_slots_straddle_the_centreline(self):
        slots = starting_grid((0.0, 0.0, 5.0), self.IDENTITY, 2, lateral_m=3.0)
        assert len(slots) == 2
        # Identity faces South (0, -1); facing south, the driver's left is East, so
        # slot 0 sits at +x and slot 1 mirrors it.
        assert slots[0][0] == pytest.approx(3.0)
        assert slots[1][0] == pytest.approx(-3.0)
        # Same row -> same longitudinal position.
        assert slots[0][1] == pytest.approx(slots[1][1])

    def test_slots_are_pairwise_separated(self):
        slots = starting_grid((10.0, -4.0, 2.0), self.IDENTITY, 4, lateral_m=3.0, stagger_m=6.0)
        for i, a in enumerate(slots):
            for b in slots[i + 1 :]:
                assert np.hypot(a[0] - b[0], a[1] - b[1]) > 2.0, "cars would spawn overlapping"

    def test_third_slot_starts_a_new_row_behind(self):
        slots = starting_grid((0.0, 0.0, 0.0), self.IDENTITY, 4, lateral_m=3.0, stagger_m=6.0)
        # Facing South, "behind" is +y.
        assert slots[2][1] == pytest.approx(slots[0][1] + 6.0)
        assert slots[3][1] == pytest.approx(slots[1][1] + 6.0)

    def test_grid_rotates_with_the_spawn_heading(self):
        # qz = sin(yaw/2) with yaw = pi/2 under the conjugated convention.
        rot = (0.0, 0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4))
        slots = starting_grid((0.0, 0.0, 0.0), rot, 2, lateral_m=3.0)
        # Facing East(-ish): the pair now separates along y rather than x.
        assert abs(slots[0][1] - slots[1][1]) == pytest.approx(6.0, abs=1e-6)
        assert slots[0][0] == pytest.approx(slots[1][0], abs=1e-6)

    def test_height_is_taken_from_the_spawn(self):
        slots = starting_grid((0.0, 0.0, 12.5), self.IDENTITY, 2)
        assert all(s[2] == pytest.approx(12.5) for s in slots)

    def test_row_spacing_exceeds_car_length(self):
        # The race car is ~4.5 m long; rows must not overlap front-to-back.
        slots = starting_grid((0.0, 0.0, 0.0), self.IDENTITY, 4, stagger_m=6.0)
        assert abs(slots[2][1] - slots[0][1]) > 4.5
