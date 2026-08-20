"""Unit tests for environments.beamng_path — pure polyline projection."""

import numpy as np
import pytest

from environments.beamng_path import (
    NEUTRAL,
    SEARCH_WINDOW_M,
    path_length,
    project_onto_path,
    waypoint_arcs,
)

# A 100 m straight east, then 100 m north. The corner is what the old
# straight-line-to-checkpoint measure could not handle.
L_SHAPE = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (100.0, 100.0, 0.0)]

# A 400 m square that returns to its own start — the shape of gridmap_v2's
# default 1767 m training path. Its start and end are the same point, so an
# unseeded search near the line can pick either.
CLOSED_LOOP = [
    (0.0, 0.0, 0.0),
    (100.0, 0.0, 0.0),
    (100.0, 100.0, 0.0),
    (0.0, 100.0, 0.0),
    (0.0, 0.0, 0.0),
]
# Just short of the line, on the inside: nearer the closing segment than the
# opening one, so the global search reads it as almost a full lap.
BEHIND_THE_LINE = (-0.5, 0.5, 0.0)


class TestDegenerateInput:
    def test_empty_polyline_is_neutral(self):
        assert project_onto_path([], (5.0, 5.0, 0.0)) == NEUTRAL

    def test_single_point_is_neutral(self):
        # One point is not a line: there is no tangent and nothing to be offset from.
        assert project_onto_path([(0.0, 0.0, 0.0)], (5.0, 0.0, 0.0)) == NEUTRAL


class TestProgress:
    def test_on_the_line_progress_is_distance_travelled(self):
        assert project_onto_path(L_SHAPE, (40.0, 0.0, 0.0)).progress_m == pytest.approx(40.0)

    def test_progress_accumulates_through_the_corner(self):
        assert project_onto_path(L_SHAPE, (100.0, 50.0, 0.0)).progress_m == pytest.approx(150.0)

    def test_progress_is_monotone_all_the_way_round_the_corner(self):
        # The property the reward depends on: driving the path never reads as
        # backward progress, even where distance to the end point grows.
        route = [(x, 0.0, 0.0) for x in range(0, 101, 10)]
        route += [(100.0, y, 0.0) for y in range(10, 101, 10)]
        values = [project_onto_path(L_SHAPE, p).progress_m for p in route]
        assert values == sorted(values)
        assert values[0] < values[-1]

    def test_before_the_start_clamps_to_zero(self):
        assert project_onto_path(L_SHAPE, (-20.0, 0.0, 0.0)).progress_m == pytest.approx(0.0)

    def test_past_the_end_clamps_to_the_full_length(self):
        assert project_onto_path(L_SHAPE, (100.0, 400.0, 0.0)).progress_m == pytest.approx(200.0)


class TestCrossTrack:
    def test_on_the_line_is_zero(self):
        assert project_onto_path(L_SHAPE, (40.0, 0.0, 0.0)).cross_track_m == pytest.approx(0.0)

    def test_left_of_travel_is_positive(self):
        # Heading east along segment 0, +y is to the left.
        assert project_onto_path(L_SHAPE, (40.0, 3.0, 0.0)).cross_track_m == pytest.approx(3.0)

    def test_right_of_travel_is_negative(self):
        assert project_onto_path(L_SHAPE, (40.0, -3.0, 0.0)).cross_track_m == pytest.approx(-3.0)

    def test_sign_follows_the_segment_direction_not_the_world_axes(self):
        # On segment 1 the car heads north, so left is -x.
        assert project_onto_path(L_SHAPE, (95.0, 50.0, 0.0)).cross_track_m == pytest.approx(5.0)


class TestTangentAndSegment:
    def test_tangent_of_the_first_segment_is_east(self):
        assert project_onto_path(L_SHAPE, (40.0, 0.0, 0.0)).tangent_rad == pytest.approx(0.0)

    def test_tangent_of_the_second_segment_is_north(self):
        pos = project_onto_path(L_SHAPE, (100.0, 50.0, 0.0))
        assert pos.tangent_rad == pytest.approx(np.pi / 2)
        assert pos.segment_index == 1

    def test_segment_length_is_the_projected_segment(self):
        assert project_onto_path(L_SHAPE, (40.0, 0.0, 0.0)).segment_len_m == pytest.approx(100.0)

    def test_nearest_segment_wins_when_the_car_cuts_the_corner(self):
        # Inside the corner, equidistant-ish from both legs: the earlier segment wins.
        assert project_onto_path(L_SHAPE, (99.0, 1.0, 0.0)).segment_index == 0


class TestPathLength:
    def test_sums_the_segments(self):
        assert path_length(L_SHAPE) == pytest.approx(200.0)

    def test_degenerate_polylines_are_zero(self):
        assert path_length([]) == 0.0
        assert path_length([(1.0, 2.0, 3.0)]) == 0.0


class TestWaypointArcs:
    """Where each checkpoint sits along the path, which is what "reached it" means."""

    def test_one_arc_per_waypoint_not_per_vertex(self):
        # The guide line is [spawn, *waypoints], so the spawn's own 0.0 is dropped
        # and entry k lines up with waypoints[k].
        assert waypoint_arcs(L_SHAPE) == pytest.approx([100.0, 200.0])

    def test_the_last_arc_is_the_path_length(self):
        assert waypoint_arcs(L_SHAPE)[-1] == pytest.approx(path_length(L_SHAPE))

    def test_degenerate_polylines_have_no_arcs(self):
        assert waypoint_arcs([]) == []
        assert waypoint_arcs([(1.0, 2.0, 3.0)]) == []


class TestSeededSearch:
    """``near_m`` pins the search near where the car was last measured.

    Without it, a path that passes close to itself projects onto whichever part
    happens to be nearest. On a closed circuit that means a car at the
    start/finish line reads either arc 0 or a full lap, and since the reward pays
    PROGRESS_COEF x the *change* in arc length, that is worth 3 x the path length
    in one step — measured at +5301 on gridmap_v2's 1767 m default path, for a car
    that had not moved.
    """

    def test_an_unseeded_search_reads_almost_a_full_lap_at_the_line(self):
        # The defect itself, pinned so a regression is visible rather than subtle.
        assert project_onto_path(CLOSED_LOOP, BEHIND_THE_LINE).progress_m == pytest.approx(399.5)

    def test_seeded_at_the_start_it_stays_at_the_start(self):
        pos = project_onto_path(CLOSED_LOOP, BEHIND_THE_LINE, near_m=0.0)
        assert pos.progress_m == pytest.approx(0.0)

    def test_seeded_at_the_end_it_stays_at_the_end(self):
        # Same physical point, but a car on its final approach really is a lap in.
        pos = project_onto_path(CLOSED_LOOP, BEHIND_THE_LINE, near_m=395.0)
        assert pos.progress_m == pytest.approx(399.5)

    def test_the_window_outreaches_one_step_at_the_car_s_top_speed(self):
        # 81 m/s x 0.333 s = 27 m. A window narrower than a single step would make
        # a fast car outrun its own seed, freezing progress.
        assert SEARCH_WINDOW_M > 27.0

    def test_a_legitimate_step_ahead_is_still_found(self):
        pos = project_onto_path(CLOSED_LOOP, (27.0, 0.0, 0.0), near_m=0.0)
        assert pos.progress_m == pytest.approx(27.0)

    def test_a_stale_seed_falls_back_to_a_global_search(self):
        # A teleport (episode reset) or a car off the map must not leave progress
        # pinned to an arc it has no relationship to.
        pos = project_onto_path(CLOSED_LOOP, (100.0, 95.0, 0.0), near_m=0.0)
        assert pos.progress_m == pytest.approx(195.0)

    def test_a_seed_beyond_the_path_falls_back_rather_than_returning_nothing(self):
        pos = project_onto_path(CLOSED_LOOP, (50.0, 0.0, 0.0), near_m=1e6)
        assert pos.progress_m == pytest.approx(50.0)

    def test_seeding_does_not_disturb_an_unambiguous_path(self):
        # Seeds a step or two from the truth are the only kind a caller produces —
        # the seed is always the previous step's reading.
        for seed in (130.0, 150.0, 170.0, 200.0):
            assert project_onto_path(L_SHAPE, (100.0, 50.0, 0.0), near_m=seed).progress_m == (
                pytest.approx(150.0)
            )

    def test_a_seed_left_far_behind_holds_progress_back_by_design(self):
        # Documented consequence, not an oversight: seeded at 0 but standing 150 m
        # along, the projection reports the near end of its window (100 m) rather
        # than believing a 150 m jump in one step. No caller can produce this — the
        # seed is refreshed every step, and 150 m is six steps at top speed — and
        # accepting such jumps is the whole defect this window exists to stop.
        pos = project_onto_path(L_SHAPE, (100.0, 50.0, 0.0), near_m=0.0)
        assert pos.progress_m == pytest.approx(100.0)

    def test_progress_stays_monotone_round_the_loop_when_seeded(self):
        # The property the reward depends on: driving the circuit once never reads
        # as going backwards, including through the seam at the end.
        seen, seed = [], 0.0
        route = [(x, 0.0, 0.0) for x in range(0, 101, 10)]
        route += [(100.0, y, 0.0) for y in range(10, 101, 10)]
        route += [(x, 100.0, 0.0) for x in range(90, -1, -10)]
        route += [(0.0, y, 0.0) for y in range(90, -1, -10)]
        for pos in route:
            seed = project_onto_path(CLOSED_LOOP, pos, near_m=seed).progress_m
            seen.append(seed)
        assert seen == sorted(seen)
        assert seen[-1] == pytest.approx(path_length(CLOSED_LOOP))
