"""Unit tests for environments.beamng_path — pure polyline projection."""

import numpy as np
import pytest

from environments.beamng_path import NEUTRAL, path_length, project_onto_path

# A 100 m straight east, then 100 m north. The corner is what the old
# straight-line-to-checkpoint measure could not handle.
L_SHAPE = [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (100.0, 100.0, 0.0)]


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
