"""Unit tests for environments.beamng_features — pure observation blocks."""

import numpy as np
import pytest

from environments.beamng_features import (
    latest_road_reading,
    road_info_features,
    wheel_info_features,
)

HALF_TRACK = 0.7


def _road(**over):
    """A plausible on-road reading: 8 m wide road, car centred, straight ahead.

    Centerline points march 10 m apart straight down +x, which is also the car's
    heading in these tests, so vehicle-local forward == world +x.
    """
    reading = {
        "time": 1.0,
        "halfWidth": 4.0,
        "dist2Left": 4.0,
        "dist2Right": 4.0,
        "headingAngle": 0.0,
        "roadRadius": float("nan"),
        "xP0onCL": 10.0, "yP0onCL": 0.0,
        "xP1onCL": 20.0, "yP1onCL": 0.0,
        "xP2onCL": 30.0, "yP2onCL": 0.0,
        "xP3onCL": 40.0, "yP3onCL": 0.0,
    }
    reading.update(over)
    return reading


class TestLatestRoadReading:
    def test_none_is_empty(self):
        assert latest_road_reading(None) == {}

    def test_flat_reading_passes_through(self):
        assert latest_road_reading({"dist2Left": 1.0})["dist2Left"] == 1.0

    def test_index_map_picks_the_newest(self):
        payload = {0.0: {"dist2Left": 1.0, "time": 1.0}, 1.0: {"dist2Left": 2.0, "time": 2.0}}
        assert latest_road_reading(payload)["dist2Left"] == 2.0

    def test_list_picks_the_newest(self):
        payload = [{"dist2Left": 1.0, "time": 2.0}, {"dist2Left": 2.0, "time": 1.0}]
        assert latest_road_reading(payload)["dist2Left"] == 1.0


class TestRoadInfoBlock:
    def test_width_is_six(self):
        out = road_info_features(_road(), HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        assert out.shape == (6,)
        assert out.dtype == np.float32

    def test_missing_payload_is_all_neutral(self):
        out = road_info_features(None, HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        np.testing.assert_allclose(out, [0.0] * 6, atol=1e-6)

    def test_centred_on_road_reads_both_edges_far(self):
        left, right = road_info_features(_road(), HALF_TRACK, (0.0, 0.0, 0.0), 0.0)[:2]
        assert left == pytest.approx((4.0 - HALF_TRACK) / 4.0, abs=1e-4)
        assert right == pytest.approx((4.0 - HALF_TRACK) / 4.0, abs=1e-4)

    def test_wheel_over_the_left_edge_reads_negative(self):
        out = road_info_features(_road(dist2Left=-1.0), HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        assert out[0] == pytest.approx((-1.0 - HALF_TRACK) / 4.0, abs=1e-4)

    def test_road_heading_is_normalized_by_a_quarter_turn(self):
        out = road_info_features(_road(headingAngle=np.pi / 4), HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        assert out[2] == pytest.approx(0.5, abs=1e-4)

    def test_road_heading_saturates(self):
        out = road_info_features(_road(headingAngle=np.pi), HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        assert out[2] == pytest.approx(1.0, abs=1e-6)

    def test_straight_road_has_zero_curvature(self):
        # The sensor reports NaN radius for a straight road.
        out = road_info_features(_road(), HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        assert out[3] == pytest.approx(0.0, abs=1e-6)

    def test_left_hand_hairpin_is_plus_one(self):
        left_bend = _road(
            roadRadius=50.0,
            xP1onCL=20.0, yP1onCL=2.0,
            xP2onCL=29.0, yP2onCL=8.0,
            xP3onCL=36.0, yP3onCL=16.0,
        )
        assert road_info_features(left_bend, HALF_TRACK, (0.0, 0.0, 0.0), 0.0)[3] == pytest.approx(1.0)

    def test_right_hand_hairpin_is_minus_one(self):
        right_bend = _road(
            roadRadius=50.0,
            xP1onCL=20.0, yP1onCL=-2.0,
            xP2onCL=29.0, yP2onCL=-8.0,
            xP3onCL=36.0, yP3onCL=-16.0,
        )
        assert road_info_features(right_bend, HALF_TRACK, (0.0, 0.0, 0.0), 0.0)[3] == pytest.approx(-1.0)

    def test_gentle_sweeper_reads_small(self):
        sweeper = _road(
            roadRadius=500.0,
            xP1onCL=20.0, yP1onCL=0.4,
            xP2onCL=30.0, yP2onCL=1.2,
            xP3onCL=40.0, yP3onCL=2.4,
        )
        assert road_info_features(sweeper, HALF_TRACK, (0.0, 0.0, 0.0), 0.0)[3] == pytest.approx(0.1, abs=1e-3)

    def test_curvature_needs_three_points_to_know_its_sign(self):
        # A magnitude with an unknown direction cannot tell a policy which way to
        # turn, so a degenerate reading reads as straight.
        two_points = _road(roadRadius=50.0)
        for key in ("xP2onCL", "yP2onCL", "xP3onCL", "yP3onCL"):
            two_points.pop(key)
        assert road_info_features(two_points, HALF_TRACK, (0.0, 0.0, 0.0), 0.0)[3] == pytest.approx(0.0)

    def test_preview_point_is_the_farthest_ahead_in_vehicle_local_metres(self):
        out = road_info_features(_road(), HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        assert out[4] == pytest.approx(40.0 / 50.0, abs=1e-4)  # P3 at 40 m ahead
        assert out[5] == pytest.approx(0.0, abs=1e-4)

    def test_preview_point_is_de_rotated_into_the_vehicle_frame(self):
        # Car heading north (pi/2): the same world points are now to its right.
        out = road_info_features(_road(), HALF_TRACK, (0.0, 0.0, 0.0), np.pi / 2)
        assert out[4] == pytest.approx(0.0, abs=1e-4)
        assert out[5] == pytest.approx(-40.0 / 50.0, abs=1e-4)

    def test_preview_saturates_beyond_the_norm_distance(self):
        far = _road(xP3onCL=500.0)
        assert road_info_features(far, HALF_TRACK, (0.0, 0.0, 0.0), 0.0)[4] == pytest.approx(1.0)

    def test_points_all_behind_give_no_preview(self):
        behind = _road(
            xP0onCL=-10.0, xP1onCL=-20.0, xP2onCL=-30.0, xP3onCL=-40.0
        )
        out = road_info_features(behind, HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        assert out[4] == pytest.approx(0.0)
        assert out[5] == pytest.approx(0.0)

    def test_non_finite_fields_fall_back_to_neutral(self):
        broken = _road(dist2Left=float("inf"), headingAngle=float("nan"), halfWidth=0.0)
        out = road_info_features(broken, HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        assert np.all(np.isfinite(out))
        assert out[2] == pytest.approx(0.0)

    def test_reads_the_index_map_poll_shape(self):
        flat = road_info_features(_road(), HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        wrapped = road_info_features({0.0: _road()}, HALF_TRACK, (0.0, 0.0, 0.0), 0.0)
        np.testing.assert_allclose(flat, wrapped, atol=1e-6)


class TestWheelInfoBlock:
    def test_width_is_four(self):
        out = wheel_info_features({}, {}, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert out.shape == (4,)
        assert out.dtype == np.float32

    def test_stationary_is_all_neutral(self):
        out = wheel_info_features({"wheelspeed": 0.0}, {}, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        np.testing.assert_allclose(out, [0.0] * 4, atol=1e-6)

    def test_wheelspin_is_positive_slip(self):
        out = wheel_info_features({"wheelspeed": 20.0}, {}, (10.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert out[0] == pytest.approx(1.0)

    def test_partial_wheelspin_is_proportional(self):
        out = wheel_info_features({"wheelspeed": 22.0}, {}, (20.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert out[0] == pytest.approx(0.1, abs=1e-4)

    def test_lockup_is_negative_slip(self):
        out = wheel_info_features({"wheelspeed": 0.0}, {}, (20.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert out[0] == pytest.approx(-1.0)

    def test_slip_is_damped_at_crawling_speed(self):
        # Dividing by a 0.5 m/s ground speed would read as full wheelspin; the
        # reference speed keeps a rolling start from saturating the feature.
        out = wheel_info_features({"wheelspeed": 1.0}, {}, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert out[0] == pytest.approx(0.1, abs=1e-4)

    def test_sliding_left_is_positive_slip_angle(self):
        # Nose points +x, the car actually travels 45 deg to its left.
        out = wheel_info_features({"wheelspeed": 20.0}, {}, (14.1, 14.1, 0.0), (1.0, 0.0, 0.0))
        assert out[1] == pytest.approx(0.5, abs=1e-3)

    def test_sliding_right_is_negative_slip_angle(self):
        out = wheel_info_features({"wheelspeed": 20.0}, {}, (14.1, -14.1, 0.0), (1.0, 0.0, 0.0))
        assert out[1] == pytest.approx(-0.5, abs=1e-3)

    def test_slip_angle_is_zero_below_walking_pace(self):
        out = wheel_info_features({"wheelspeed": 0.5}, {}, (0.3, 0.3, 0.0), (1.0, 0.0, 0.0))
        assert out[1] == pytest.approx(0.0)

    def test_slip_angle_is_zero_when_travelling_where_the_nose_points(self):
        # Any heading, not just +x: a car facing +y and driving +y is not sliding.
        out = wheel_info_features({"wheelspeed": 20.0}, {}, (0.0, 20.0, 0.0), (0.0, 1.0, 0.0))
        assert out[1] == pytest.approx(0.0)

    def test_abs_flag_reads_the_renamed_electrics_key(self):
        out = wheel_info_features(
            {"wheelspeed": 10.0, "abs_active": True}, {}, (10.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        assert out[2] == pytest.approx(1.0)

    def test_abs_flag_accepts_a_numeric_state(self):
        out = wheel_info_features(
            {"wheelspeed": 10.0, "abs_active": 1}, {}, (10.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        assert out[2] == pytest.approx(1.0)

    def test_lateral_g_prefers_gx2_and_normalizes(self):
        out = wheel_info_features(
            {"wheelspeed": 10.0}, {"gx2": 0.75, "gx": 99.0}, (10.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        assert out[3] == pytest.approx(0.5, abs=1e-4)

    def test_lateral_g_falls_back_to_gx(self):
        out = wheel_info_features(
            {"wheelspeed": 10.0}, {"gx": -1.5}, (10.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        assert out[3] == pytest.approx(-1.0)

    def test_missing_gforces_is_zero(self):
        out = wheel_info_features({"wheelspeed": 10.0}, None, (10.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        assert out[3] == pytest.approx(0.0)
