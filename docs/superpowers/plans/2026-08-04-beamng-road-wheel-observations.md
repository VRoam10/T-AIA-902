# Road position, wheel performance, and path-relative pace — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the policy two new optional observation blocks — where the car sits on the road, and how the tyres are coping — and make the reward's pace terms measure progress along the path so that checkpoints hundreds of metres apart become trainable.

**Architecture:** Two pure-math modules (`environments/beamng_path.py` for polyline projection, `environments/beamng_features.py` for the sensor-payload → feature-block math) feed the existing envs. `road_info` (6 features, from a `RoadsSensor`) replaces the disabled `wheel_terrain` flag; `wheel_info` (4 features, from Electrics + vehicle state + a `GForces` sensor) is new. Both are booleans, default false, appended at the observation tail, and both contribute a token to the derived checkpoint path so configs cannot overwrite each other.

**Tech Stack:** Python 3.11 + numpy + pytest (run through `.venv`), beamngpy 1.3x sensors, TypeScript TUI on bun.

**Spec:** [`docs/superpowers/specs/2026-08-04-beamng-road-wheel-observations-design.md`](../specs/2026-08-04-beamng-road-wheel-observations-design.md)

## Global Constraints

- **Interpreter:** always the project venv — `.venv/Scripts/python.exe`. The system Python has torch but no beamngpy, so it silently breaks BeamNG imports.
- **Test command:** `.venv/Scripts/python.exe -m pytest -q` (full suite is ~30 s, 489 tests green at plan time). TUI: `bun test` and `bun run typecheck`, both run from `tui/`.
- **No simulator in unit tests.** Every test in this plan runs without BeamNG, using plain dicts, `MagicMock`, or the fake-`bng` doubles already in `tests/`.
- **New observation features default OFF.** `road_info=False`, `wheel_info=False` everywhere, so every existing `.pth` keeps loading at its current width.
- **Feature widths live in one place:** `beamng_spec.ROAD_FEATURES` / `WHEEL_FEATURES`, consumed by `beamng_spec.obs_size`. No other module may compute an observation width.
- **Every feature fails soft.** A missing payload, missing key, or non-finite value yields that feature's neutral value. A sensor hiccup must never raise inside `observe`.
- **Normalization constants** (exact values, used verbatim): `CURV_NORM_M = 50.0`, `ROAD_AHEAD_NORM_M = 50.0`, `SLIP_REF_MS = 5.0`, `SLIP_ANGLE_MIN_SPEED_MS = 1.0`, `LAT_G_NORM = 1.5`, `SEGMENT_TIME_BONUS = 25.0`.
- **Save-path token order** (exact): `_h{n}` then `_ori` then `_road` then `_whl`.
- **Commits:** subject line only — no body, no `Co-Authored-By` trailer (matches this repo's history). Commit at the end of each task; never squash, never push.

---

### Task 1: Polyline projection module

**Files:**
- Create: `environments/beamng_path.py`
- Test: `tests/test_beamng_path.py`

**Interfaces:**
- Consumes: nothing (pure numpy).
- Produces: `PathPosition` (frozen dataclass with `progress_m: float`, `cross_track_m: float`, `tangent_rad: float`, `segment_index: int`, `segment_len_m: float`), `NEUTRAL: PathPosition`, `project_onto_path(polyline, pos) -> PathPosition`, `path_length(polyline) -> float`. Tasks 6 and 7 depend on these exact names.

- [ ] **Step 1: Write the failing test**

Create `tests/test_beamng_path.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_path.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'environments.beamng_path'`

- [ ] **Step 3: Write the implementation**

Create `environments/beamng_path.py`:

```python
"""Where a car sits along its path: arc length, cross-track error and tangent.

Checkpoints answer "which target is next". This module answers "where am I on the
line between them", which is the question a reward can differentiate over a
kilometre-long segment. Everything here is a function of position alone, so the
numbers never jump when the target checkpoint advances, and following the road
round a bend can never read as backward progress — which is exactly what
"straight-line distance to the next checkpoint" got wrong.

Pure math: no beamngpy, no ``self``, no logging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathPosition:
    """A projection result, in the terms the envs and the reward consume."""

    progress_m: float  # arc length from the polyline start to the projection point
    cross_track_m: float  # signed perpendicular offset; + = left of travel
    tangent_rad: float  # heading of the segment projected onto
    segment_index: int  # which segment that was (0 = first)
    segment_len_m: float  # its length


# What "no usable path" reads as. Zeros are already how the observation and the
# reward spell "no data", so a degenerate polyline needs no special casing upstream.
NEUTRAL = PathPosition(0.0, 0.0, 0.0, 0, 0.0)


def path_length(polyline) -> float:
    """Total XY length of a polyline. 0.0 for fewer than two points."""
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return 0.0
    return float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())


def project_onto_path(polyline, pos) -> PathPosition:
    """Project ``pos`` onto ``polyline`` in the XY plane and describe where it landed.

    The segment chosen is the one whose clamped perpendicular distance is smallest,
    so a car cutting a corner still projects onto that corner; ties go to the
    earlier segment. Positions before the start or past the end clamp to the ends,
    which is what keeps progress bounded by the path length.

    Returns :data:`NEUTRAL` for an empty or single-point polyline.

    Caveat for closed circuits: a track that passes close to itself can project
    onto the wrong lap of the same geometry. Laps are 1 today; lap counting belongs
    to the caller, which adds ``laps_done * path_length(polyline)``.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 2:
        return NEUTRAL

    a = pts[:-1, :2]
    seg = pts[1:, :2] - a
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    # A repeated point would divide by zero; its t collapses to 0 and its cross
    # product is 0, so substituting 1.0 keeps the arithmetic finite and harmless.
    safe_len = np.where(seg_len > 0.0, seg_len, 1.0)

    p = np.array([float(pos[0]), float(pos[1])], dtype=np.float64)
    rel = p - a
    t = np.clip(
        (rel[:, 0] * seg[:, 0] + rel[:, 1] * seg[:, 1]) / (safe_len * safe_len), 0.0, 1.0
    )
    foot = a + seg * t[:, None]
    i = int(np.argmin(np.hypot(p[0] - foot[:, 0], p[1] - foot[:, 1])))

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    cross = seg[i, 0] * rel[i, 1] - seg[i, 1] * rel[i, 0]
    return PathPosition(
        progress_m=float(cum[i] + t[i] * seg_len[i]),
        cross_track_m=float(cross / safe_len[i]),
        tangent_rad=float(np.arctan2(seg[i, 1], seg[i, 0])),
        segment_index=i,
        segment_len_m=float(seg_len[i]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_path.py -q`
Expected: PASS (19 tests)

- [ ] **Step 5: Commit**

```bash
git add environments/beamng_path.py tests/test_beamng_path.py
git commit -m "feat: add polyline projection for path progress and cross-track error"
```

---

### Task 2: Road and wheel feature blocks (pure math)

**Files:**
- Create: `environments/beamng_features.py`
- Test: `tests/test_beamng_features.py`

**Interfaces:**
- Consumes: nothing (pure numpy). The `_latest_road_reading` logic is copied from `environments/beamng_geometry.py:300-325` and made public as `latest_road_reading`; the geometry module keeps its copy until Task 3 deletes it.
- Produces: `road_info_features(roads_payload, half_track_width, pos, heading) -> np.ndarray` (6 float32), `wheel_info_features(electrics, gforces, vel, dir_vec) -> np.ndarray` (4 float32), `latest_road_reading(payload) -> dict`, and the module constants `CURV_NORM_M`, `ROAD_AHEAD_NORM_M`, `SLIP_REF_MS`, `SLIP_ANGLE_MIN_SPEED_MS`, `LAT_G_NORM`. Tasks 3 and 4 call these.

- [ ] **Step 1: Write the failing test**

Create `tests/test_beamng_features.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'environments.beamng_features'`

- [ ] **Step 3: Write the implementation**

Create `environments/beamng_features.py`:

```python
"""Optional observation blocks, built from raw sensor payloads.

Two pure functions, one per opt-in flag:

  * ``road_info_features``  — where the car sits on the road, and where the road
    goes next. Six values from one ``RoadsSensor`` poll.
  * ``wheel_info_features`` — whether the tyres are coping: slip, slide, ABS and
    cornering load. Four values from Electrics, the vehicle state and ``GForces``.

Both return fixed-width float32 blocks and both fail soft: a missing payload, a
missing key or a non-finite value yields that feature's neutral value rather than
raising. An observation is built every step, and a sensor hiccup must never end an
episode.

The widths live in :mod:`environments.beamng_spec` (``ROAD_FEATURES`` /
``WHEEL_FEATURES``), which is the only place observation arithmetic is done.
"""

from __future__ import annotations

import math

import numpy as np

# --- Road ---------------------------------------------------------------------
# A 50 m radius bend reads as maximum curvature; a 500 m sweeper reads 0.1.
CURV_NORM_M = 50.0
# The look-ahead centerline point is expressed in units of this distance.
ROAD_AHEAD_NORM_M = 50.0

# --- Wheels -------------------------------------------------------------------
# Slip divides by ground speed, floored at this value: at a standstill the ratio
# would otherwise saturate on any wheel movement at all.
SLIP_REF_MS = 5.0
# Below this ground speed the velocity vector is noise, so slip angle reads 0.
SLIP_ANGLE_MIN_SPEED_MS = 1.0
# 1.5 g of lateral load is taken as "all the grip there is".
LAT_G_NORM = 1.5

_CENTERLINE_KEYS = (
    ("xP0onCL", "yP0onCL"),
    ("xP1onCL", "yP1onCL"),
    ("xP2onCL", "yP2onCL"),
    ("xP3onCL", "yP3onCL"),
)


def _finite(value, default=None):
    """``value`` as a finite float, or ``default``.

    Sensor payloads arrive from the simulator: a field can be absent, a string
    placeholder, or NaN/inf (the RoadsSensor documents NaN for a straight road).
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _truthy(value) -> bool:
    """Whether a game-side flag is set, across bool / number / string spellings."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    number = _finite(value, 0.0)
    return bool(number)


def latest_road_reading(roads_payload) -> dict:
    """Normalize any RoadsSensor poll shape to a single, most-recent reading dict.

    The default (GE bulk) poll returns an index->reading map keyed by reading
    index — ``{0.0: {...}, 1.0: {...}}`` — with the fields (dist2Left, halfWidth,
    ...) nested one level down. The send-immediately / ad-hoc paths return either
    a flat reading dict or a list of them. This collapses all three to one
    reading, picking the latest by ``time``, and returns ``{}`` when there is no
    usable reading. (Reading the index-map's top level finds none of the expected
    keys, which is why the feature it feeds was once stuck at neutral.)
    """
    if roads_payload is None:
        return {}
    if isinstance(roads_payload, dict):
        # A flat single reading already carries the fields at the top level.
        if any(k in roads_payload for k in ("dist2Left", "dist2Right", "halfWidth")):
            return roads_payload
        readings = [v for v in roads_payload.values() if isinstance(v, dict)]
    elif isinstance(roads_payload, list):
        readings = [v for v in roads_payload if isinstance(v, dict)]
    else:
        return {}
    if not readings:
        return {}
    return max(readings, key=lambda r: r.get("time", 0.0))


def _centerline_local(reading: dict, pos, heading: float) -> list[tuple[float, float]]:
    """The reading's centerline points as vehicle-local (forward, left), by distance.

    The sensor documents P0..P3 as the four *closest* centerline points, not four
    points ahead, so the caller cannot assume an order — they are sorted here by
    their forward component. Unreadable points are dropped.
    """
    cos_h = math.cos(-heading)
    sin_h = math.sin(-heading)
    out = []
    for x_key, y_key in _CENTERLINE_KEYS:
        x = _finite(reading.get(x_key))
        y = _finite(reading.get(y_key))
        if x is None or y is None:
            continue
        rel_x = x - float(pos[0])
        rel_y = y - float(pos[1])
        out.append((rel_x * cos_h - rel_y * sin_h, rel_x * sin_h + rel_y * cos_h))
    return sorted(out, key=lambda p: p[0])


def _signed_curvature(reading: dict, ahead: list[tuple[float, float]]) -> float:
    """Curvature magnitude from ``roadRadius``, sign from the centerline shape.

    The sign comes from the turn direction of the three farthest points rather
    than from a single point's lateral offset: that offset also contains the car's
    own displacement from the centerline, which would read a straight road as
    curved whenever the car runs wide. With fewer than three usable points the
    direction is unknown, and a magnitude without a direction cannot tell a policy
    which way to turn — so it reads as straight.
    """
    radius = _finite(reading.get("roadRadius"))
    if radius is None or radius <= 0.0 or len(ahead) < 3:
        return 0.0
    magnitude = float(np.clip(CURV_NORM_M / radius, 0.0, 1.0))
    (x0, y0), (x1, y1), (x2, y2) = ahead[-3:]
    cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
    if cross == 0.0:
        return 0.0
    return magnitude if cross > 0.0 else -magnitude


def road_info_features(roads_payload, half_track_width: float, pos, heading: float) -> np.ndarray:
    """Six road-relative features from one RoadsSensor poll.

    ``[edge_left, edge_right, road_heading, curvature, ahead_fwd, ahead_left]``

    * edges: +1 = well on road, 0 = a wheel at the edge, -1 = off road, measured
      at the front-axle midpoint in units of the road's half width.
    * road_heading: the car's yaw relative to the road reference line, over a
      quarter turn. This is the feature that does not care how far away the next
      checkpoint is.
    * curvature: signed, + = the road bends left. See :func:`_signed_curvature`.
    * ahead: the farthest centerline point in front of the car, in vehicle-local
      (forward, left) over ``ROAD_AHEAD_NORM_M``.

    All zeros when there is no usable reading.
    """
    reading = latest_road_reading(roads_payload)
    half_w = max(_finite(reading.get("halfWidth"), 3.0) or 3.0, 0.5)
    d_left = _finite(reading.get("dist2Left"), half_track_width)
    d_right = _finite(reading.get("dist2Right"), half_track_width)
    left = np.clip((d_left - half_track_width) / half_w, -1.0, 1.0)
    right = np.clip((d_right - half_track_width) / half_w, -1.0, 1.0)

    road_heading = np.clip(
        (_finite(reading.get("headingAngle"), 0.0) or 0.0) / (math.pi / 2.0), -1.0, 1.0
    )

    ahead = _centerline_local(reading, pos, heading)
    curvature = _signed_curvature(reading, ahead)

    in_front = [p for p in ahead if p[0] > 0.0]
    if in_front:
        far_fwd, far_left = in_front[-1]
        ahead_fwd = np.clip(far_fwd / ROAD_AHEAD_NORM_M, -1.0, 1.0)
        ahead_left = np.clip(far_left / ROAD_AHEAD_NORM_M, -1.0, 1.0)
    else:
        ahead_fwd = ahead_left = 0.0

    return np.array(
        [left, right, road_heading, curvature, ahead_fwd, ahead_left], dtype=np.float32
    )


def wheel_info_features(electrics, gforces, vel, dir_vec) -> np.ndarray:
    """Four grip features from Electrics, the vehicle state and GForces.

    ``[long_slip, slip_angle, abs_active, lat_g]``

    * long_slip: wheel speed minus ground speed, over ground speed. + = wheelspin,
      - = lockup. Ground speed comes from the state vector rather than
      ``electrics.airspeed``: it is already polled and unambiguous.
    * slip_angle: the angle between where the car points and where it is going,
      over a quarter turn. + = travelling to the left of the nose.
    * abs_active: 1.0 while ABS is intervening. The only aid fitted on the race
      car — its config deletes ESC and TC, so those flags would be constant zero.
    * lat_g: lateral load over ``LAT_G_NORM``. GForces is a raw passthrough with no
      documented keys, so the lateral axis is read as ``gx2`` then ``gx``; this
      project's vehicle frame has forward = -Y, which makes x the lateral axis.
    """
    elec = electrics or {}
    forces = gforces or {}

    ground = float(math.hypot(_finite(vel[0], 0.0) or 0.0, _finite(vel[1], 0.0) or 0.0))
    wheelspeed = _finite(elec.get("wheelspeed"), 0.0) or 0.0
    long_slip = np.clip((wheelspeed - ground) / max(ground, SLIP_REF_MS), -1.0, 1.0)

    slip_angle = 0.0
    if ground >= SLIP_ANGLE_MIN_SPEED_MS:
        heading = math.atan2(_finite(dir_vec[1], 0.0) or 0.0, _finite(dir_vec[0], 1.0) or 1.0)
        course = math.atan2(vel[1], vel[0])
        delta = (course - heading + math.pi) % (2.0 * math.pi) - math.pi
        slip_angle = float(np.clip(delta / (math.pi / 2.0), -1.0, 1.0))

    abs_active = 1.0 if _truthy(elec.get("abs_active")) else 0.0

    lat = 0.0
    for key in ("gx2", "gx"):
        value = _finite(forces.get(key))
        if value is not None:
            lat = value
            break
    lat_g = np.clip(lat / LAT_G_NORM, -1.0, 1.0)

    return np.array([long_slip, slip_angle, abs_active, lat_g], dtype=np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_features.py -q`
Expected: PASS (33 tests)

- [ ] **Step 5: Run the full suite to confirm nothing else moved**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS — 489 existing + the new files

- [ ] **Step 6: Commit**

```bash
git add environments/beamng_features.py tests/test_beamng_features.py
git commit -m "feat: add road-position and wheel-performance feature blocks"
```

---

### Task 3: `road_info` replaces `wheel_terrain` (Python end to end)

**Files:**
- Modify: `environments/beamng_spec.py:133-151` (`obs_size`), plus a new `ROAD_FEATURES` constant near `KINEMATIC_FEATURES`
- Modify: `environments/beamng.py` (`__init__` kwarg, `_attach_roads_sensor`, `_wheel_terrain_features` → `_road_info_features`, `_extra_features`, `_observe`, `_format_observation_lines`)
- Modify: `environments/beamng_multi.py` (`VehicleSlot.wheel_terrain`, `slot_n_states`, `build_slots`, `_create_slot_sensor`, `_slot_extra_features`, `observe`)
- Modify: `environments/beamng_race.py:34-76` (`build_race_slots`)
- Modify: `environments/beamng_geometry.py:300-343` (delete `_latest_road_reading` and `wheel_terrain_features`)
- Modify: `environments/__init__.py:19-51` (`_make_beamng`)
- Modify: `core/pipeline_actions.py` (`BeamNGOptions:88`, `RacerOptions:169`, `_beamng_kwargs:310`, `obs_size` calls at `:281`, `:454`, `:650`, `:708`)
- Test: `tests/test_beamng_spec.py`, `tests/test_beamng.py:150-190`, `tests/test_beamng_multi.py:10,350-365`, `tests/test_beamng_geometry.py:190-232` (delete the moved tests)

**Interfaces:**
- Consumes: `road_info_features` from Task 2.
- Produces: `beamng_spec.ROAD_FEATURES = 6`; `obs_size(sensor, trajectory_hints=0, body_orientation=False, road_info=False)`; env kwarg `road_info: bool = False`; `VehicleSlot.road_info`; `slot_n_states(sensor, trajectory_hints=0, body_orientation=False, road_info=False)`; `BeamNGOptions.road_info`; `RacerOptions.road_info`. Task 4 adds `wheel_info` alongside each of these.

- [ ] **Step 1: Write the failing tests**

In `tests/test_beamng_spec.py`, replace the `wheel_terrain` sizing assertions with:

```python
class TestRoadInfoSizing:
    def test_road_info_adds_six(self):
        base = obs_size("lidar")
        assert obs_size("lidar", road_info=True) == base + 6

    def test_road_info_stacks_with_the_other_tails(self):
        assert obs_size("lidar", 2, True, True) == 14 + 4 + 2 + 6

    def test_wheel_terrain_is_gone(self):
        import pytest

        with pytest.raises(TypeError):
            obs_size("lidar", wheel_terrain=True)
```

In `tests/test_beamng.py`, replace `test_extra_features_order_is_orientation_then_terrain` and the `_wheel_terrain_features` test with:

```python
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
```

In `tests/test_beamng_multi.py`, change the import at line 10 to
`from environments.beamng_features import road_info_features` (keeping
`from environments.beamng_geometry import body_orientation_features`), and replace the
`expected_wheel` assertion block at line 358 with:

```python
        expected_road = road_info_features(
            slot.roads_sensor.poll(), BeamNGMultiEnv.HALF_TRACK_WIDTH, slot.current_pos, 0.0
        )
        np.testing.assert_allclose(obs[-6:], expected_road, atol=1e-6)
```

Delete the five `wheel_terrain_features` tests in `tests/test_beamng_geometry.py:190-232` — Task 2's `tests/test_beamng_features.py` covers that math, including the cases those tests held.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_spec.py tests/test_beamng.py -q`
Expected: FAIL — `TypeError: obs_size() got an unexpected keyword argument 'road_info'` and `BeamNGDrivingEnv.__init__() got an unexpected keyword argument 'road_info'`

- [ ] **Step 3: Update the spec module**

In `environments/beamng_spec.py`, after `KINEMATIC_FEATURES` (line 31), add:

```python
# Optional observation tails, in the order _observe concatenates them.
# road:  [edge_left, edge_right, road_heading, curvature, ahead_fwd, ahead_left]
ROAD_FEATURES = 6
```

Then replace `obs_size` (lines 133-151) with:

```python
def obs_size(
    sensor: str,
    trajectory_hints: int = 0,
    body_orientation: bool = False,
    road_info: bool = False,
) -> int:
    """Observation length for a sensor plus the optional observation flags.

    Layout (blocks appended in this order, matching ``_observe``):

        kinematic(6) | perception(P) | hints(2*H) | [pitch, roll]? | road(6)?
    """
    return (
        KINEMATIC_FEATURES
        + perception_features(sensor)
        + 2 * int(trajectory_hints)
        + (2 if body_orientation else 0)
        + (ROAD_FEATURES if road_info else 0)
    )
```

- [ ] **Step 4: Update the single-vehicle env**

In `environments/beamng.py`:

Change the import at line 19 to keep only the orientation helper and add the new module:

```python
from environments.beamng_features import road_info_features
from environments.beamng_geometry import body_orientation_features
```

In `__init__`, rename the kwarg (line 121) and the attribute (line 180):

```python
        road_info: bool = False,
```
```python
        self.road_info = road_info
```

and update the sizing call (lines 186-188):

```python
        self.n_states = beamng_spec.obs_size(sensor, trajectory_hints, body_orientation, road_info)
```

Replace `_attach_roads_sensor` (lines 448-453):

```python
    def _attach_roads_sensor(self):
        """Attach a RoadsSensor when road_info is on; replace any prior one."""
        if not self.road_info:
            return
        self._remove_roads_sensor()
        self.roads_sensor = beamng_sensors.create_roads_sensor("roads", self.bng, self.vehicle)
```

Replace `_wheel_terrain_features` (lines 839-842) with:

```python
    def _road_info_features(self, pos, heading) -> np.ndarray:
        """The six road-relative features (neutral without a RoadsSensor)."""
        payload = self.roads_sensor.poll() if self.roads_sensor is not None else None
        return road_info_features(payload, self.HALF_TRACK_WIDTH, pos, heading)
```

Replace `_extra_features` (lines 844-857):

```python
    def _extra_features(self, state, pos, heading) -> np.ndarray:
        """Optional observation tail: body orientation and/or road position.

        Appended after the waypoint hints. Empty array when both flags are off, so
        a flag-off observation is byte-for-byte what it was.
        """
        blocks = []
        if self.body_orientation:
            blocks.append(self._body_orientation_features(state))
        if self.road_info:
            blocks.append(self._road_info_features(pos, heading))
        if not blocks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(blocks)
```

In `_observe`, update the call (line 716):

```python
        extra = self._extra_features(state, pos, vehicle_heading)
```

In `_format_observation_lines`, replace the label block (lines 773-776):

```python
            if self.body_orientation:
                labels += ["pitch", "roll"]
            if self.road_info:
                labels += ["edgeL", "edgeR", "rdhead", "curv", "aheadF", "aheadL"]
```

- [ ] **Step 5: Update the multi and race envs**

In `environments/beamng_multi.py`:

```python
from environments.beamng_features import road_info_features
from environments.beamng_geometry import body_orientation_features
```

(keep the other names that import block already pulls in), then:

- `VehicleSlot` line 54: `road_info: bool = False`
- `slot_n_states` (lines 126-138): rename the parameter to `road_info` and forward it
- `build_slots` (lines 174-190): `road_info = spec.get("road_info", False)`, passed to both `VehicleSlot(...)` and `slot_n_states(...)`; update the docstring's spec-dict list
- `_create_slot_sensor` line 598: `if slot.road_info:`
- `_slot_extra_features` (lines 388-406):

```python
    def _slot_extra_features(self, slot, state, pos, heading) -> np.ndarray:
        """Optional observation tail for a slot (body orientation / road position).

        Calls the shared feature helpers; empty when both flags are off.
        """
        state = state or {}
        blocks = []
        if slot.body_orientation:
            blocks.append(
                body_orientation_features(
                    state.get("dir", (0.0, 1.0, 0.0)), state.get("up", (0.0, 0.0, 1.0))
                )
            )
        if slot.road_info:
            payload = slot.roads_sensor.poll() if slot.roads_sensor is not None else None
            blocks.append(road_info_features(payload, self.HALF_TRACK_WIDTH, pos, heading))
        if not blocks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(blocks)
```

- `observe` line 372: `self._slot_extra_features(slot, state, pos, vehicle_heading),`

In `environments/beamng_race.py`, `build_race_slots` (lines 58-75): rename the local
`wheels = spec.get("wheel_terrain", False)` to `road = spec.get("road_info", False)`, pass
`road_info=road` to `VehicleSlot(...)` and `road` to `slot_n_states(...)`, and update the
docstring's spec-dict list.

- [ ] **Step 6: Delete the moved geometry helpers**

In `environments/beamng_geometry.py`, delete `_latest_road_reading` (lines 300-325) and
`wheel_terrain_features` (lines 328-343). They now live in `beamng_features.py`; leaving a
second copy behind is how the two drift.

- [ ] **Step 7: Update the factory and the action plumbing**

In `environments/__init__.py`, rename in `_make_beamng`'s signature (line 25) and its
forwarded kwargs (line 47): `road_info: bool = False` / `road_info=road_info`.

In `core/pipeline_actions.py`:

- `BeamNGOptions` line 88: `road_info: bool = False`
- `RacerOptions` (after line 169): `road_info: bool = False`
- `_beamng_kwargs` line 310: `"road_info": beamng.road_info,`
- line 281: `options.road_info,`
- line 454: `beamng.sensor, beamng.trajectory_hints, beamng.body_orientation, beamng.road_info`
- line 650: `spec.get("road_info", False),`
- line 708: `racer.sensor, racer.trajectory_hints, racer.body_orientation, racer.road_info`

No change is needed in `core/tui_backend.py`: `BeamNGOptions(**raw)` picks the new key up
from the payload, and Task 8 sends it.

- [ ] **Step 8: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS. If anything still references `wheel_terrain`, it fails here — confirm with
`git grep -n wheel_terrain -- '*.py'`, which must return nothing.

- [ ] **Step 9: Commit**

```bash
git add environments core tests
git commit -m "feat: replace wheel_terrain with the six-feature road_info block"
```

---

### Task 4: `wheel_info` flag and the GForces sensor

**Files:**
- Modify: `environments/beamng_spec.py` (`WHEEL_FEATURES`, `obs_size`)
- Modify: `environments/beamng.py` (`__init__`, `_load_scenario` sensor attach, `_observe`, `_extra_features`, `_format_observation_lines`, `close`)
- Modify: `environments/beamng_multi.py` (`VehicleSlot`, `slot_n_states`, `build_slots`, `_load_scenario` attach, `observe`, `_slot_extra_features`, `close`)
- Modify: `environments/beamng_race.py` (`build_race_slots`)
- Modify: `environments/__init__.py`, `core/pipeline_actions.py` (same five sites as Task 3)
- Test: `tests/test_beamng_spec.py`, `tests/test_beamng.py`

**Interfaces:**
- Consumes: `wheel_info_features` from Task 2; the Task 3 plumbing.
- Produces: `beamng_spec.WHEEL_FEATURES = 4`; `obs_size(..., road_info=False, wheel_info=False)`; env kwarg `wheel_info: bool = False`; `self.gforces` / `slot.gforces` sensor handles; `VehicleSlot.wheel_info`; `BeamNGOptions.wheel_info`; `RacerOptions.wheel_info`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_beamng_spec.py`:

```python
class TestWheelInfoSizing:
    def test_wheel_info_adds_four(self):
        assert obs_size("lidar", wheel_info=True) == obs_size("lidar") + 4

    def test_both_new_blocks_stack_after_the_old_tails(self):
        assert obs_size("camera", 3, True, True, True) == 6 + 256 + 6 + 2 + 6 + 4
```

Add to `tests/test_beamng.py` (inside the class holding the other option tests):

```python
    def test_wheel_block_is_neutral_without_sensors(self):
        env = _bare_env(wheel_info=True)
        out = env._wheel_info_features({}, {"vel": (0.0, 0.0, 0.0), "dir": (1.0, 0.0, 0.0)})
        assert out.shape == (4,)
        np.testing.assert_allclose(out, [0.0] * 4, atol=1e-6)

    def test_extra_features_order_is_orientation_then_road_then_wheel(self):
        env = _bare_env(body_orientation=True, road_info=True, wheel_info=True)
        state = {"dir": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0), "vel": (0.0, 0.0, 0.0)}
        out = env._extra_features(state, (0.0, 0.0, 0.0), 0.0)
        assert out.shape == (12,)

    def test_n_states_counts_both_new_blocks(self):
        base = _bare_env().n_states
        assert _bare_env(road_info=True, wheel_info=True).n_states == base + 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_spec.py tests/test_beamng.py -q`
Expected: FAIL — unexpected keyword argument `wheel_info`

- [ ] **Step 3: Extend the spec module**

In `environments/beamng_spec.py`, beside `ROAD_FEATURES`:

```python
# wheel: [long_slip, slip_angle, abs_active, lat_g]
WHEEL_FEATURES = 4
```

and in `obs_size`, add the parameter and term (and extend the layout docstring to
`... | [pitch, roll]? | road(6)? | wheel(4)?`):

```python
    wheel_info: bool = False,
```
```python
        + (WHEEL_FEATURES if wheel_info else 0)
```

- [ ] **Step 4: Wire the single-vehicle env**

In `environments/beamng.py`:

```python
from environments.beamng_features import road_info_features, wheel_info_features
```

`__init__`: add `wheel_info: bool = False` after `road_info`, store `self.wheel_info = wheel_info`, add `self.gforces = None` beside the other sensor handles (line 160), and extend the sizing call:

```python
        self.n_states = beamng_spec.obs_size(
            sensor, trajectory_hints, body_orientation, road_info, wheel_info
        )
```

In `_load_scenario`, right after the Damage sensor is attached (near line 596), add:

```python
        if self.wheel_info:
            # A classic sensor, so it rides the poll_sensors() round-trip the env
            # already makes rather than costing one of its own.
            self.gforces = GForces()
            self.vehicle.attach_sensor("gforces", self.gforces)
```

and add `GForces` to the beamngpy import at line 11 plus the `except ImportError` fallback
at line 13 (`... = GForces = None`).

Add the feature helper next to `_road_info_features`:

```python
    def _wheel_info_features(self, elec, state) -> np.ndarray:
        """The four grip features (neutral without a GForces sensor)."""
        forces = self.gforces.data if self.gforces is not None else None
        return wheel_info_features(
            elec, forces, state.get("vel", (0.0, 0.0, 0.0)), state.get("dir", (1.0, 0.0, 0.0))
        )
```

Extend `_extra_features` to take the electrics dict and append the block last:

```python
    def _extra_features(self, state, pos, heading, elec=None) -> np.ndarray:
        ...
        if self.wheel_info:
            blocks.append(self._wheel_info_features(elec or {}, state))
```

and in `_observe`: `extra = self._extra_features(state, pos, vehicle_heading, elec)`.

Extend the log labels:

```python
            if self.wheel_info:
                labels += ["slip", "slipang", "abs", "latg"]
```

- [ ] **Step 5: Wire the multi and race envs**

In `environments/beamng_multi.py`: `VehicleSlot` gains `wheel_info: bool = False` and
`gforces: Any = None`; `slot_n_states` and `build_slots` gain the flag exactly as in Task 3;
`_load_scenario` attaches `GForces()` per slot where it attaches Damage (around line 551,
`slot.vehicle.attach_sensor("gforces", slot.gforces)`); `_slot_extra_features` gains the
`elec` argument and the block; `observe` passes `elec`; `close`'s teardown tuple (line 695)
is left alone — classic sensors detach with the vehicle, unlike the automated
LiDAR/camera/roads sensors.

In `environments/beamng_race.py`, `build_race_slots` reads `spec.get("wheel_info", False)`
and forwards it to `VehicleSlot(...)` and `slot_n_states(...)`.

- [ ] **Step 6: Extend the factory and the action plumbing**

Add `wheel_info: bool = False` to `_make_beamng` (signature + forwarded kwargs),
`BeamNGOptions`, `RacerOptions`, `_beamng_kwargs`, and the four `obs_size`/`slot_n_states`
call sites listed in Task 3 Step 7.

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add environments core tests
git commit -m "feat: add the wheel_info observation block and its GForces sensor"
```

---

### Task 5: Road-poll guard

**Files:**
- Modify: `environments/beamng.py` (`_road_pollable` flag, `_advance`, `reset`, `step`, `_load_scenario`, `_road_info_features`)
- Modify: `environments/beamng_multi.py` (same flag on the env, `reset_all`, `reset_vehicle`, `step_physics`, `_load_scenario`, `_slot_extra_features`)
- Modify: `environments/beamng_sensors.py:180-187` (docstring)
- Test: `tests/test_beamng_road_guard.py` (new)

**Context the implementer needs:** `docs/romain.md:38-61` records the measured cause — the
main thread blocked in `roads_sensor.poll()` called from `observe()` inside `reset_vehicle()`
**with no physics step after the teleport**, on road-dense maps only. Since then, commit
`4a1a210` added `bng.step(5)` to `reset_vehicle` for an unrelated reward bug, so both reset
paths happen to step before observing today. This task makes that an enforced invariant
instead of a lucky call order, and Task 10 is what actually proves the freeze is gone.

**Interfaces:**
- Consumes: the Task 3 road block.
- Produces: `BeamNGDrivingEnv._road_pollable: bool`, `BeamNGDrivingEnv._advance(steps)`, `BeamNGMultiEnv._road_pollable`. Nothing later depends on these beyond the envs themselves.

- [ ] **Step 1: Write the failing test**

Create `tests/test_beamng_road_guard.py`:

```python
"""The RoadsSensor must not be polled before the sim has stepped past a teleport.

Measured (docs/romain.md, seventh issue): a poll with no intervening physics step
hangs the simulator's game-engine side forever on road-dense maps, and Python
blocks in the socket recv. The guard makes the invariant explicit rather than
relying on the order reset() happens to call things in.
"""

import numpy as np

from environments.beamng import BeamNGDrivingEnv


class _CountingRoads:
    def __init__(self):
        self.polls = 0

    def poll(self):
        self.polls += 1
        return {"halfWidth": 4.0, "dist2Left": 4.0, "dist2Right": 4.0}


def _env():
    env = BeamNGDrivingEnv(beamng_home="unused", road_info=True)
    env.roads_sensor = _CountingRoads()
    return env


class TestRoadPollGuard:
    def test_no_poll_before_the_first_step(self):
        env = _env()
        env._road_pollable = False
        out = env._road_info_features((0.0, 0.0, 0.0), 0.0)
        assert env.roads_sensor.polls == 0
        np.testing.assert_allclose(out, [0.0] * 6, atol=1e-6)

    def test_polls_once_the_sim_has_stepped(self):
        env = _env()
        env._road_pollable = True
        env._road_info_features((0.0, 0.0, 0.0), 0.0)
        assert env.roads_sensor.polls == 1

    def test_advance_opens_the_gate(self):
        env = _env()
        env._road_pollable = False

        class _Bng:
            def __init__(self):
                self.steps = 0

            def step(self, n):
                self.steps += n

        env.bng = _Bng()
        env._advance(5)
        assert env.bng.steps == 5
        assert env._road_pollable is True

    def test_a_fresh_env_starts_closed(self):
        assert BeamNGDrivingEnv(beamng_home="unused", road_info=True)._road_pollable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_road_guard.py -q`
Expected: FAIL — `AttributeError: 'BeamNGDrivingEnv' object has no attribute '_road_pollable'`

- [ ] **Step 3: Implement the guard in the single env**

In `environments/beamng.py` `__init__`, beside the other episode state:

```python
        # Whether the RoadsSensor may be polled: False from a teleport or a scenario
        # load until the simulator has advanced at least one physics step. Polling in
        # between hangs the sensor's game-engine side on road-dense maps and blocks
        # Python in the socket recv (docs/romain.md, seventh issue).
        self._road_pollable = False
```

Add the one place that steps the sim:

```python
    def _advance(self, steps: int) -> None:
        """Advance the simulation and mark the road sensor pollable again."""
        self.bng.step(steps)
        self._road_pollable = True
```

Replace the two `self.bng.step(...)` calls — `reset` line 272 (`self._advance(5)`) and `step`
line 311 (`self._advance(beamng_spec.PHYSICS_STEPS_PER_ENV_STEP)`) — and close the gate where
the pose changes: after the `self.vehicle.teleport(...)` in `reset` (line 249) and at the end
of `_load_scenario`, set `self._road_pollable = False`.

Guard the poll itself:

```python
    def _road_info_features(self, pos, heading) -> np.ndarray:
        """The six road-relative features (neutral without a sensor or before a step)."""
        payload = None
        if self.roads_sensor is not None and self._road_pollable:
            payload = self.roads_sensor.poll()
        return road_info_features(payload, self.HALF_TRACK_WIDTH, pos, heading)
```

- [ ] **Step 4: Implement the guard in the multi env**

In `environments/beamng_multi.py` `__init__`, add `self._road_pollable = False`; set it False
at the end of `_load_scenario` and after each `slot.vehicle.teleport(...)` in `reset_all`
(line 647) and `reset_vehicle` (line 664); set it True in `step_physics` and after the
`self.bng.step(5)` calls in both reset paths. In `_slot_extra_features`, poll only when
`slot.roads_sensor is not None and self._road_pollable`.

- [ ] **Step 5: Update the sensor-factory docstring**

In `environments/beamng_sensors.py`, replace the note in `create_roads_sensor`:

```python
def create_roads_sensor(name: str, bng, vehicle):
    """Create a RoadsSensor for the ``road_info`` observation block.

    Callers must not poll it between a teleport and the next physics step: the
    sensor's game-engine side never answers on road-dense maps and the caller
    blocks forever in the socket recv (docs/romain.md, seventh issue). Both envs
    enforce that with a ``_road_pollable`` gate.
    """
    return RoadsSensor(name, bng, vehicle)
```

- [ ] **Step 6: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_road_guard.py -q`
Expected: PASS (4 tests)

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add environments tests
git commit -m "fix: never poll the RoadsSensor between a teleport and a physics step"
```

---

### Task 6: Path projection wired into the envs

**Files:**
- Modify: `environments/beamng.py` (`_guide_line`, `_path_pos`, `_last_progress_m`, `_path_errors`, `_observe`, `reset`, `_launch`, `_pick_episode_path`)
- Modify: `environments/beamng_multi.py` (`VehicleSlot.guide_line` / `path_pos`, `_apply_path`, `_assign_shared_path`, `_path_errors`, `observe`, `progress_of`)
- Modify: `environments/beamng_race.py:30,166-168` (drop the import and the override)
- Modify: `environments/beamng_geometry.py:196-235` (delete `track_progress_m`)
- Modify: `tests/test_beamng_race_geometry.py` (drop the `track_progress_m` class, keep `starting_grid`)
- Test: `tests/test_beamng_path_wiring.py` (new)

**Interfaces:**
- Consumes: `project_onto_path`, `path_length`, `PathPosition`, `NEUTRAL` from Task 1.
- Produces: `BeamNGDrivingEnv._guide_line: list`, `._path_pos: PathPosition`, `.progress_m() -> float`; `VehicleSlot.guide_line`, `VehicleSlot.path_pos`; `BeamNGMultiEnv.progress_of(slot) -> float`. `_path_errors` now returns `(heading_err, dist)` in both envs. Task 7 consumes `_path_pos` / `slot.path_pos` and `progress_m()` / `progress_of`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_beamng_path_wiring.py`:

```python
"""The envs' guide polyline: spawn-first, and the source of cross-track + progress."""

import numpy as np
import pytest

from core.trajectory import TrajectoryData
from environments.beamng import BeamNGDrivingEnv
from environments.beamng_multi import BeamNGMultiEnv, VehicleSlot

SPAWN = (0.0, 0.0, 0.0)
WAYPOINTS = [(100.0, 0.0, 0.0), (100.0, 100.0, 0.0)]


def _traj():
    return TrajectoryData(
        spawn_pos=SPAWN,
        spawn_rot=(0.0, 0.0, 0.0, 1.0),
        sparse_waypoints=list(WAYPOINTS),
        dense_waypoints=list(WAYPOINTS),
        map_name="gridmap_v2",
        generated_at="2026-08-04T00:00:00+00:00",
        source="test",
    )


def _env():
    env = BeamNGDrivingEnv(beamng_home="unused")
    env.trajectory = _traj()
    env.waypoints = list(WAYPOINTS)
    env._rebuild_guide_line()
    return env


class TestGuideLine:
    def test_starts_at_the_spawn_so_the_first_segment_counts(self):
        # Projecting onto the waypoints alone would clamp progress to 0 until the
        # car had passed checkpoint 0 — the whole first segment would be invisible.
        env = _env()
        assert env._guide_line[0] == SPAWN
        assert len(env._guide_line) == len(WAYPOINTS) + 1

    def test_progress_grows_along_the_first_segment(self):
        env = _env()
        assert env._project((40.0, 0.0, 0.0)).progress_m == pytest.approx(40.0)

    def test_progress_keeps_growing_round_the_corner(self):
        env = _env()
        assert env._project((100.0, 60.0, 0.0)).progress_m == pytest.approx(160.0)

    def test_no_trajectory_means_a_neutral_projection(self):
        bare = BeamNGDrivingEnv(beamng_home="unused")
        assert bare._project((5.0, 5.0, 0.0)).progress_m == 0.0


class TestPathErrorsReturnsTwoValues:
    def test_heading_and_dist_only(self):
        env = _env()
        heading_err, dist = env._path_errors((0.0, 0.0, 0.0), {"vel": (1.0, 0.0, 0.0)})
        assert dist == pytest.approx(100.0)
        assert heading_err == pytest.approx(0.0)


class TestMultiEnvProgress:
    def test_progress_of_uses_the_slot_guide_line(self):
        slot = VehicleSlot(name="ego_0", color="White", agent=None, save_path="")
        slot.waypoints = list(WAYPOINTS)
        slot.spawn_pos = SPAWN
        slot.guide_line = [SPAWN, *WAYPOINTS]
        slot.current_pos = (40.0, 0.0, 0.0)
        env = BeamNGMultiEnv(slots=[slot], beamng_home="unused")
        assert env.progress_of(slot) == pytest.approx(40.0)

    def test_progress_of_is_zero_without_a_guide_line(self):
        slot = VehicleSlot(name="ego_0", color="White", agent=None, save_path="")
        env = BeamNGMultiEnv(slots=[slot], beamng_home="unused")
        assert env.progress_of(slot) == 0.0


class TestTrackProgressIsGone:
    def test_the_superseded_helper_is_deleted(self):
        import environments.beamng_geometry as geometry

        assert not hasattr(geometry, "track_progress_m")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_path_wiring.py -q`
Expected: FAIL — `AttributeError: 'BeamNGDrivingEnv' object has no attribute '_rebuild_guide_line'`

- [ ] **Step 3: Wire the single env**

In `environments/beamng.py`, import the projection:

```python
from environments.beamng_path import NEUTRAL as NEUTRAL_PATH_POS
from environments.beamng_path import PathPosition, path_length, project_onto_path
```

In `__init__`, beside the waypoint state:

```python
        # The polyline the observation and the reward measure against: the spawn
        # followed by every checkpoint. The spawn is what makes the first segment
        # count — `waypoints` starts after it, so projecting onto that alone would
        # report zero progress until checkpoint 0 was behind the car.
        self._guide_line: list[tuple[float, float, float]] = []
        self._path_pos: PathPosition = NEUTRAL_PATH_POS
        self._last_progress_m = 0.0
```

Add the three helpers next to `_select_waypoints`:

```python
    def _rebuild_guide_line(self) -> None:
        """Refresh the guide polyline from the current trajectory + waypoints."""
        if self.trajectory is None:
            self._guide_line = []
            return
        self._guide_line = [tuple(self.trajectory.spawn_pos), *self.waypoints]

    def _project(self, pos) -> PathPosition:
        """Where ``pos`` sits on the guide polyline."""
        return project_onto_path(self._guide_line, pos)

    def progress_m(self) -> float:
        """Metres covered along the path, laps included."""
        laps_done = self._waypoint_idx // len(self.waypoints) if self.waypoints else 0
        return self._path_pos.progress_m + laps_done * path_length(self._guide_line)
```

Call `self._rebuild_guide_line()` wherever `self.waypoints` is assigned — in `_launch`
(after line 478) and in `reset` (after line 259).

In `_path_errors`, drop the lateral computation and return two values:

```python
    def _path_errors(self, pos, state):
        """Return (heading_error_rad, dist) for the next waypoint; advance on arrival.

        Cross-track error no longer comes from here: ``dist * sin(heading_err)`` is a
        function of the two values this returns, so it carried no information. The
        observation uses the guide-line projection instead.
        """
```

with the body unchanged except the final two lines:

```python
        lateral = None  # (removed — see docstring)
        return float(heading_err), dist
```

(delete the `lateral_err` assignment entirely rather than leaving a dead local).

In `_observe`, project once and use it:

```python
        heading_err, dist = self._path_errors(pos, state)
        self._path_pos = self._project(pos)
```

and in the `kin` array, replace the lateral entry with:

```python
                np.clip(self._path_pos.cross_track_m / 5.0, -1.0, 1.0),
```

In `reset`, after the priming observe (line 276), seed the progress baseline beside
`_last_dist`:

```python
        self._last_progress_m = self.progress_m()
```

- [ ] **Step 4: Wire the multi and race envs**

In `environments/beamng_multi.py`:

- `VehicleSlot`: `guide_line: list = field(default_factory=list)` beside `waypoints`, and
  `path_pos: Any = None` beside `current_pos`
- `_apply_path` and `_assign_shared_path`: after assigning `slot.waypoints` and
  `slot.spawn_pos`, set `slot.guide_line = [tuple(slot.spawn_pos), *slot.waypoints]`
- `_path_errors`: same two-value change as the single env
- `observe`: `heading_err, dist = self._path_errors(slot, pos, state)`, then
  `slot.path_pos = project_onto_path(slot.guide_line, pos)`, and the kin array's lateral
  entry becomes `np.clip(slot.path_pos.cross_track_m / 5.0, -1.0, 1.0)`
- add the progress method (this is what the race env used to own):

```python
    def progress_of(self, slot: VehicleSlot) -> float:
        """How far along its path a slot is, in metres, laps included.

        Lives here rather than in the race env because the pace reward needs it in
        training too — one definition of "how far along am I", shared by pace and by
        the race gap term.
        """
        if not slot.guide_line:
            return 0.0
        pos = project_onto_path(slot.guide_line, slot.current_pos)
        laps_done = slot.waypoint_idx // len(slot.waypoints) if slot.waypoints else 0
        return pos.progress_m + laps_done * path_length(slot.guide_line)
```

In `environments/beamng_race.py`, delete the `track_progress_m` import (line 30) and the
`progress_of` override (lines 166-168) — the base class now provides it.

- [ ] **Step 5: Delete the superseded helper**

Delete `track_progress_m` from `environments/beamng_geometry.py` (lines 196-235) and delete
the `track_progress_m` test class from `tests/test_beamng_race_geometry.py`, keeping its
`starting_grid` tests and updating the module docstring's first paragraph to describe only
the grid. The projection tests in `tests/test_beamng_path.py` cover ordering two cars along
a shared path (`TestProgress`).

- [ ] **Step 6: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_path_wiring.py -q`
Expected: PASS (8 tests)

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS. `tests/test_beamng_race.py:213-214` still passes — it compares
`progress_of` against itself, so the change of measure does not affect it.

- [ ] **Step 7: Commit**

```bash
git add environments tests
git commit -m "feat: measure cross-track error and progress by projecting onto the path"
```

---

### Task 7: Path-relative pace in the reward

**Files:**
- Modify: `environments/beamng_reward.py` (module docstring, constants, `compute_race_reward`, `RewardOutcome`)
- Modify: `environments/beamng.py:893-921` (`_compute_reward`)
- Modify: `environments/beamng_multi.py:300-331` (`compute_reward`)
- Test: `tests/test_beamng_reward_path.py` (new); `tests/test_beamng_reward.py` must keep passing untouched

**Interfaces:**
- Consumes: `progress_m()` / `progress_of` and `_path_pos` / `slot.path_pos` from Task 6.
- Produces: `compute_race_reward(..., progress_m=None, last_progress_m=None, path_alignment=None, segment_len_m=None)` and `RewardOutcome.progress_m`. `SEGMENT_TIME_BONUS` replaces `SEGMENT_TARGET_STEPS` and `SEGMENT_TIME_COEF`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_beamng_reward_path.py`:

```python
"""The pace terms measured along the path instead of at the next checkpoint.

Two things break once checkpoints are hundreds of metres apart: straight-line
closure reads a bend as backward progress, and a segment-time par fixed at the
25 m generated spacing can never be met. Both are fixed here; the fallback
behaviour (no path arguments supplied) is asserted to be byte-identical so the
existing reward tests stay meaningful.
"""

import numpy as np
import pytest

from environments import beamng_reward
from environments.beamng_reward import compute_race_reward

N_PERCEPTION = 8


def _obs(speed=0.0, heading_err=0.0, damage=0.0):
    obs = np.zeros(6 + N_PERCEPTION, dtype=np.float32)
    obs[0] = speed
    obs[2] = heading_err
    obs[4] = damage
    obs[6:] = 1.0  # all-clear LiDAR, so the obstacle penalty stays out of the way
    return obs


def _reward(**over):
    kwargs = dict(
        perception="lidar",
        n_perception=N_PERCEPTION,
        waypoints_len=10,
        waypoint_idx=1,
        checkpoint_hit=False,
        last_dist=100.0,
        current_dist=100.0,
        last_damage=0.0,
        steps=10,
        invuln_steps=0,
        max_steps=5000,
        max_damage=1000.0,
    )
    obs = over.pop("obs", _obs())
    kwargs.update(over)
    return compute_race_reward(obs, **kwargs)


class TestProgressAlongThePath:
    def test_metres_gained_along_the_path_are_paid(self):
        out = _reward(progress_m=140.0, last_progress_m=130.0)
        # 10 m of path progress at PROGRESS_COEF, minus the step penalty.
        assert out.reward == pytest.approx(10.0 * beamng_reward.PROGRESS_COEF - beamng_reward.STEP_PENALTY)

    def test_a_bend_that_increases_straight_line_distance_still_pays(self):
        # The failure this fixes: driving the road round a corner moves the car
        # away from the checkpoint, which the old term scored as going backwards.
        out = _reward(
            progress_m=140.0,
            last_progress_m=130.0,
            last_dist=100.0,
            current_dist=112.0,
        )
        assert out.reward > 0.0

    def test_progress_is_not_zeroed_on_a_checkpoint_step(self):
        # Position-based progress does not jump when the target index advances, so
        # the old zeroing hack must not swallow a real 10 m of progress.
        out = _reward(progress_m=140.0, last_progress_m=130.0, checkpoint_hit=True)
        assert out.reward > beamng_reward.CHECKPOINT_BONUS

    def test_the_outcome_carries_the_progress_used(self):
        assert _reward(progress_m=140.0, last_progress_m=130.0).progress_m == pytest.approx(140.0)

    def test_falls_back_to_straight_line_closure(self):
        out = _reward(last_dist=100.0, current_dist=90.0)
        assert out.reward == pytest.approx(10.0 * beamng_reward.PROGRESS_COEF - beamng_reward.STEP_PENALTY)

    def test_fallback_still_zeroes_progress_on_the_hit_step(self):
        out = _reward(last_dist=100.0, current_dist=10.0, checkpoint_hit=True)
        # Only the checkpoint bonus and the segment bonus, no 90 m windfall.
        assert out.reward < beamng_reward.CHECKPOINT_BONUS + beamng_reward.SEGMENT_TIME_BONUS


class TestSpeedAlignment:
    def test_path_alignment_overrides_the_checkpoint_bearing(self):
        # Pointing 180 deg from the checkpoint but along the road: the old term
        # would charge for it, the tangent term pays.
        out = _reward(obs=_obs(speed=0.5, heading_err=1.0), path_alignment=1.0)
        assert out.reward > 0.0

    def test_without_it_the_checkpoint_bearing_is_used(self):
        aligned = _reward(obs=_obs(speed=0.5, heading_err=0.0)).reward
        opposed = _reward(obs=_obs(speed=0.5, heading_err=1.0)).reward
        assert aligned > opposed


class TestSegmentTimeBonus:
    def test_par_comes_from_the_segment_being_driven(self):
        # 1000 m at SEGMENT_PAR_SPEED_MS is a long par; 90 steps is well inside it.
        out = _reward(checkpoint_hit=True, steps_since_checkpoint=90, segment_len_m=1000.0)
        assert out.reward > beamng_reward.CHECKPOINT_BONUS

    def test_the_bonus_does_not_grow_with_track_length(self):
        # The scale trap: on italy highway1 (1064 m average) a par-relative bonus
        # must stay worth about half a checkpoint, not ten times one.
        short = _reward(checkpoint_hit=True, steps_since_checkpoint=1, segment_len_m=25.0).reward
        long = _reward(checkpoint_hit=True, steps_since_checkpoint=1, segment_len_m=1000.0).reward
        assert long == pytest.approx(short, abs=1e-3)
        assert long <= beamng_reward.CHECKPOINT_BONUS + beamng_reward.SEGMENT_TIME_BONUS + 1.0

    def test_missing_par_floors_the_bonus_at_zero(self):
        out = _reward(checkpoint_hit=True, steps_since_checkpoint=9999, segment_len_m=25.0)
        assert out.reward == pytest.approx(beamng_reward.CHECKPOINT_BONUS - beamng_reward.STEP_PENALTY)

    def test_the_old_scale_constants_are_gone(self):
        assert not hasattr(beamng_reward, "SEGMENT_TARGET_STEPS")
        assert not hasattr(beamng_reward, "SEGMENT_TIME_COEF")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_reward_path.py -q`
Expected: FAIL — `TypeError: compute_race_reward() got an unexpected keyword argument 'path_alignment'`

- [ ] **Step 3: Change the reward**

In `environments/beamng_reward.py`, replace the segment constants (lines 62-71) with:

```python
# The segment-time bonus is *relative*: beat par and it pays up to
# SEGMENT_TIME_BONUS, miss it and it floors at zero. Par is derived per segment
# from the geometry actually being driven — a constant par (it used to be
# SPARSE_SPACING_M, 25 m) is unmeetable on a game track, where 30 of the 44
# shipped sprint/lap tracks have gaps over 300 m.
#
# Relative rather than "steps under par x a coefficient" because that shape scales
# with track length: on italy's highway1 (1064 m average segment) par is ~266 steps
# and a 40 m/s run takes ~80, which would have paid 744 — more than the checkpoint
# and finish bonuses combined.
SEGMENT_PAR_SPEED_MS = 12.0
SEGMENT_TIME_BONUS = 25.0  # half a checkpoint bonus for a perfect segment
```

Delete the `SPARSE_SPACING_M` import if nothing else uses it (`git grep -n SPARSE_SPACING_M
-- environments/beamng_reward.py`), add `progress_m: float = 0.0` to `RewardOutcome`, and add
the two new parameters to the signature beside the existing progress ones:

```python
    path_alignment: float | None = None,
    segment_len_m: float | None = None,
```

Replace the alignment line (line 161):

```python
    # Speed is projected onto the direction we want to be going. With a path tangent
    # that is where the road goes; without one it falls back to the bearing to the
    # next checkpoint, which is the same thing only while checkpoints are close.
    if path_alignment is None:
        alignment = float(np.cos(heading_err * np.pi))
    else:
        alignment = float(np.clip(path_alignment, -1.0, 1.0))
```

Replace the progress term (lines 175-180):

```python
    # 1. Progress. Metres gained *along the path* when the caller measures it, which
    #    is continuous across a checkpoint and cannot read a bend as going backwards.
    #    Falling back to straight-line closure keeps the old behaviour — including
    #    zeroing the hit step, where the target jumping to the next waypoint would
    #    otherwise look like a large step backwards.
    if progress_m is not None and last_progress_m is not None:
        reward += (float(progress_m) - float(last_progress_m)) * PROGRESS_COEF
    else:
        dist_delta = 0.0 if checkpoint_hit else (last_dist - current_dist)
        reward += dist_delta * PROGRESS_COEF
    last_dist = current_dist
```

Replace the checkpoint bonus block (lines 218-224):

```python
    # 7. Checkpoint: a flat bonus plus a bonus for having got there quickly.
    steps_since_checkpoint += 1
    if checkpoint_hit:
        reward += CHECKPOINT_BONUS
        par = beamng_spec.steps_for_distance(
            float(segment_len_m) if segment_len_m else SPARSE_SPACING_M, SEGMENT_PAR_SPEED_MS
        )
        reward += SEGMENT_TIME_BONUS * float(np.clip(1.0 - steps_since_checkpoint / par, 0.0, 1.0))
        steps_since_checkpoint = 0
        checkpoint_hit = False
```

(keep the `SPARSE_SPACING_M` import for this fallback), and add `progress_m` to the returned
`RewardOutcome`:

```python
        progress_m=float(progress_m) if progress_m is not None else 0.0,
```

Finally, update the module docstring's first bullet: progress is "dense progress along the
path (x3)" and the segment bonus is "a relative bonus for beating the segment's own par".

- [ ] **Step 4: Feed it from the envs**

In `environments/beamng.py` `_compute_reward`, add the four arguments and the write-back:

```python
            progress_m=self.progress_m(),
            last_progress_m=self._last_progress_m,
            path_alignment=float(np.cos(self._heading_minus_tangent())),
            segment_len_m=self._path_pos.segment_len_m,
```
```python
        self._last_progress_m = outcome.progress_m
```

with the small helper beside `_project`:

```python
    def _heading_minus_tangent(self) -> float:
        """Angle between where the car points and where the path goes, in radians."""
        dir_vec = (self.vehicle.state or {}).get("dir", (1.0, 0.0, 0.0)) if self.vehicle else (1.0, 0.0, 0.0)
        heading = float(np.arctan2(dir_vec[1], dir_vec[0]))
        return (heading - self._path_pos.tangent_rad + np.pi) % (2 * np.pi) - np.pi
```

In `environments/beamng_multi.py` `compute_reward`, pass the same four (per slot) and write
back `slot.last_progress_m = outcome.progress_m`:

```python
            progress_m=self.progress_of(slot),
            last_progress_m=slot.last_progress_m,
            path_alignment=self._slot_path_alignment(slot),
            segment_len_m=slot.path_pos.segment_len_m if slot.path_pos else None,
            **race_kwargs,
```

with:

```python
    def _slot_path_alignment(self, slot) -> float:
        """cos(angle between the slot's heading and its path tangent)."""
        if slot.path_pos is None:
            return None
        state = slot.vehicle.state or {} if slot.vehicle else {}
        dir_vec = state.get("dir", (1.0, 0.0, 0.0))
        heading = float(np.arctan2(dir_vec[1], dir_vec[0]))
        return float(np.cos(heading - slot.path_pos.tangent_rad))
```

Note: `race_kwargs` may already contain `progress_m` / `last_progress_m` from
`BeamNGRaceEnv.compute_race_reward_for`. Remove those two keys from that call site (lines
190-195 of `beamng_race.py`) — the base method now supplies them, and passing both would
raise `TypeError: got multiple values for keyword argument`. The race env keeps
`rival_progress_m`, `last_rival_progress_m`, `laps` and `rival_finished`.

- [ ] **Step 5: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_beamng_reward_path.py tests/test_beamng_reward.py -q`
Expected: PASS — the new file, and the existing reward tests unchanged (they pass no path
arguments, so they exercise the fallback).

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add environments tests
git commit -m "feat: score pace along the path and make the segment bonus scale-free"
```

---

### Task 8: TUI options, save-path tokens and human play

**Files:**
- Modify: `tui/src/workflows.ts:47-66` (`BeamNGFields`, `BEAMNG_DEFAULTS`), `:112-160` (`MultiSpecState`, `RacerState`, `HumanPlayState`, `beamngPathSuffix`, `trainSavePath`), `:187-240` (payload builders)
- Modify: `tui/src/forms.ts:44-60` (`beamngFieldsFrom`), `:88-110` (`addBeamngFields`), `:214-256` (multi spec + chips), `:270-280` (multi fields), `:319-356` (racer fields/paths), human-play form
- Modify: `tui/src/controller.ts:83-93` (refresh allow-lists)
- Modify: `core/pipeline_actions.py:132-136` (`HumanPlayRequest`), `:557-573` (`run_human_play`)
- Modify: `core/tui_backend.py` (human-play command)
- Test: `tui/src/__tests__/workflows.test.ts`

**Interfaces:**
- Consumes: `BeamNGOptions.road_info` / `.wheel_info` and `RacerOptions` from Tasks 3-4.
- Produces: `beamngPathSuffix({trajectory_hints, body_orientation, road_info, wheel_info})` emitting `_h{n}_ori_road_whl`; payload keys `road_info` / `wheel_info` on the train, multi-spec, racer and human-play payloads.

- [ ] **Step 1: Write the failing test**

Add to `tui/src/__tests__/workflows.test.ts`:

```typescript
describe("beamngPathSuffix with the new observation flags", () => {
  test("road_info and wheel_info each add a token", () => {
    expect(beamngPathSuffix({ trajectory_hints: 0, body_orientation: false, road_info: true, wheel_info: false }))
      .toBe("_road");
    expect(beamngPathSuffix({ trajectory_hints: 0, body_orientation: false, road_info: false, wheel_info: true }))
      .toBe("_whl");
  });

  test("tokens keep their order regardless of how the flags arrive", () => {
    expect(
      beamngPathSuffix({ trajectory_hints: 3, body_orientation: true, road_info: true, wheel_info: true }),
    ).toBe("_h3_ori_road_whl");
  });

  test("train save path carries both tokens", () => {
    expect(
      trainSavePath("td3", "camera", {
        trajectory_hints: 0,
        body_orientation: false,
        road_info: true,
        wheel_info: true,
      }),
    ).toBe("outputs/td3_camera_road_whl.pth");
  });

  test("flags off leave the path untouched", () => {
    expect(trainSavePath("dqn", "lidar", BEAMNG_DEFAULTS)).toBe("outputs/dqn_lidar.pth");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `tui/`): `bun test`
Expected: FAIL — the suffix comes back `""` for the new flags.

- [ ] **Step 3: Update the TypeScript types and helpers**

In `tui/src/workflows.ts`:

```typescript
export interface BeamNGFields {
  map_name: string;
  sensor: string;
  trajectory_hints: number;
  body_orientation: boolean;
  road_info: boolean;
  wheel_info: boolean;
  random_path?: boolean;
  dense_episodes?: number;
  // A game-track key, or "" for the generated paths.
  track?: string;
}

export const BEAMNG_DEFAULTS: BeamNGFields = {
  map_name: "gridmap_v2",
  sensor: "lidar",
  trajectory_hints: 0,
  body_orientation: false,
  road_info: false,
  wheel_info: false,
  track: "",
};
```

Replace `wheel_terrain` with `road_info: boolean; wheel_info: boolean;` in `MultiSpecState`,
add both to `RacerState`, and add both to `HumanPlayState`. Then:

```typescript
// Encode the beamng options that change what a checkpoint represents into the file
// name, so different configs cannot overwrite each other: "_h<n>" for checkpoint
// hints (>0), "_ori" for body orientation, "_road" for road position, "_whl" for
// wheel performance. The order is fixed so a path is reproducible.
export function beamngPathSuffix(beamng?: {
  trajectory_hints: number;
  body_orientation: boolean;
  road_info?: boolean;
  wheel_info?: boolean;
}): string {
  if (!beamng) return "";
  let suffix = "";
  if (beamng.trajectory_hints > 0) suffix += `_h${beamng.trajectory_hints}`;
  if (beamng.body_orientation) suffix += "_ori";
  if (beamng.road_info) suffix += "_road";
  if (beamng.wheel_info) suffix += "_whl";
  return suffix;
}
```

Widen `trainSavePath`'s `beamng` parameter type to match, add both keys to the racer objects
in `buildCoursePayload`, and add them to `buildHumanPlayPayload`.

- [ ] **Step 4: Update the forms**

In `tui/src/forms.ts`:

- `beamngFieldsFrom`: `road_info: bool(values.road_info), wheel_info: bool(values.wheel_info),`
- `addBeamngFields`: replace the `wheel_terrain is intentionally NOT offered` comment with
  the two fields:

```typescript
  addChoice(ctx, "body_orientation", "Body orientation", ["false", "true"]);
  // Road position (edges, road-relative heading, curvature, look-ahead) and wheel
  // performance (slip, slide, ABS, lateral g). Both change the observation width, so
  // both feed the derived save path — see controller.onChoiceChanged.
  addChoice(ctx, "road_info", "Road position", ["false", "true"]);
  addChoice(ctx, "wheel_info", "Wheel performance", ["false", "true"]);
```

- multi-agent block: add `addChoice(ctx, "multi_road_info", "Road position", ["false", "true"])`
  and `addChoice(ctx, "multi_wheel_info", "Wheel performance", ["false", "true"])` after
  `multi_body_orientation`; in `addMultiSpec` read them, pass them into `beamngPathSuffix`,
  and store them on the spec (replacing the `wheel_terrain: false` line); in
  `addMultiSpecList` extend the chips: `s.road_info ? "road" : ""`, `s.wheel_info ? "whl" : ""`
- racer block: add `addChoice(ctx, \`r${n}_road_info\`, "Road position", ["false", "true"])`
  and the `wheel_info` twin; include both in `refreshRacerPaths`' `trainSavePath` call and in
  `racerFrom`
- human-play form: add the same two choices, and include them in the `HumanPlayState` the
  run action builds

In `tui/src/controller.ts`, extend both allow-lists:

```typescript
  if (
    wf === "train" &&
    (f.key === "sensor" ||
      f.key === "body_orientation" ||
      f.key === "road_info" ||
      f.key === "wheel_info")
  ) {
```
```typescript
  if (
    wf === "course" &&
    (f.key.endsWith("_algo") ||
      f.key.endsWith("_sensor") ||
      f.key.endsWith("_body_orientation") ||
      f.key.endsWith("_road_info") ||
      f.key.endsWith("_wheel_info"))
  ) {
```

- [ ] **Step 5: Extend the human-play request on the Python side**

In `core/pipeline_actions.py`:

```python
@dataclass
class HumanPlayRequest:
    map_name: str
    sensor: str = beamng_spec.DEFAULT_SENSOR
    random_path: bool = False
    track: str = ""
    road_info: bool = False
    wheel_info: bool = False
```

and forward them in `run_human_play`'s factory call:

```python
            road_info=request.road_info,
            wheel_info=request.wheel_info,
```

In `core/tui_backend.py`'s `_cmd_human_play`, pass `road_info=payload.get("road_info", False)`
and `wheel_info=payload.get("wheel_info", False)`.

- [ ] **Step 6: Run the tests**

Run (from `tui/`): `bun test` then `bun run typecheck`
Expected: PASS, no type errors. `git grep -n wheel_terrain -- tui` must return nothing.

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tui core
git commit -m "feat: offer road_info and wheel_info in the menus and the save path"
```

---

### Task 9: Documentation

**Files:**
- Modify: `docs/beamng_environment.md:60-90` (observation layout, index table, optional tails)
- Modify: `README.md` (the beamng options list)
- Modify: `docs/romain.md:38-61` (resolve the seventh issue) and append the new issue entry

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update the environment reference**

In `docs/beamng_environment.md`, replace the layout block and the two stale rows:

```
kinematic(6) | perception(P) | hints(2*H) | [pitch, roll]? | road(6)? | wheel(4)?
```

| Index | Name | Raw source | Normalization |
|---|---|---|---|
| 3 | `lateral_error` | signed cross-track distance from the guide polyline (m) | `/ 5.0`, clipped |

and replace the optional-tails list:

```markdown
- `trajectory_hints=H` — vehicle-local `(forward, left)` of the next `H` waypoints,
  normalized over 100 m. **+2H** dims. Saturates on game tracks, whose checkpoints are
  far further apart than the 100 m norm.
- `body_orientation` — `[pitch, roll]` from the vehicle's forward/up vectors. **+2**.
- `road_info` — `[edge_left, edge_right, road_heading, curvature, ahead_fwd, ahead_left]`
  from a `RoadsSensor`. **+6**. Road-relative, so it does not care how far away the next
  checkpoint is.
- `wheel_info` — `[long_slip, slip_angle, abs_active, lat_g]` from Electrics, the vehicle
  state and a `GForces` sensor. **+4**.

`beamng_spec.obs_size(sensor, hints, body_orientation, road_info, wheel_info)` is the only
place this arithmetic lives.
```

Also fix the action-space paragraph at line 95: the car is AWD, not "a mid-engine RWD" —
the table itself is unchanged, only the reason given for it is stale.

- [ ] **Step 2: Update the README options list**

Add `road_info` and `wheel_info` next to the other BeamNG options, one line each, matching
the file's French phrasing.

- [ ] **Step 3: Close out the seventh issue in the journal**

In `docs/romain.md`, replace the "Fix for now" paragraph (lines 59-61) with the resolution:
the poll is now gated on a physics step having happened since the last teleport
(`_road_pollable`), the option is offered again as `road_info`, and it carries six features
instead of two. Then append a new numbered issue describing the long-segment problem and the
path-projection fix, in the same voice as the surrounding entries.

- [ ] **Step 4: Verify the docs match the code**

Run: `git grep -n "wheel_terrain" -- docs README.md`
Expected: only historical mentions inside `docs/romain.md` and `docs/superpowers/` (the
journal and the dated specs/plans are records, not live docs).

- [ ] **Step 5: Commit**

```bash
git add docs README.md
git commit -m "docs: document the road_info and wheel_info observation blocks"
```

---

### Task 10: In-sim verification

**Files:** none — this is a session at the simulator, and its findings either close the task or open a follow-up.

**Interfaces:** none.

**Why it exists:** three things in this design cannot be settled by a unit test — whether the
freeze is really gone, whether `lat_g` reads the lateral axis, and whether the curvature sign
matches the road. All three are cheap to check in human play, where the observation is logged
with labels.

- [ ] **Step 1: Human play on the map that froze**

Run the TUI, choose human play on `west_coast_usa` with `Road position: true` and
`Wheel performance: true`. Drive, then trigger several resets.
Expected: no freeze; the obs log prints `obs extra | edgeL=… edgeR=… rdhead=… curv=… aheadF=… aheadL=… slip=… slipang=… abs=… latg=…`.
If it freezes: the fallback is the design's rejected option — wrap the poll in a daemon
thread with a timeout (mirroring `beamng_sensors.remove_sensor`) so a hang degrades to
neutral features. Record the finding in `docs/romain.md` before changing anything.

- [ ] **Step 2: Check the `lat_g` axis**

Hold a steady left-hand corner, then a right-hand one.
Expected: `latg` takes a consistent sign per direction and swaps between them.
If it stays ~0 while cornering, the lateral axis is `gy`, not `gx`: change the key order in
`wheel_info_features` to `("gy2", "gy")` and update its docstring plus the test
`test_lateral_g_prefers_gx2_and_normalizes`.

- [ ] **Step 3: Check the curvature sign**

Drive a long left-hand bend, then a right-hand one.
Expected: `curv` is positive on the left-hander, negative on the right-hander, ~0 on a
straight. If inverted, flip the comparison in `_signed_curvature` and its two hairpin tests.

- [ ] **Step 4: Check the slip features respond**

Full throttle from a standstill, then a hard stop.
Expected: `slip` goes clearly positive under wheelspin and negative under braking; `abs`
reads 1 while ABS is working.

- [ ] **Step 5: A short training run on a long-segment track**

Train ~20 episodes on `italy` / sprint `highway1` with both flags on.
Expected: no freeze across resets; the run's save path is
`outputs/<algo>_<sensor>_road_whl.pth`; episode reward is not dominated by a single
checkpoint's segment bonus.

- [ ] **Step 6: Record the outcome**

Append the results to `docs/romain.md` (verified / adjusted, with the numbers seen), then:

```bash
git add docs/romain.md
git commit -m "docs: record the in-sim verification of the road and wheel observations"
```

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| Observation layout + `obs_size` signature | 3 (road), 4 (wheel) |
| `road_info` 6 features, normalization, curvature sign, point ordering | 2 (math), 3 (wiring) |
| `wheel_info` 4 features, GForces, ground speed from state | 2 (math), 4 (wiring) |
| Fail-soft on every field | 2 (tests assert it), enforced by the Global Constraints |
| Sensors created only when their flag is on | 3 (roads), 4 (gforces) |
| Reset freeze fix | 5, verified in 10 |
| `beamng_path.project_onto_path` + spawn-first polyline + lap offset | 1, 6 |
| `track_progress_m` deleted, tests migrated | 6 |
| Reward: path progress, path alignment, scale-free segment bonus | 7 |
| `last_progress_m` seeded at reset | 6 (single env), `reset_episode` already zeroes the slot field |
| `lateral_err` slot swap | 6 |
| Plumbing: envs, factory, `BeamNGOptions`, `RacerOptions`, `tui_backend` | 3, 4, 8 |
| Save-path tokens + refresh allow-lists | 8 |
| Forms incl. human play | 8 |
| Docs | 9 |
| `lat_g` axis + curvature sign verification | 10 |

**Placeholder scan:** no TBD/TODO; every code step carries the code, every test step the test,
every run step the exact command and expected result. Task 10's steps are manual by nature and
each states its expected reading and its fallback.

**Type consistency:** `PathPosition` field names (`progress_m`, `cross_track_m`, `tangent_rad`,
`segment_index`, `segment_len_m`) are used identically in Tasks 1, 6 and 7. `road_info_features(payload, half_track_width, pos, heading)` and `wheel_info_features(electrics, gforces, vel, dir_vec)`
keep their argument order across Tasks 2, 3 and 4. `progress_m()` (single env, a method) and
`progress_of(slot)` (multi env) are deliberately different names for the two shapes and are used
that way in Task 7. `beamngPathSuffix`'s object type gains `road_info?` / `wheel_info?` as
optional so `BEAMNG_DEFAULTS` and the racer states both satisfy it.

**One collision worth re-flagging:** Task 7 Step 4 removes `progress_m` / `last_progress_m` from
`BeamNGRaceEnv.compute_race_reward_for`'s kwargs because the base `compute_reward` now supplies
them. Skipping that is a `TypeError` at the first race step, not a silent wrong number.
