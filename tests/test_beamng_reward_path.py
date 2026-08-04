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
        # Same relative pace on both (par / steps_since_checkpoint, after the
        # in-function +1): par is 7 steps for 25 m and 250 for 1000 m, so 106
        # steps on the long segment matches the same ratio as 2 steps on the
        # short one (250/107 ~= 7/3). Deliberately not proportional-by-distance
        # (2 : 80 looks equivalent at 25 m/1000 m, but that ignores the +1 and
        # overshoots the tolerance below by ~1.1 — the short segment's par is
        # small enough that +1 is a much bigger fraction of it).
        short = _reward(checkpoint_hit=True, steps_since_checkpoint=2, segment_len_m=25.0).reward
        long = _reward(checkpoint_hit=True, steps_since_checkpoint=106, segment_len_m=1000.0).reward
        assert long == pytest.approx(short, abs=1.0)  # spread is par's ceil() on 25 m
        assert long <= beamng_reward.CHECKPOINT_BONUS + beamng_reward.SEGMENT_TIME_BONUS + 1.0

    def test_the_bonus_still_discriminates_at_race_pace(self):
        # The trap a naive ratio-past-par shape falls into: flat above 2x par, so
        # no gradient anywhere a 682 hp car actually drives.
        fast = _reward(checkpoint_hit=True, steps_since_checkpoint=60, segment_len_m=1000.0).reward
        faster = _reward(checkpoint_hit=True, steps_since_checkpoint=50, segment_len_m=1000.0).reward
        assert faster > fast

    def test_missing_par_floors_the_bonus_at_zero(self):
        out = _reward(checkpoint_hit=True, steps_since_checkpoint=9999, segment_len_m=25.0)
        assert out.reward == pytest.approx(beamng_reward.CHECKPOINT_BONUS - beamng_reward.STEP_PENALTY)

    def test_the_old_scale_constants_are_gone(self):
        assert not hasattr(beamng_reward, "SEGMENT_TARGET_STEPS")
        assert not hasattr(beamng_reward, "SEGMENT_TIME_COEF")
