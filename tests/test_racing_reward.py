"""Behavioural tests for the racing reward — the properties it exists to create.

`test_beamng_reward.py` covers observation slicing through the real env callers;
this module tests the reward function directly, asserting the incentives:
faster beats slower, contact is priced but not fatal, and the gap term rewards
being ahead without being farmable.
"""

import numpy as np
import pytest

import environments.beamng_reward as reward_mod
from environments.beamng_reward import compute_race_reward


def _obs(n=14, *, speed=0.5, dist_norm=1.0, damage_norm=0.0):
    """Obs with a moving vehicle, perfect alignment and a clear perception block."""
    obs = np.zeros(n, dtype=np.float32)
    obs[0] = speed
    obs[4] = damage_norm
    obs[5] = dist_norm
    obs[6:] = 1.0
    return obs


# speed=0.5 x alignment=1 x SPEED_ALIGN_COEF, minus the per-step time penalty.
CLEAN = 0.5 * reward_mod.SPEED_ALIGN_COEF - reward_mod.STEP_PENALTY


def _reward(**overrides):
    """Call compute_race_reward from a neutral baseline, overriding as needed."""
    kwargs = dict(
        perception="lidar",
        n_perception=8,
        waypoints_len=10,
        waypoint_idx=1,
        checkpoint_hit=False,
        last_dist=50.0,
        current_dist=50.0,
        last_damage=0.0,
        steps=10,
        invuln_steps=0,
        steps_since_checkpoint=5,
        max_steps=500,
        max_damage=1000.0,
    )
    obs = overrides.pop("obs", None)
    if obs is None:
        obs = _obs()
    kwargs.update(overrides)
    return compute_race_reward(obs, **kwargs)


class TestTimePressure:
    """The point of the rewrite: over the same path, faster must score higher. The
    reward this replaced had no time term at all, so it could not express that."""

    def test_every_step_costs(self):
        assert _reward(obs=_obs(speed=0.0)).reward == pytest.approx(-reward_mod.STEP_PENALTY)

    def test_a_fast_traversal_outscores_a_slow_one(self):
        def traverse(n_steps: int) -> float:
            per_step = 100.0 / n_steps  # metres closed per step
            total, last = 0.0, 100.0
            for i in range(n_steps):
                now = last - per_step
                total += _reward(
                    obs=_obs(speed=per_step / 5.0),  # speed consistent with the pace
                    last_dist=last,
                    current_dist=now,
                    steps=i + 1,
                ).reward
                last = now
            return total

        assert traverse(20) > traverse(60)

    def test_standing_still_loses_to_crawling_forward(self):
        still = _reward(obs=_obs(speed=0.0), last_dist=50.0, current_dist=50.0).reward
        crawl = _reward(obs=_obs(speed=0.02), last_dist=50.0, current_dist=49.9).reward
        assert crawl > still


class TestCheckpointBonus:
    def test_bonus_is_flat_not_growing_with_index(self):
        """A growing bonus made late-path reward dwarf early-path reward for
        identical driving, which destabilises the value function."""
        early = _reward(checkpoint_hit=True, waypoint_idx=1).reward
        late = _reward(checkpoint_hit=True, waypoint_idx=8).reward
        assert early == pytest.approx(late)

    def test_reaching_a_checkpoint_sooner_pays_more(self):
        quick = _reward(checkpoint_hit=True, steps_since_checkpoint=5).reward
        slow = _reward(checkpoint_hit=True, steps_since_checkpoint=25).reward
        assert quick > slow

    def test_segment_bonus_floors_at_zero_past_par(self):
        # SEGMENT_TARGET_STEPS/SEGMENT_TIME_COEF are gone (the segment bonus is now
        # relative, see TestSegmentTimeBonus in test_beamng_reward_path.py); this
        # computes the same "steps for the fallback 25 m segment" par live.
        par = reward_mod.beamng_spec.steps_for_distance(
            reward_mod.SPARSE_SPACING_M, reward_mod.SEGMENT_PAR_SPEED_MS
        )
        at_par = _reward(checkpoint_hit=True, steps_since_checkpoint=par).reward
        way_over = _reward(checkpoint_hit=True, steps_since_checkpoint=500).reward
        assert at_par == pytest.approx(way_over)

    def test_counter_resets_on_a_hit(self):
        assert _reward(checkpoint_hit=True, steps_since_checkpoint=9).steps_since_checkpoint == 0

    def test_counter_advances_without_a_hit(self):
        assert _reward(steps_since_checkpoint=9).steps_since_checkpoint == 10


class TestFinish:
    def test_finishing_ends_the_episode_and_pays_a_bonus(self):
        out = _reward(waypoint_idx=10, waypoints_len=10)
        assert out.finished is True
        assert out.done is True
        assert out.reward > reward_mod.FINISH_BONUS

    def test_finishing_with_more_budget_left_pays_more(self):
        early = _reward(waypoint_idx=10, waypoints_len=10, steps=100).reward
        late = _reward(waypoint_idx=10, waypoints_len=10, steps=400).reward
        assert early > late

    def test_not_finished_before_the_last_checkpoint(self):
        assert _reward(waypoint_idx=9, waypoints_len=10).finished is False

    def test_laps_scales_the_finish_line(self):
        # laps stays 1 until closed circuits land, but the arithmetic is already here.
        assert _reward(waypoint_idx=10, waypoints_len=10, laps=2).finished is False
        assert _reward(waypoint_idx=20, waypoints_len=10, laps=2).finished is True


class TestContactIsPricedNotFatal:
    def test_a_hard_hit_costs_but_does_not_end_the_race(self):
        out = _reward(obs=_obs(damage_norm=0.2), last_damage=0.0)  # 200 damage in one step
        assert out.reward < 0
        assert out.done is False, "rubbing wheels mid-overtake must not end the race"

    def test_writing_the_car_off_still_ends_it(self):
        out = _reward(obs=_obs(damage_norm=1.0), last_damage=0.0)  # == MAX_DAMAGE
        assert out.done is True
        assert out.reward < -reward_mod.CRASH_PENALTY / 2

    def test_damage_is_forgiven_during_the_checkpoint_grace_window(self):
        out = _reward(obs=_obs(damage_norm=0.05), last_damage=0.0, invuln_steps=2)
        assert out.reward == pytest.approx(CLEAN)


class TestGapTerm:
    def test_absent_without_rival_progress(self):
        assert _reward().reward == pytest.approx(CLEAN)

    def test_gaining_ground_is_rewarded(self):
        # progress_m/last_progress_m are one measurement doing two jobs (see
        # beamng_reward's docstring): the 10 m of own progress also pays the pace
        # term, on top of the 10 m of gap gained on a rival that did not move.
        out = _reward(
            progress_m=110.0,
            last_progress_m=100.0,
            rival_progress_m=100.0,
            last_rival_progress_m=100.0,
        )
        assert out.reward == pytest.approx(
            CLEAN + 10.0 * reward_mod.PROGRESS_COEF + 10.0 * reward_mod.GAP_COEF
        )

    def test_losing_ground_is_penalised(self):
        out = _reward(
            progress_m=100.0,
            last_progress_m=100.0,
            rival_progress_m=110.0,
            last_rival_progress_m=100.0,
        )
        assert out.reward == pytest.approx(CLEAN - 10.0 * reward_mod.GAP_COEF)

    def test_matching_the_rival_pace_is_neutral(self):
        """Defending is worth as much as attacking: both cars gaining equally leaves
        the gap term at zero. (The pace term still pays for the 20 m actually
        covered — it is the GAP contribution, not the total reward, that is
        neutral here.)"""
        out = _reward(
            progress_m=120.0,
            last_progress_m=100.0,
            rival_progress_m=120.0,
            last_rival_progress_m=100.0,
        )
        assert out.reward == pytest.approx(CLEAN + 20.0 * reward_mod.PROGRESS_COEF)

    def test_telescopes_to_the_final_gap(self):
        """Summed over an episode the GAP term equals GAP_COEF x the final gap, so
        it cannot be farmed by oscillating alongside the rival. The pace term
        telescopes the same way over the same progress_m readings, to
        PROGRESS_COEF x the total distance covered."""
        mine = [0.0, 10.0, 15.0, 12.0, 40.0]
        theirs = [0.0, 12.0, 14.0, 20.0, 25.0]
        total = 0.0
        for i in range(1, len(mine)):
            out = _reward(
                progress_m=mine[i],
                last_progress_m=mine[i - 1],
                rival_progress_m=theirs[i],
                last_rival_progress_m=theirs[i - 1],
            )
            total += out.reward - CLEAN
        final_gap = (mine[-1] - theirs[-1]) - (mine[0] - theirs[0])
        own_progress = mine[-1] - mine[0]
        assert total == pytest.approx(
            final_gap * reward_mod.GAP_COEF + own_progress * reward_mod.PROGRESS_COEF
        )

    def test_rival_finishing_first_ends_the_race_with_a_penalty(self):
        out = _reward(
            rival_finished=True,
            progress_m=50.0,
            last_progress_m=50.0,
            rival_progress_m=50.0,
            last_rival_progress_m=50.0,
        )
        assert out.done is True
        assert out.reward == pytest.approx(CLEAN - reward_mod.LOSE_PENALTY)

    def test_winning_pays_the_win_bonus_on_top_of_the_finish(self):
        solo = _reward(waypoint_idx=10, waypoints_len=10).reward
        won = _reward(
            waypoint_idx=10,
            waypoints_len=10,
            progress_m=50.0,
            last_progress_m=50.0,
            rival_progress_m=40.0,
            last_rival_progress_m=40.0,
        ).reward
        assert won == pytest.approx(solo + reward_mod.WIN_BONUS)

    def test_partial_progress_arguments_do_not_half_apply_the_term(self):
        # Missing the two rival_* arguments means the GAP is unknowable; it must
        # be skipped rather than computed against a None-as-zero. progress_m and
        # last_progress_m are supplied here, though, so the pace term still fires
        # on its own — it does not need a rival to mean something.
        out = _reward(progress_m=110.0, last_progress_m=100.0)
        assert out.reward == pytest.approx(CLEAN + 10.0 * reward_mod.PROGRESS_COEF)


class TestNoOffTrackRule:
    """Distance to the next checkpoint must not end or penalise an episode.

    A rule used to do both — a graded penalty past 200 m, a -100 and `done` past
    300 m — treating "far from the next checkpoint" as "lost". That only held for
    generated paths, whose checkpoints sit 25 m apart. The game's own sprint and lap
    tracks are marked out far more sparsely (30 of 44 have a gap over 300 m; italy's
    `highway1` averages 1064 m), so the rule fired while the car was driving the
    racing line correctly and ended the episode for doing the right thing.
    """

    def test_the_reward_takes_no_checkpoint_distance_at_all(self):
        # The strongest guard: the argument is gone, so the rule cannot be revived
        # by accident, and a caller still passing it fails loudly.
        import inspect

        params = inspect.signature(compute_race_reward).parameters
        assert "checkpoint_dist" not in params
        assert "warn_dist" not in params
        assert "reset_dist" not in params

    def test_a_wide_checkpoint_gap_neither_penalises_nor_ends_the_episode(self):
        # Nothing distinguishes a car 1 km from its next checkpoint (normal on
        # highway1) from one 10 m away: same clean-step reward, still running.
        out = _reward()
        assert out.done is False
        assert out.reward == pytest.approx(CLEAN)

    def test_wandering_is_still_bounded_by_the_step_budget(self):
        # Removing the rule must not make a lost car free to wander forever:
        # exhausting max_steps still ends the episode.
        assert _reward(steps=500, max_steps=500).done is True

    def test_going_nowhere_still_costs(self):
        # And it is not free per step either: with no progress the step penalty is
        # the entire reward, so a lost car bleeds score until the budget runs out.
        out = _reward(obs=_obs(speed=0.0), last_dist=50.0, current_dist=50.0)
        assert out.reward == pytest.approx(-reward_mod.STEP_PENALTY)
