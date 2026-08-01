"""The racing reward, shared by the single-vehicle, multi-vehicle and race envs.

All three call `compute_race_reward`, so a policy is rewarded for the same
behaviour whether it trains alone, alongside others, or against a rival.

What it rewards, and why:

  * **Speed, not just arrival.** Dense progress toward the next waypoint (x3),
    speed projected onto the target direction (x3), a flat checkpoint bonus, a
    bonus for reaching each checkpoint *quickly*, a finish bonus that scales with
    how much of the step budget is left — and a penalty on every step. Time
    pressure is the point: the reward this replaced paid
    ``CHECKPOINT_BONUS_PER_IDX * waypoint_idx`` with no time term at all, so a slow
    clean run scored the same as a fast one over the same path.
  * **Being ahead.** In a race, a telescoping gap term pays for metres gained on
    the rival, so the episode total is ``GAP_COEF x final gap``. That single signal
    rewards overtaking and defending alike, with no per-step position bookkeeping.
  * **Staying in one piece.** A LiDAR obstacle-proximity penalty and damage costs —
    but contact is priced, not fatal: rubbing wheels mid-overtake must not end the
    race, while writing the car off still must.

There is deliberately **no off-track term**. One used to penalise, then end, an
episode once the car was 200/300 m from its target checkpoint, on the assumption
that being far from the next checkpoint means being lost. That assumption is a
property of the generated paths (checkpoints every 25 m), not of racing: the game's
own tracks are marked out far more sparsely, so the rule punished correct driving
and ended most sprint/lap episodes early. Wandering is now bounded by ``max_steps``
and by the step penalty, which cost a lost car the episode anyway.

Two exploit fixes carried over from the earlier reward:
  * there is no flat ``alignment`` term, so a vehicle cannot farm reward by
    hovering near a checkpoint;
  * the progress term is zeroed on the checkpoint-hit step, so the target
    switching to the next (far) waypoint does not register as a large backward
    jump — reaching a checkpoint is unambiguously good.

Reaching a checkpoint also grants a short damage-invulnerability window
(``INVULN_GRACE_STEPS``) so brushing/settling right at the checkpoint is not
punished.

Design note on the checkpoint bonus: it is deliberately *flat* rather than growing
with the checkpoint index. A growing bonus made late-path reward an order of
magnitude larger than early-path reward for identical driving, which destabilises
the value function; the segment-time bonus puts the incentive where it belongs —
reach *each* checkpoint sooner.
"""

from dataclasses import dataclass

import numpy as np

from core.trajectory import SPARSE_SPACING_M
from environments import beamng_spec

# --- Pace ---------------------------------------------------------------------
PROGRESS_COEF = 3.0  # weight on distance closed toward the waypoint
SPEED_ALIGN_COEF = 3.0  # weight on speed projected onto the target direction
STEP_PENALTY = 0.5  # charged every step — this is what makes "fast" beat "slow"

CHECKPOINT_BONUS = 50.0  # flat, per checkpoint reached

# "Par" for one checkpoint segment, derived from the real geometry rather than
# guessed: sparse checkpoints sit SPARSE_SPACING_M apart and one env step lasts
# SECONDS_PER_ENV_STEP, so par is the step count for covering that gap at
# SEGMENT_PAR_SPEED_MS. Beat par and the segment-time bonus pays; miss it and it
# floors at zero. The target speed is deliberately modest so the bonus stays a
# usable gradient early in training instead of saturating.
SEGMENT_PAR_SPEED_MS = 12.0
SEGMENT_TARGET_STEPS = beamng_spec.steps_for_distance(SPARSE_SPACING_M, SEGMENT_PAR_SPEED_MS)
# Scaled so a near-perfect segment is worth roughly half a checkpoint bonus.
SEGMENT_TIME_COEF = 4.0
FINISH_BONUS = 300.0  # for completing the path
FINISH_TIME_COEF = 1.0  # bonus per unused step at the finish

# --- Contact and damage -------------------------------------------------------
DAMAGE_DELTA_COEF = 0.15  # penalty per unit of new damage
HARD_HIT_DAMAGE = 150.0  # single-step damage that counts as a hard hit
HARD_HIT_PENALTY = 30.0  # priced, but does NOT end a race
CRASH_PENALTY = 1000.0  # penalty when total damage reaches MAX_DAMAGE
INVULN_GRACE_STEPS = 3  # damage-immune steps granted on a checkpoint hit (~3 s)

# Removed: the off-track handling that penalised, then ended, an episode once the
# car was 200/300 m from its target checkpoint. It used distance-to-next-checkpoint
# as a proxy for "lost", which only held for generated paths (checkpoints 25 m
# apart). The game's own tracks are marked out far more sparsely — 30 of the 44
# shipped sprint/lap tracks have a gap over 300 m, and italy's `highway1` averages
# 1064 m — so the proxy fired while the car was driving the racing line correctly,
# ending the episode with a large penalty for doing the right thing.

# --- Racing (only active when a rival's progress is supplied) -----------------
GAP_COEF = 5.0  # reward per metre gained on the rival
WIN_BONUS = 200.0  # finishing first
LOSE_PENALTY = 100.0  # the rival finished first

_LIDAR_PERCEPTIONS = ("lidar", "adv_lidar")


@dataclass
class RewardOutcome:
    """Result of `compute_race_reward`.

    Carries the reward/done plus the episode-state fields the caller must write
    back to its own storage (``self._*`` for the single env, ``slot.*`` for the
    multi/race envs), keeping all reward logic in one place.
    """

    reward: float
    done: bool
    last_dist: float
    last_damage: float
    invuln_steps: int
    checkpoint_hit: bool
    waypoint_idx: int
    steps_since_checkpoint: int
    finished: bool = False


def compute_race_reward(
    obs,
    *,
    perception: str,
    n_perception: int,
    waypoints_len: int,
    waypoint_idx: int,
    checkpoint_hit: bool,
    last_dist: float,
    current_dist: float,
    last_damage: float,
    steps: int,
    invuln_steps: int,
    max_steps: int,
    max_damage: float,
    steps_since_checkpoint: int = 0,
    laps: int = 1,
    progress_m: float | None = None,
    last_progress_m: float | None = None,
    rival_progress_m: float | None = None,
    last_rival_progress_m: float | None = None,
    rival_finished: bool = False,
) -> RewardOutcome:
    """Compute the racing reward for one env step.

    `obs` is the normalized observation vector: the first six entries are
    ``speed, steering, heading_err, lateral_err, damage_norm, dist_norm`` and the
    ``n_perception`` entries after them are the perception block (LiDAR distance
    bins or camera pixels); waypoint hints and extra features may follow and must
    not be read here. `perception` selects whether the block is treated as LiDAR
    ranges for the obstacle penalty.

    The four ``*_progress_m`` arguments enable the racing gap term; pass none of
    them (the default) for solo running and the term contributes exactly nothing.
    `rival_finished` settles the win/lose bonus when the other car got there first.

    All other arguments are the caller's current episode state; the returned
    `RewardOutcome` holds the updated values to write back.
    """
    speed, _steering, heading_err, _lateral_err, damage_norm = obs[:5]
    perception_bins = obs[6 : 6 + n_perception]
    damage = damage_norm * 1000.0
    # heading_err is normalized by pi in the observation; undo it for cos().
    alignment = float(np.cos(heading_err * np.pi))

    done = False
    finished = False
    reward = 0.0

    # Invulnerability grace: a checkpoint hit (re)starts a short window in which
    # damage is ignored. The hit step itself is covered.
    if checkpoint_hit:
        invuln_steps = INVULN_GRACE_STEPS
    invulnerable = invuln_steps > 0
    if invulnerable:
        invuln_steps -= 1

    # 1. Progress toward the waypoint (telescoping). Zeroed on the hit step so
    #    the target switching to the next, far waypoint is not counted as
    #    backward progress.
    dist_delta = 0.0 if checkpoint_hit else (last_dist - current_dist)
    reward += dist_delta * PROGRESS_COEF
    last_dist = current_dist

    # 2. Speed projected onto the target direction (driving toward it = good).
    reward += float(speed) * alignment * SPEED_ALIGN_COEF

    # 3. Time pressure. Every step costs, so a slower line over the same path
    #    always scores lower. This also subsumes the old stationary penalty: a
    #    stopped car earns no progress and still pays this.
    reward -= STEP_PENALTY

    # 4. Obstacle proximity (LiDAR distance bins only; a camera block is not a
    #    range field, so the penalty would be meaningless there).
    if perception in _LIDAR_PERCEPTIONS and perception_bins.size:
        min_lidar = float(np.min(perception_bins))
        if min_lidar < 0.2:
            reward -= (1.0 - min_lidar) * 5.0
        elif min_lidar < 0.4:
            reward -= (1.0 - min_lidar) * 2.0

    # 5. Damage — ignored entirely during the invulnerability grace window. A hard
    #    hit is charged for but does not end the episode: contact is part of racing.
    if invulnerable:
        last_damage = damage  # forgive damage taken during grace; rebaseline
    else:
        damage_delta = damage - last_damage
        if damage_delta > 0:
            reward -= damage_delta * DAMAGE_DELTA_COEF
        if damage_delta > HARD_HIT_DAMAGE:
            reward -= HARD_HIT_PENALTY
        if damage >= max_damage:
            reward -= CRASH_PENALTY
            done = True
        last_damage = damage

    # 6. Step limit.
    if steps >= max_steps:
        done = True

    # 7. Checkpoint: a flat bonus plus a bonus for getting there quickly.
    steps_since_checkpoint += 1
    if checkpoint_hit:
        reward += CHECKPOINT_BONUS
        reward += max(0.0, SEGMENT_TARGET_STEPS - steps_since_checkpoint) * SEGMENT_TIME_COEF
        steps_since_checkpoint = 0
        checkpoint_hit = False

    # 8. Finish: the strongest "go fast" signal — what is left of the step budget.
    if waypoints_len and waypoint_idx >= waypoints_len * max(1, laps):
        reward += FINISH_BONUS
        reward += max(0, max_steps - steps) * FINISH_TIME_COEF
        waypoint_idx = 0
        done = True
        finished = True

    # 9. Racing: metres gained on the rival this step. Telescoping, so summing it
    #    over an episode gives GAP_COEF x the final gap — one signal that rewards
    #    overtaking and defending equally.
    have_gap = None not in (progress_m, last_progress_m, rival_progress_m, last_rival_progress_m)
    if have_gap:
        gap_now = progress_m - rival_progress_m
        gap_before = last_progress_m - last_rival_progress_m
        reward += (gap_now - gap_before) * GAP_COEF

    if finished and rival_progress_m is not None:
        reward += WIN_BONUS
    elif rival_finished:
        reward -= LOSE_PENALTY
        done = True

    return RewardOutcome(
        reward=float(reward),
        done=done,
        last_dist=float(last_dist),
        last_damage=float(last_damage),
        invuln_steps=int(invuln_steps),
        checkpoint_hit=checkpoint_hit,
        waypoint_idx=waypoint_idx,
        steps_since_checkpoint=int(steps_since_checkpoint),
        finished=finished,
    )
