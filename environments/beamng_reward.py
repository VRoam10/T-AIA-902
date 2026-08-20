"""The racing reward, shared by the single-vehicle, multi-vehicle and race envs.

All three call `compute_race_reward`, so a policy is rewarded for the same
behaviour whether it trains alone, alongside others, or against a rival.

What it rewards, and why:

  * **Speed, not just arrival.** Dense progress along the path (x3), speed
    projected onto the target direction (x3), a flat checkpoint bonus, a relative
    bonus for beating the segment's own par, a finish bonus with a relative pace
    term of the same shape — and a penalty on every step. Time pressure
    is the point: the reward this replaced paid
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

# The segment-time bonus is *relative*: beat par and it pays up to
# SEGMENT_TIME_BONUS, miss it and it floors at zero. Par is derived per segment
# from the geometry actually being driven — a constant par (it used to be
# SPARSE_SPACING_M, 25 m) is unmeetable on a game track, where 30 of the 44
# shipped sprint/lap tracks have gaps over 300 m. Because par scales with the
# segment's own length, `steps_since_checkpoint / par` is a pace ratio (equal to
# SEGMENT_PAR_SPEED_MS / actual average speed) and the bonus stays scale-free:
# the same relative pace pays the same on a 25 m segment and a 1000 m one, to
# within par's ceil() rounding on short segments.
#
# The target speed is deliberately modest so the bonus stays a usable gradient
# across the pace a car actually drives, rather than saturating at the cap the
# moment it is going reasonably fast. That rules out the tempting-looking
# alternative of comparing par to steps as a *ratio past 1* (par / steps, capped
# at 2x par speed): on this segment length range a 682 hp car clears 2x par speed
# (24 m/s, 86 km/h) on nearly every segment, so that shape is flat — no
# gradient — across the entire operating regime, which is arithmetically the
# same as folding it into a bigger CHECKPOINT_BONUS.
#
# What this replaces — "(par - steps) x a coefficient" with a *fixed* par — is
# not scale-free at all: on italy's highway1 (1064 m average segment, par ~266
# steps at SEGMENT_PAR_SPEED_MS), a 40 m/s run takes ~80 steps and that shape
# would have paid ~744, more than the checkpoint and finish bonuses combined.
SEGMENT_PAR_SPEED_MS = 12.0
SEGMENT_TIME_BONUS = 25.0  # half a checkpoint bonus for a perfect segment
FINISH_BONUS = 300.0  # for completing the path
# The finish's time bonus is relative for exactly the reason the segment one is,
# and was left absolute by oversight when the segment term was rewritten. It paid
# ``1.0 x (max_steps - steps)``, and because max_steps is a constant 5000 while
# the shipped paths run from 65 m to 10.7 km, "unused steps" was a flat ~5000 for
# completing *anything*. Measured by replaying the cached paths through this
# function: finishing east_coast_usa's 75.5 m path in 15 steps paid 5285 of a
# 6104-point episode — 87% of it, 81 reward per metre — against 1.3 reward per
# metre for driving gridmap_v2's 1767 m path well. So the shortest path in the
# pool was the whole game, and the per-step driving signal (7-90) was invisible
# beside one terminal spike, which is all a critic then has to learn from. Par
# now comes from the path's own length, so the same relative pace pays the same
# whatever the distance.
FINISH_TIME_BONUS = 150.0  # sliding scale: 0 at par pace, approaching this as time -> 0

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
    progress_m: float = 0.0
    # Checkpoints reached this episode, as it stood on entry. Report this, never
    # ``waypoint_idx``: the finish zeroes that, and callers build their ``info``
    # dict *after* the reward, so the metric read 0 on exactly the episodes that
    # completed the path. That blanked the checkpoint panel of the training plot
    # and hid a path being finished in 14 steps.
    checkpoints_reached: int = 0


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
    path_alignment: float | None = None,
    segment_len_m: float | None = None,
    path_length_m: float | None = None,
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

    ``progress_m``/``last_progress_m`` are one measurement doing two jobs: the
    pace term reads the metres gained along the path directly (falling back to
    straight-line closure when the caller has no projection), and the same pair
    feeds the racing gap term together with the two ``rival_*`` arguments — pass
    none of the rival ones (the default) for solo running and the gap term
    contributes exactly nothing. `path_alignment` overrides the checkpoint-bearing
    alignment with cos(heading - path tangent); `segment_len_m` sets the
    segment-time bonus's par distance and `path_length_m` the finish bonus's
    (without the latter, par falls back to ``waypoints_len x SPARSE_SPACING_M``,
    which is the true distance only for a generated path). `rival_finished`
    settles the win/lose bonus when the other car got there first.

    All other arguments are the caller's current episode state; the returned
    `RewardOutcome` holds the updated values to write back.
    """
    speed, _steering, heading_err, _lateral_err, damage_norm = obs[:5]
    perception_bins = obs[6 : 6 + n_perception]
    damage = damage_norm * 1000.0
    # Speed is projected onto the direction we want to be going. With a path
    # tangent that is where the road goes; without one it falls back to the
    # bearing to the next checkpoint (heading_err is normalized by pi in the
    # observation; undo it for cos()) — the same thing only while checkpoints
    # are close.
    if path_alignment is None:
        alignment = float(np.cos(heading_err * np.pi))
    else:
        alignment = float(np.clip(path_alignment, -1.0, 1.0))

    done = False
    finished = False
    reward = 0.0
    checkpoints_reached = int(waypoint_idx)

    # Invulnerability grace: a checkpoint hit (re)starts a short window in which
    # damage is ignored. The hit step itself is covered.
    if checkpoint_hit:
        invuln_steps = INVULN_GRACE_STEPS
    invulnerable = invuln_steps > 0
    if invulnerable:
        invuln_steps -= 1

    # 1. Progress. Metres gained *along the path* when the caller measures it,
    #    which is continuous across a checkpoint and cannot read a bend as going
    #    backwards. Falling back to straight-line closure keeps the old
    #    behaviour — including zeroing the hit step, where the target jumping to
    #    the next waypoint would otherwise look like a large step backwards.
    if progress_m is not None and last_progress_m is not None:
        reward += (float(progress_m) - float(last_progress_m)) * PROGRESS_COEF
    else:
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

    # 7. Checkpoint: a flat bonus plus a relative bonus for beating the segment's
    #    own par. `par` is the step count for covering this segment's real length
    #    at SEGMENT_PAR_SPEED_MS (falling back to SPARSE_SPACING_M without a
    #    measured length); `steps_since_checkpoint / par` is a pace ratio, so a
    #    near-perfect 25 m segment and a near-perfect 1000 m segment pay the same
    #    instead of one dwarfing the other, and the bonus still has a gradient at
    #    race pace instead of saturating the moment a car is going reasonably fast.
    steps_since_checkpoint += 1
    if checkpoint_hit:
        reward += CHECKPOINT_BONUS
        par = beamng_spec.steps_for_distance(
            float(segment_len_m) if segment_len_m else SPARSE_SPACING_M, SEGMENT_PAR_SPEED_MS
        )
        reward += SEGMENT_TIME_BONUS * float(np.clip(1.0 - steps_since_checkpoint / par, 0.0, 1.0))
        steps_since_checkpoint = 0
        checkpoint_hit = False

    # 8. Finish: the flat completion bonus plus a *relative* pace bonus, on the
    #    same scale-free shape as the segment bonus — par from the distance the
    #    run actually covered (x laps), not from what is left of a constant step
    #    budget. `waypoint_idx` is zeroed so a slot stepped again before its reset
    #    cannot re-fire the bonus; `checkpoints_reached` keeps the count for the
    #    caller's metrics.
    if waypoints_len and waypoint_idx >= waypoints_len * max(1, laps):
        reward += FINISH_BONUS
        driven_m = float(path_length_m) if path_length_m else waypoints_len * SPARSE_SPACING_M
        par = beamng_spec.steps_for_distance(driven_m * max(1, laps), SEGMENT_PAR_SPEED_MS)
        reward += FINISH_TIME_BONUS * float(np.clip(1.0 - steps / par, 0.0, 1.0))
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
        progress_m=float(progress_m) if progress_m is not None else 0.0,
        checkpoints_reached=checkpoints_reached,
    )
