"""Single merged reward shared by the single-vehicle and multi-vehicle envs.

Both `BeamNGDrivingEnv` (and its subclasses) and `BeamNGMultiEnv` call
`compute_merged_reward` so the two environments reward identical behaviour.
It combines the good parts of the old `default` and `ddpg` rewards:

  * dense progress shaping toward the next waypoint (telescoping, ``x3``)
  * speed projected onto the target direction (``x3``)
  * increasing checkpoint bonus (``100 * waypoint_idx``) + lap bonus
  * graded damage penalty and a big crash penalty
  * LiDAR obstacle-proximity penalty (skipped for camera perception)

Two design fixes over the old rewards:
  * the exploitable flat ``alignment * 0.5`` term is gone, so a vehicle can no
    longer farm reward by hovering near a checkpoint;
  * the progress term is zeroed on the checkpoint-hit step, so the target
    switching to the next (far) waypoint no longer registers as a large
    backward jump — reaching a checkpoint is unambiguously good.

Reaching a checkpoint also grants a short damage-invulnerability window
(``INVULN_GRACE_STEPS``) so brushing/settling right at the checkpoint is not
punished.
"""

from dataclasses import dataclass

import numpy as np

# --- Reward tunables (shared by both environments) --------------------------
PROGRESS_COEF = 3.0  # weight on distance closed toward the waypoint
SPEED_ALIGN_COEF = 3.0  # weight on speed projected onto the target direction
STATIONARY_SPEED = 0.05  # normalized speed below which the vehicle counts as stopped
STATIONARY_PENALTY = 1.0
CHECKPOINT_BONUS_PER_IDX = 100.0  # bonus = this * waypoint_idx (grows per checkpoint)
LAP_BONUS = 200.0
DAMAGE_DELTA_COEF = 0.3  # penalty per unit of new damage
HARD_HIT_DAMAGE = 150.0  # single-step damage that counts as a hard hit
HARD_HIT_PENALTY = 30.0
CRASH_PENALTY = 1000.0  # penalty when total damage reaches MAX_DAMAGE
INVULN_GRACE_STEPS = 3  # damage-immune steps granted on a checkpoint hit (~3 s)
OFF_TRACK_WARN_PENALTY = 10.0  # max graded penalty between warn and reset distance
OFF_TRACK_RESET_PENALTY = 100.0  # penalty + episode end past the reset distance

_LIDAR_PERCEPTIONS = ("lidar", "lidar_grid")


@dataclass
class RewardOutcome:
    """Result of `compute_merged_reward`.

    Carries the reward/done plus the episode-state fields the caller must write
    back to its own storage (``self._*`` for the single env, ``slot.*`` for the
    multi env), keeping all reward logic in one place.
    """

    reward: float
    done: bool
    last_dist: float
    last_damage: float
    invuln_steps: int
    checkpoint_hit: bool
    waypoint_idx: int


def compute_merged_reward(
    obs,
    *,
    perception: str,
    waypoints_len: int,
    waypoint_idx: int,
    checkpoint_hit: bool,
    last_dist: float,
    current_dist: float,
    checkpoint_dist: float,
    last_damage: float,
    steps: int,
    invuln_steps: int,
    max_steps: int,
    max_damage: float,
    warn_dist: float,
    reset_dist: float,
) -> RewardOutcome:
    """Compute the merged driving reward for one env step.

    `obs` is the normalized observation vector: the first five entries are
    ``speed, steering, heading_err, lateral_err, damage_norm`` and ``obs[5:]``
    is the perception block (LiDAR distance bins or camera pixels). `perception`
    selects whether ``obs[5:]`` is treated as LiDAR ranges for the obstacle
    penalty. `checkpoint_dist` is the distance to the current target checkpoint
    (for the off-track penalty), and `warn_dist`/`reset_dist` are its thresholds.
    All other arguments are the caller's current episode state; the returned
    `RewardOutcome` holds the updated values to write back.
    """
    speed, _steering, heading_err, _lateral_err, damage_norm = obs[:5]
    perception_bins = obs[5:]
    damage = damage_norm * 1000.0
    # heading_err is normalized by pi in the observation; undo it for cos().
    alignment = float(np.cos(heading_err * np.pi))

    done = False
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

    # 3. Penalise being stationary — the agent must move.
    if speed < STATIONARY_SPEED:
        reward -= STATIONARY_PENALTY

    # 4. Obstacle proximity (LiDAR distance bins only; a camera block is not a
    #    range field, so the penalty would be meaningless there).
    if perception in _LIDAR_PERCEPTIONS and perception_bins.size:
        min_lidar = float(np.min(perception_bins))
        if min_lidar < 0.2:
            reward -= (1.0 - min_lidar) * 5.0
        elif min_lidar < 0.4:
            reward -= (1.0 - min_lidar) * 2.0

    # 5. Damage — ignored entirely during the invulnerability grace window.
    if invulnerable:
        last_damage = damage  # forgive damage taken during grace; rebaseline
    else:
        damage_delta = damage - last_damage
        if damage_delta > 0:
            reward -= damage_delta * DAMAGE_DELTA_COEF
        if damage_delta > HARD_HIT_DAMAGE:
            reward -= HARD_HIT_PENALTY
            done = True
        if damage >= max_damage:
            reward -= CRASH_PENALTY
            done = True
        last_damage = damage

    # 6. Step limit.
    if steps >= max_steps:
        done = True

    # 7. Increasing checkpoint bonus (grows with each checkpoint reached).
    if checkpoint_hit:
        reward += CHECKPOINT_BONUS_PER_IDX * waypoint_idx
        checkpoint_hit = False

    # 8. Lap completion.
    if waypoint_idx >= waypoints_len:
        reward += LAP_BONUS
        waypoint_idx = 0
        done = True

    # 9. Off-track handling: graded penalty once far from the target checkpoint,
    #    hard reset (penalty + episode end) once hopelessly far — stops the
    #    vehicle from wandering into the void for the rest of the episode.
    if checkpoint_dist >= reset_dist:
        reward -= OFF_TRACK_RESET_PENALTY
        done = True
    elif checkpoint_dist >= warn_dist:
        reward -= (checkpoint_dist - warn_dist) / (reset_dist - warn_dist) * OFF_TRACK_WARN_PENALTY

    return RewardOutcome(
        reward=float(reward),
        done=done,
        last_dist=float(last_dist),
        last_damage=float(last_damage),
        invuln_steps=int(invuln_steps),
        checkpoint_hit=checkpoint_hit,
        waypoint_idx=waypoint_idx,
    )
