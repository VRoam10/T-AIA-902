"""The fine spawn height: where the car actually comes to rest.

Two things set a spawn's height, and they are deliberately separate:

  * ``core.trajectory.SPAWN_Z_OFFSET_M`` — the coarse, global height above the road
    centerline. Now 0.0: it used to be 1.0 on the assumption that the scenario's
    ``cling=True`` would drop the car the rest of the way, and in-sim it does not.
  * this module — the remaining centimetres, measured per scenario rather than
    guessed, because the car's ride height is not knowable from level geometry and
    ``beamng_spec.RACE_CAR`` could change to a model that rests differently.

The measurement needs no extra probing. Once the scenario is running, the car's
settled position IS the ground truth for that spot, and the delta between it and
the cached height is the correction to add to every later teleport target.

Teleports are what need this. ``Vehicle.teleport`` has no cling parameter — it puts
the vehicle's reference point at exactly the z it is given — so any error in the
cached height is a drop onto stiff race suspension on every single reset, which
BeamNG scores as damage and which accumulates into a broken engine.
"""

from __future__ import annotations

import math

from environments import beamng_spec

# One settle attempt advances this many physics steps, and we make at most this
# many attempts before giving up — together ~1 s of sim time, comfortably longer
# than the suspension needs to stop moving after cling drops the car.
SETTLE_STEPS = 5
SETTLE_TRIES = 6

# Vertical speed below which the car counts as resting. Reading the height while
# it is still falling would bake the fall into the correction.
RESTING_VZ_MS = 0.25

# With the coarse offset at 0 the spawn is already on the road surface, so a
# plausible correction is only the car's ride height — tens of centimetres. Cap it
# well under a metre: anything bigger means the reading is not what we think it is
# (the car is mid-air, a bad poll, the wrong reference frame), and it must not be
# allowed to silently undo the global lowering.
MAX_CORRECTION_M = 1.0


def _component(values, index: int) -> float | None:
    """``values[index]`` as a finite float, or None if it is unusable.

    Vehicle state arrives from the simulator, and a poll before the scenario is
    ready yields missing keys or non-numeric placeholders rather than an error.
    """
    if values is None:
        return None
    try:
        out = float(values[index])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    return out if math.isfinite(out) else None


def measure_spawn_z_correction(bng, vehicle, cached_z: float) -> float:
    """Delta to add to a cached spawn z so a teleport lands where cling did.

    Call once per scenario load, with ``vehicle`` sitting at the clung spawn whose
    cached height is ``cached_z``. Steps the simulation until the car stops moving
    vertically, then returns ``settled_z - cached_z``.

    Returns 0.0 — keeping the cached height, i.e. the old behaviour — whenever the
    measurement cannot be trusted: the vehicle never comes to rest, the state is
    unreadable, the simulator raises, or the result is implausibly large.
    """
    for _ in range(SETTLE_TRIES):
        try:
            vehicle.poll_sensors()
        except Exception as exc:  # noqa: BLE001 — any poll failure means no measurement
            print(f"[spawn] height measurement skipped (poll failed: {exc})")
            return 0.0

        state = vehicle.state or {}
        z = _component(state.get("pos"), 2)
        if z is None:
            print("[spawn] height measurement skipped (no position in vehicle state)")
            return 0.0

        vz = _component(state.get("vel"), 2)
        if vz is None or abs(vz) <= RESTING_VZ_MS:
            correction = z - float(cached_z)
            if abs(correction) > MAX_CORRECTION_M:
                print(
                    f"[spawn] ignoring implausible height correction {correction:+.2f} m "
                    f"(settled z {z:.2f}, cached z {float(cached_z):.2f})"
                )
                return 0.0
            return correction

        try:
            bng.step(SETTLE_STEPS)
        except Exception as exc:  # noqa: BLE001 — cannot settle, so cannot measure
            print(f"[spawn] height measurement skipped (step failed: {exc})")
            return 0.0

    print(
        f"[spawn] vehicle never came to rest in "
        f"{SETTLE_TRIES * SETTLE_STEPS / beamng_spec.PHYSICS_STEPS_PER_SECOND:.1f} s; "
        "keeping the cached spawn height"
    )
    return 0.0


def corrected_spawn(spawn_pos, correction: float) -> tuple[float, float, float]:
    """``spawn_pos`` with its z shifted by ``correction``.

    Teleport targets go through here so the car is placed at the height it rests
    at, instead of being dropped from the cache's geometric estimate.
    """
    return (float(spawn_pos[0]), float(spawn_pos[1]), float(spawn_pos[2]) + correction)
