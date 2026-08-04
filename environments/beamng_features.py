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
    half_w = max(_finite(reading.get("halfWidth"), 3.0), 0.5)
    d_left = _finite(reading.get("dist2Left"), half_track_width)
    d_right = _finite(reading.get("dist2Right"), half_track_width)
    left = np.clip((d_left - half_track_width) / half_w, -1.0, 1.0)
    right = np.clip((d_right - half_track_width) / half_w, -1.0, 1.0)

    road_heading = np.clip(
        _finite(reading.get("headingAngle"), 0.0) / (math.pi / 2.0), -1.0, 1.0
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
      That reasoning, and the resulting sign, is **not yet confirmed in-sim** —
      nobody has watched ``lat_g`` swing through a real corner to check it.
    """
    elec = electrics or {}
    forces = gforces or {}

    ground = float(math.hypot(_finite(vel[0], 0.0), _finite(vel[1], 0.0)))
    wheelspeed = _finite(elec.get("wheelspeed"), 0.0)
    long_slip = np.clip((wheelspeed - ground) / max(ground, SLIP_REF_MS), -1.0, 1.0)

    slip_angle = 0.0
    if ground >= SLIP_ANGLE_MIN_SPEED_MS:
        heading = math.atan2(_finite(dir_vec[1], 0.0), _finite(dir_vec[0], 1.0))
        course = math.atan2(_finite(vel[1], 0.0), _finite(vel[0], 1.0))
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
