"""Pure, stateless LiDAR geometry helpers shared by the BeamNG environments.

Extracted from environments.beamng so the single-vehicle env and the
multi-vehicle env use one implementation. No `self`, no BeamNG connection,
no logging side effects — callers handle those.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarConfig:
    """Binning + filtering parameters for a LiDAR sensor.

    Built once from an environment's class constants and passed to
    :func:`process_lidar`.
    """

    rays: int  # horizontal azimuth bins
    v_bins: int  # vertical elevation bins (1 = legacy single row)
    channels: int  # values stored per cell (currently 1: distance)
    fov_deg: float  # total forward azimuth field of view
    vert_angle: float  # total vertical field of view (used when v_bins > 1)
    max_dist: float  # metres — normalization range
    self_margin: float  # metres — ego OBB expansion for self-hit rejection
    ground_clearance: float  # metres above floor before a point counts as obstacle


def ego_local_extents_from_bbox(bbox, state, margin):
    """Return ego OBB extents in vehicle-local frame, or None.

    Tuple layout: (x_min, x_max, y_min, y_max, z_min, z_max), each already
    expanded by ``margin``. Returns None when bbox or pos is missing — callers
    fall back to a flat ground threshold.
    """
    if not bbox or "pos" not in state:
        return None

    corners = np.asarray(list(bbox.values()), dtype=np.float32)
    pos = np.asarray(state.get("pos", (0.0, 0.0, 0.0)), dtype=np.float32)
    dir_vec = np.asarray(state.get("dir", (1.0, 0.0, 0.0)), dtype=np.float32)
    heading = float(np.arctan2(dir_vec[1], dir_vec[0]))

    rel = corners - pos
    c, s = np.cos(-heading), np.sin(-heading)
    lx = rel[:, 0] * c - rel[:, 1] * s
    ly = rel[:, 0] * s + rel[:, 1] * c
    lz = rel[:, 2]
    return (
        float(lx.min() - margin),
        float(lx.max() + margin),
        float(ly.min() - margin),
        float(ly.max() + margin),
        float(lz.min() - margin),
        float(lz.max() + margin),
    )


def world_to_local(points, pos, heading):
    """Transform Nx3 world points into the vehicle-local frame.

    Returns (local_x, local_y, local_z) as separate 1-D arrays.
    """
    rel = points - np.asarray(pos, dtype=np.float32)
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    local_x = rel[:, 0] * cos_h - rel[:, 1] * sin_h
    local_y = rel[:, 0] * sin_h + rel[:, 1] * cos_h
    local_z = rel[:, 2]
    return local_x, local_y, local_z


def lidar_keep_mask(local_x, local_y, local_z, ego_extents, self_margin, ground_clearance):
    """Reject points inside the ego OBB or below the ground threshold.

    Returns (keep_mask, debug_dict). ``ground_clearance`` is measured above the
    true bbox floor (z_min + self_margin) when extents are known, else above 0.
    """
    n_total = int(local_x.size)
    inside_self = np.zeros(n_total, dtype=bool)

    if ego_extents is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = ego_extents
        inside_self = (
            (local_x >= x_min)
            & (local_x <= x_max)
            & (local_y >= y_min)
            & (local_y <= y_max)
            & (local_z >= z_min)
            & (local_z <= z_max)
        )
        floor = z_min + self_margin
        ground_z = floor + ground_clearance
    else:
        ground_z = ground_clearance

    below_ground = local_z <= ground_z
    keep = ~inside_self & ~below_ground

    debug = {
        "total": n_total,
        "self": int(inside_self.sum()),
        "ground": int((below_ground & ~inside_self).sum()),
        "kept": int(keep.sum()),
        "extents_none": ego_extents is None,
        "ground_z": float(ground_z),
    }
    return keep, debug


def process_lidar(point_cloud, vehicle_pos, vehicle_heading, ego_extents, cfg):
    """Bin a raw LiDAR point cloud into a (v_bins x rays x channels) grid.

    Returns (distances, debug). ``distances`` is a flat float32 array in [0, 1]
    where 0 means an obstacle is right there and 1 means clear. ``debug`` holds
    filtering counts plus the nearest in-FOV point's distance/height.
    """
    v_bins = cfg.v_bins
    h_bins = cfg.rays
    ch = cfg.channels
    n_out = v_bins * h_bins * ch
    distances = np.ones(n_out, dtype=np.float32)
    debug = {}

    if point_cloud is None or len(point_cloud) == 0:
        return distances, debug

    pts = np.asarray(point_cloud, dtype=np.float32).reshape(-1, 3)
    local_x, local_y, local_z = world_to_local(pts, vehicle_pos, vehicle_heading)

    keep, debug = lidar_keep_mask(
        local_x, local_y, local_z, ego_extents, cfg.self_margin, cfg.ground_clearance
    )
    local_x = local_x[keep]
    local_y = local_y[keep]
    local_z = local_z[keep]
    if local_x.size == 0:
        return distances, debug

    angles = np.arctan2(local_y, local_x)
    dists = np.hypot(local_x, local_y)

    half_fov = np.radians(cfg.fov_deg / 2.0)
    in_fov = np.abs(angles) <= half_fov
    angles = angles[in_fov]
    dists = dists[in_fov]
    local_z = local_z[in_fov]
    if angles.size == 0:
        return distances, debug

    nearest = int(np.argmin(dists))
    debug["fov"] = int(angles.size)
    debug["min_dist_m"] = float(dists[nearest])
    debug["min_dist_z"] = float(local_z[nearest])

    h_edges = np.linspace(-half_fov, half_fov, h_bins + 1)
    h_idx = np.clip(np.digitize(angles, h_edges) - 1, 0, h_bins - 1)

    if v_bins == 1:
        v_idx = np.zeros(angles.shape, dtype=np.intp)
    else:
        half_vfov = np.radians(cfg.vert_angle / 2.0)
        elevation = np.arctan2(local_z, dists)
        v_edges = np.linspace(-half_vfov, half_vfov, v_bins + 1)
        v_idx = np.clip(np.digitize(elevation, v_edges) - 1, 0, v_bins - 1)

    for v in range(v_bins):
        for h in range(h_bins):
            sel = dists[(v_idx == v) & (h_idx == h)]
            if sel.size:
                distances[(v * h_bins + h) * ch] = np.clip(sel.min() / cfg.max_dist, 0.0, 1.0)

    return distances, debug
