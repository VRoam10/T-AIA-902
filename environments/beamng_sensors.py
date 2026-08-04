"""Construction, polling and teardown of the BeamNG perception sensors.

This is the one place that builds beamngpy ``Lidar`` / ``Camera`` / ``RoadsSensor``
objects and turns their polls into observation feature blocks. It exists because
the single-vehicle env and the multi-vehicle env had two full copies of all of it
(mount resolution, creation kwargs, point-cloud binning, camera downsampling,
bounded sensor removal), which drifted — the single env still carried its own
inline point-cloud math while the multi env used the shared geometry helpers.

The pure math lives in :mod:`environments.beamng_geometry` (no beamngpy import);
the axes and sizes live in :mod:`environments.beamng_spec`. This module is the
thin, sim-facing layer between them: everything here needs a live ``bng`` or a
sensor handle, so it is exercised through the fake-``bng`` test doubles rather
than unit-tested directly.
"""

import threading

import numpy as np

try:
    from beamngpy.sensors import Camera, Lidar, RoadsSensor
except ImportError:  # tests and non-BeamNG hosts
    Camera = Lidar = RoadsSensor = None

from config import LIDAR_VISUALISE
from environments import beamng_spec
from environments.beamng_camera_util import process_camera_frame
from environments.beamng_geometry import (
    LidarConfig,
    ego_local_extents_from_bbox,
    process_lidar,
)

# --- Physical LiDAR mount / sweep parameters (shared by every env) -----------
LIDAR_FOV_DEG = 360.0  # total azimuth span; 360 = full ring
LIDAR_MAX_DIST = 50.0  # metres — normalization range
LIDAR_GROUND_CLEARANCE = 0.30  # metres above ego bbox floor before a point is an obstacle
LIDAR_SELF_MARGIN = 0.30  # metres of ego OBB expansion when rejecting self-hits
LIDAR_ROOF_CLEARANCE = LIDAR_SELF_MARGIN  # metres above the cached roof for the mount

# Vehicle-local centre-roof fallback, used only when bbox sampling fails.
LIDAR_MOUNT_POS = (0.0, 0.0, 2.4)
LIDAR_MOUNT_DIR = (0, -1, 0)  # forward in vehicle space
LIDAR_MOUNT_UP = (0, 0, 1)

# Plausible ego bbox half-extent (metres). Anything larger is world-scale garbage
# from a bad pose, so the box is discarded — a wrong box is worse than none.
BBOX_MAX_HALF_EXTENT = 10.0

SENSOR_REMOVE_TIMEOUT = 3.0


# --------------------------------------------------------------------------- #
# LiDAR
# --------------------------------------------------------------------------- #
def lidar_config(sensor: str) -> LidarConfig:
    """Binning/filtering config for a lidar sensor, for :func:`process_lidar`."""
    geom = beamng_spec.lidar_geometry(sensor)
    return LidarConfig(
        rays=beamng_spec.LIDAR_RAYS,
        v_bins=int(geom["v_bins"]),
        channels=beamng_spec.LIDAR_CHANNELS_PER_RAY,
        fov_deg=LIDAR_FOV_DEG,
        vert_angle=float(geom["vert_angle"]),
        max_dist=LIDAR_MAX_DIST,
        self_margin=LIDAR_SELF_MARGIN,
        ground_clearance=LIDAR_GROUND_CLEARANCE,
    )


def lidar_mount_pos(ego_extents) -> tuple[float, float, float]:
    """Vehicle-local mount centred just above the cached ego roof.

    ``ego_extents``' ``z_max`` is already margin-expanded; one more
    ``LIDAR_ROOF_CLEARANCE`` goes on top. Falls back to ``LIDAR_MOUNT_POS`` when
    bbox sampling failed — which matters for the race car, since a hardcoded
    2.4 m mount sits well above a car this low.
    """
    if ego_extents is None:
        return LIDAR_MOUNT_POS
    _, _, _, _, _, z_max = ego_extents
    return (0.0, 0.0, float(z_max + LIDAR_ROOF_CLEARANCE))


def lidar_creation_kwargs(sensor: str, ego_extents) -> dict:
    """BeamNGpy ``Lidar`` kwargs for a sensor, shared by env creation and tests."""
    geom = beamng_spec.lidar_geometry(sensor)
    return {
        "pos": lidar_mount_pos(ego_extents),
        "dir": LIDAR_MOUNT_DIR,
        "up": LIDAR_MOUNT_UP,
        "requested_update_time": 0.05,
        "frequency": 30,
        "vertical_resolution": int(geom["vert_res"]),
        "vertical_angle": float(geom["vert_angle"]),
        "horizontal_angle": LIDAR_FOV_DEG,
        "max_distance": LIDAR_MAX_DIST,
        "is_360_mode": True,
        "is_rotate_mode": False,
        "is_using_shared_memory": False,
        "is_visualised": LIDAR_VISUALISE,
        "is_snapping_desired": False,
        "is_force_inside_triangle": False,
    }


def create_lidar(name: str, bng, vehicle, sensor: str, ego_extents):
    """Create a LiDAR bound to ``vehicle``. Must run after the scenario starts."""
    return Lidar(name, bng, vehicle, **lidar_creation_kwargs(sensor, ego_extents))


def poll_lidar(lidar, pos, heading, ego_extents, sensor: str):
    """Poll a LiDAR and bin it into the sensor's feature block.

    Returns ``(bins, debug)``. ``bins`` is a flat float32 array in [0, 1] where 0
    means an obstacle is right there and 1 means clear; a missing sensor yields
    all-clear rather than raising, so a sensor hiccup never kills an episode.
    """
    point_cloud = lidar.poll().get("pointCloud", None) if lidar is not None else None
    return process_lidar(point_cloud, pos, heading, ego_extents, lidar_config(sensor))


def cache_ego_local_bbox(vehicle):
    """Sample the ego OBB once and return its vehicle-local extents, or None.

    The bbox is queried in world space, so state must be fresh — ``get_bbox()``
    without a preceding poll returns coordinates we cannot de-rotate (state is
    None at scenario-load time and pos would default to the origin, producing
    world-scale garbage extents that silently disable the LiDAR). Expressed
    relative to the vehicle and de-rotated by heading, the extents are invariant
    under rigid motion, so once per scenario load is enough.
    """
    try:
        vehicle.poll_sensors()
        bbox = vehicle.get_bbox()
    except Exception:
        return None
    return ego_local_extents_from_bbox(
        bbox, vehicle.state or {}, LIDAR_SELF_MARGIN, BBOX_MAX_HALF_EXTENT
    )


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
def create_camera(name: str, bng, vehicle, visualise: bool = False):
    """Create the forward dashcam used by the ``camera`` sensor."""
    return Camera(
        name,
        bng,
        vehicle,
        pos=beamng_spec.CAM_POS,
        dir=beamng_spec.CAM_DIR,
        field_of_view_y=beamng_spec.CAM_FOV_Y,
        resolution=beamng_spec.CAM_RESOLUTION,
        is_render_colours=True,
        is_render_depth=False,
        is_render_annotations=False,
        is_visualised=visualise,
        is_static=False,
    )


def poll_camera(camera):
    """Poll the dashcam. Returns ``(flat_pixels, frame_2d)``.

    ``flat_pixels`` is the grayscale observation block in [0, 1]; ``frame_2d`` is
    the same data shaped ``CAM_OUT_SIZE`` for the human-play ASCII render. A
    missing camera yields a black frame rather than raising.
    """
    colour = camera.poll().get("colour", None) if camera is not None else None
    flat = process_camera_frame(colour, beamng_spec.CAM_OUT_SIZE)
    return flat, flat.reshape(beamng_spec.CAM_OUT_SIZE)


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
def create_roads_sensor(name: str, bng, vehicle):
    """Create a RoadsSensor for the ``road_info`` observation block.

    Callers must not poll it between a teleport and the next physics step: the
    sensor's game-engine side never answers on road-dense maps and the caller
    blocks forever in the socket recv (docs/romain.md, seventh issue). Both envs
    enforce that with a ``_road_pollable`` gate.
    """
    return RoadsSensor(name, bng, vehicle)


def remove_sensor(sensor) -> None:
    """Detach a sensor, bounded so a wedged sim cannot hang teardown.

    ``sensor.remove()`` round-trips to the simulator, which may never answer if it
    is already dying; a daemon thread with a join timeout means teardown always
    completes. Safe to call with None.
    """
    if sensor is None:
        return
    t = threading.Thread(target=sensor.remove, daemon=True)
    t.start()
    t.join(timeout=SENSOR_REMOVE_TIMEOUT)


def perception_block(
    *,
    sensor: str,
    lidar=None,
    camera=None,
    pos=(0.0, 0.0, 0.0),
    heading: float = 0.0,
    ego_extents=None,
):
    """Poll whichever sensor this config uses and return its feature block.

    Returns ``(block, debug, frame_2d)``: ``debug`` carries the LiDAR filtering
    counts (empty for camera) and ``frame_2d`` the camera frame (None for LiDAR).
    One call site for both branches, so the single, multi and race envs cannot
    drift in how they build the perception block.
    """
    if sensor == "camera":
        flat, frame = poll_camera(camera)
        return flat, {}, frame
    bins, debug = poll_lidar(lidar, pos, heading, ego_extents, sensor)
    return bins, debug, None


def block_summary(sensor: str, block) -> str:
    """One labeled log line for a perception block.

    LiDAR blocks print every cell (``h<n>`` for a single row, ``v<n>_h<n>`` for a
    grid); a camera block prints summary statistics instead of 256 pixels.
    """
    values = np.asarray(block).ravel()
    if sensor == "camera":
        if values.size == 0:
            return "obs cam   | (empty)"
        h, w = beamng_spec.CAM_OUT_SIZE
        return (
            f"obs cam   | min={values.min():+.2f} max={values.max():+.2f} "
            f"mean={values.mean():+.2f} px={values.size} ({h}x{w})"
        )

    h_bins = beamng_spec.LIDAR_RAYS
    ch = beamng_spec.LIDAR_CHANNELS_PER_RAY
    v_bins = int(beamng_spec.lidar_geometry(sensor)["v_bins"])
    parts = []
    for i in range(values.size):
        cell = i // ch
        v, h, c = cell // h_bins, cell % h_bins, i % ch
        label = f"h{h}" if v_bins == 1 else f"v{v}_h{h}"
        if ch > 1:
            label += f"c{c}"
        parts.append(f"{label}={values[i]:+.2f}")
    return f"obs lidar | {' '.join(parts)}"
