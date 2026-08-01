"""The game's own race tracks, read straight from the level files.

BeamNG ships hand-authored quickrace (Time Trial) tracks per level: an ordered
chain of checkpoints, a standing-start position, and a ``closed`` flag saying
whether the track is a **sprint** (point to point) or a **lap** (returns to its
start). These are proper racing lines drawn by the level authors, so they beat
anything :mod:`core.trajectory` can infer from the road network — and they are the
only source of a genuinely closed circuit.

They are read from disk rather than from a running simulator: levels are zip
archives under ``<BEAMNG_HOME>/content/levels``, so listing or loading a track
costs no BeamNG launch. Generated trajectories are untouched by this module; the
two are alternative sources of the same :class:`~core.trajectory.TrajectoryData`
shape, and callers choose.

Two on-disk formats coexist, both supported:

  * **legacy** (italy, west_coast_usa, east_coast_usa) — ``<key>.json`` holds
    ``closed`` and ``lapConfig`` (the checkpoint names, in order); the sibling
    ``<key>.prefab`` holds their positions as ``BeamNGWaypoint`` objects plus a
    ``<key>_standing_spawn`` ``SpawnSphere``.
  * **current** (gridmap_v2) — ``<key>.race.json``, self-contained, with
    ``classification.closed``, ``pathnodes`` and ``startPositions``.

Neither format's stored rotation is trusted for the spawn: headings here are
derived from the checkpoint chain with :func:`core.trajectory.heading_to_quat`,
the one convention this project has actually measured in-sim.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.trajectory import (
    DENSE_SPACING_M,
    SPAWN_CLEARANCE_M,
    TrajectoryData,
    heading_to_quat,
    resample,
)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# Where the level archives live under a BeamNG install.
LEVELS_SUBDIR = Path("content") / "levels"
QUICKRACE_DIR = "quickrace"

# The standing (non-rolling) forward start is the one we want: a rolling start
# would need the car already at speed, and the reverse variants run the track
# backwards.
STANDING_SPAWN_SUFFIX = "_standing_spawn"


@dataclass(frozen=True)
class QuickRace:
    """One of the game's race tracks, in the terms this project cares about."""

    key: str  # file stem, e.g. "mixedCircuit1" — what a user selects
    map_name: str
    closed: bool  # True = lap/circuit, False = sprint (point to point)
    reversible: bool
    spawn_pos: Vec3
    checkpoints: list[Vec3]  # in racing order, starting after the spawn
    radii: list[float]  # per-checkpoint trigger radius, as authored
    title: str = ""  # human name when the file carries one (or a locale key)

    @property
    def kind(self) -> str:
        return "lap" if self.closed else "sprint"

    def length_m(self) -> float:
        """Total checkpoint-to-checkpoint distance in the XY plane."""
        pts = [self.spawn_pos, *self.checkpoints]
        return sum(
            math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            for i in range(len(pts) - 1)
        )


# --------------------------------------------------------------------------- #
# Level archive access
# --------------------------------------------------------------------------- #
def _level_archive(map_name: str, beamng_home: str | Path) -> Path | None:
    """Path to ``<map_name>.zip``, matched case-insensitively, or None."""
    levels = Path(beamng_home) / LEVELS_SUBDIR
    if not levels.is_dir():
        return None
    target = f"{map_name.lower()}.zip"
    for entry in levels.glob("*.zip"):
        if entry.name.lower() == target:
            return entry
    return None


def _read_quickrace_files(map_name: str, beamng_home: str | Path) -> dict[str, bytes]:
    """Every file under the level's ``quickrace/`` folder, keyed by base name.

    Reads the packed level (a zip) and, if the level is also present unpacked
    (some installs and mods ship loose folders), the directory — the directory
    wins, matching how the game overlays them.
    """
    out: dict[str, bytes] = {}

    archive = _level_archive(map_name, beamng_home)
    if archive is not None:
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                parts = name.split("/")
                if len(parts) >= 2 and parts[-2] == QUICKRACE_DIR and parts[-1]:
                    out[parts[-1]] = z.read(name)

    loose = Path(beamng_home) / LEVELS_SUBDIR.parent / "levels" / map_name / QUICKRACE_DIR
    if loose.is_dir():
        for entry in loose.iterdir():
            if entry.is_file():
                out[entry.name] = entry.read_bytes()

    return out


def _load_lenient_json(payload: bytes) -> dict:
    """Parse BeamNG's JSON, which is not strictly valid.

    The shipped files carry trailing commas (``"quickrace_wp11",]``) and
    occasional ``//`` comments, both of which json.loads rejects.
    """
    text = payload.decode("utf-8", errors="replace")
    text = re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Prefab parsing (legacy format)
# --------------------------------------------------------------------------- #
_OBJECT_RE = re.compile(
    r"new\s+(?P<cls>\w+)\s*\(\s*(?P<name>[\w.\-]+)\s*\)\s*\{(?P<body>.*?)\n\s*\};",
    re.DOTALL,
)


def _field(body: str, key: str) -> str | None:
    m = re.search(rf'{key}\s*=\s*"([^"]*)"', body)
    return m.group(1) if m else None


def _floats(raw: str | None) -> list[float]:
    if not raw:
        return []
    out = []
    for token in raw.split():
        try:
            out.append(float(token))
        except ValueError:
            return []
    return out


def parse_prefab_objects(text: str) -> dict[str, dict]:
    """Objects in a ``.prefab``, as ``{name: {"class", "pos", "scale"}}``.

    Torque prefabs are a small declarative dialect, not JSON. Only the fields
    this module needs are pulled out; anything unparseable is skipped rather than
    raising, because one malformed object should not cost the whole track.
    """
    objects: dict[str, dict] = {}
    for m in _OBJECT_RE.finditer(text):
        body = m.group("body")
        pos = _floats(_field(body, "position"))
        if len(pos) != 3:
            continue
        scale = _floats(_field(body, "scale"))
        objects[m.group("name")] = {
            "class": m.group("cls"),
            "pos": (pos[0], pos[1], pos[2]),
            "scale": scale[0] if scale else 0.0,
        }
    return objects


def _legacy_race(key: str, files: dict[str, bytes], map_name: str) -> QuickRace | None:
    """Build a QuickRace from ``<key>.json`` + ``<key>.prefab``."""
    prefab = files.get(f"{key}.prefab")
    if prefab is None:
        return None
    spec = _load_lenient_json(files[f"{key}.json"])
    objects = parse_prefab_objects(prefab.decode("utf-8", errors="replace"))

    names = list(spec.get("lapConfig") or [])
    finish = spec.get("finishLineCheckpoint")
    # A sprint finishes somewhere else than it started, and that finish line is
    # not in lapConfig — it is the last checkpoint to clear. A closed track's
    # finish IS its start line, so appending it closes the loop.
    if finish and finish not in names:
        names.append(finish)

    checkpoints = [objects[n]["pos"] for n in names if n in objects]
    radii = [objects[n]["scale"] for n in names if n in objects]
    # One checkpoint is enough — a drag strip is spawn plus a finish line.
    if not checkpoints:
        return None

    spawn = objects.get(f"{key}{STANDING_SPAWN_SUFFIX}")
    if spawn is None:
        return None

    return QuickRace(
        key=key,
        map_name=map_name,
        closed=bool(spec.get("closed", False)),
        reversible=bool(spec.get("reversible", False)),
        spawn_pos=spawn["pos"],
        checkpoints=checkpoints,
        radii=radii,
        title=str(spec.get("name", "")),
    )


# --------------------------------------------------------------------------- #
# Self-contained format (.race.json)
# --------------------------------------------------------------------------- #
def _race_json(key: str, files: dict[str, bytes], map_name: str) -> QuickRace | None:
    """Build a QuickRace from a self-contained ``<key>.race.json``."""
    spec = _load_lenient_json(files[f"{key}.race.json"])
    classification = spec.get("classification") or {}

    nodes = [n for n in (spec.get("pathnodes") or []) if len(n.get("pos") or ()) == 3]
    if not nodes:
        return None
    # File order is racing order, but it is only guaranteed to *begin* at
    # `startNode`; rotate so it does rather than trusting that it already does —
    # starting mid-chain would drive the track from the wrong place.
    start = spec.get("startNode")
    first = next((i for i, n in enumerate(nodes) if n.get("oldId") == start), 0)
    nodes = nodes[first:] + nodes[:first]
    checkpoints = [tuple(float(v) for v in n["pos"]) for n in nodes]
    radii = [float(n.get("radius", 0.0)) for n in nodes]

    starts = spec.get("startPositions") or []
    if not starts:
        return None
    # `defaultStartPosition` names a start by its `oldId`; the others are the
    # reverse and rolling variants, which we do not want.
    wanted = spec.get("defaultStartPosition")
    chosen = next((s for s in starts if s.get("oldId") == wanted), starts[0])
    spawn = chosen.get("pos")
    if len(spawn or ()) != 3:
        return None

    closed = bool(classification.get("closed", False))
    if closed:
        checkpoints.append(checkpoints[0])
        radii.append(radii[0])

    return QuickRace(
        key=key,
        map_name=map_name,
        closed=closed,
        reversible=bool(classification.get("reversible", False)),
        spawn_pos=tuple(float(v) for v in spawn),
        checkpoints=checkpoints,
        radii=radii,
        title=str(spec.get("name", "")),
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def available(map_name: str, beamng_home: str | Path) -> list[str]:
    """Sorted track keys for a map. Empty when the level ships none."""
    files = _read_quickrace_files(map_name, beamng_home)
    keys = {
        name[: -len(".race.json")] for name in files if name.endswith(".race.json")
    } | {
        name[: -len(".json")]
        for name in files
        if name.endswith(".json")
        and not name.endswith(".race.json")
        and not name.endswith(".prefab.json")
        and f"{name[:-len('.json')]}.prefab" in files
    }
    return sorted(keys)


def load(map_name: str, key: str, beamng_home: str | Path) -> QuickRace:
    """Load one track by key. Raises ValueError if it is missing or unusable."""
    files = _read_quickrace_files(map_name, beamng_home)
    race = _build(key, files, map_name)
    if race is None:
        raise ValueError(
            f"'{key}' is not a usable race track on '{map_name}'. "
            f"Available: {', '.join(available(map_name, beamng_home)) or 'none'}"
        )
    return race


def load_all(map_name: str, beamng_home: str | Path) -> list[QuickRace]:
    """Every usable track on a map, longest first. Unusable ones are skipped."""
    files = _read_quickrace_files(map_name, beamng_home)
    races = []
    for key in available(map_name, beamng_home):
        race = _build(key, files, map_name)
        if race is not None:
            races.append(race)
    return sorted(races, key=lambda r: r.length_m(), reverse=True)


def _build(key: str, files: dict[str, bytes], map_name: str) -> QuickRace | None:
    if f"{key}.race.json" in files:
        return _race_json(key, files, map_name)
    if f"{key}.json" in files:
        return _legacy_race(key, files, map_name)
    return None


def to_trajectory(race: QuickRace) -> TrajectoryData:
    """Convert a track into the TrajectoryData every env already consumes.

    The authored checkpoints become the sparse waypoints; the dense set is the
    same line resampled at the generator's spacing, so a dense warm-up curriculum
    works on a game track exactly as it does on a generated one.

    The spawn rotation is derived from the line ahead rather than from the file's
    stored rotation, so it obeys the same measured convention as generated paths.
    Leading checkpoints sitting on top of the spawn are dropped: on a circuit the
    start line and the first checkpoint are the same place, and a checkpoint that
    close would score as cleared the instant the episode began.
    """
    ahead = _checkpoints_ahead(race.spawn_pos, race.checkpoints)
    line = [race.spawn_pos, *ahead]
    target = _first_distinct(race.spawn_pos, ahead)
    if target is None:
        raise ValueError(
            f"track '{race.key}' on '{race.map_name}' has every checkpoint on top of "
            "its start position, so no spawn heading can be derived"
        )
    return TrajectoryData(
        spawn_pos=race.spawn_pos,
        spawn_rot=heading_to_quat(race.spawn_pos, target),
        sparse_waypoints=ahead,
        dense_waypoints=resample(line, DENSE_SPACING_M)[1:],
        map_name=race.map_name,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=f"quickrace:{race.key}:{race.kind}",
    )


def _first_distinct(spawn: Vec3, checkpoints: list[Vec3]) -> Vec3 | None:
    """First checkpoint that differs from the spawn in the XY plane, or None."""
    for cp in checkpoints:
        if abs(cp[0] - spawn[0]) > 1e-6 or abs(cp[1] - spawn[1]) > 1e-6:
            return cp
    return None


def _checkpoints_ahead(spawn: Vec3, checkpoints: list[Vec3]) -> list[Vec3]:
    """Checkpoints from the first one clear of the spawn onward.

    Always keeps at least the last checkpoint, so a track whose every node sits
    within the clearance still yields something to drive at.
    """
    for i, cp in enumerate(checkpoints):
        if math.hypot(cp[0] - spawn[0], cp[1] - spawn[1]) >= SPAWN_CLEARANCE_M:
            return checkpoints[i:]
    return checkpoints[-1:]
