"""Tests for core.quickrace — the game's own race tracks.

Fixtures are written as real level archives in tmp_path, in both shipped formats,
because the parsing IS the feature: the legacy format needs a Torque prefab read
alongside its json, and the current format needs the right start position picked
out of several. Nothing here touches a running simulator, which is the point of
the module.
"""

import json
import math
import zipfile
from pathlib import Path

import pytest

from core import quickrace
from core.trajectory import _quat_to_forward

# --------------------------------------------------------------------------- #
# Fixtures: minimal but real-shaped level content
# --------------------------------------------------------------------------- #
LEGACY_JSON = {
    "name": "quickrace.testmap.circuit.title",
    "closed": True,
    "reversible": True,
    "lapConfig": ["quickrace_wp1", "quickrace_wp2"],
    "finishLineCheckpoint": "quickrace_finish",
}

LEGACY_SPRINT_JSON = {
    "name": "quickrace.testmap.sprint.title",
    "closed": False,
    "reversible": False,
    "lapConfig": ["quickrace_wp1", "quickrace_wp2"],
    "finishLineCheckpoint": "quickrace_finish",
}


def _waypoint(name, pos, scale=6.8):
    return f"""
   new BeamNGWaypoint({name}) {{
      drawDebug = "0";
      position = "{pos[0]} {pos[1]} {pos[2]}";
      scale = "{scale} {scale} {scale}";
      rotationMatrix = "1 0 0 0 1 0 0 0 1";
      mode = "Ignore";
   }};"""


def _spawn(name, pos):
    return f"""
   new SpawnSphere({name}) {{
      SpawnClass = "player";
      radius = "1";
      position = "{pos[0]} {pos[1]} {pos[2]}";
      scale = "1 1 1";
      rotationMatrix = "0.1 0.98 0 -0.98 0.1 0 0 0 1";
      mode = "Ignore";
   }};"""


def _prefab(key, *, spawn=(0.0, 0.0, 10.0)):
    """A prefab shaped like the shipped ones: spawn plus three waypoints."""
    body = "".join(
        [
            _waypoint("quickrace_finish", (0.0, 5.0, 10.0), scale=5.0),
            _waypoint("quickrace_wp2", (100.0, 100.0, 12.0)),
            _waypoint("quickrace_wp1", (100.0, 0.0, 11.0)),
            _spawn(f"{key}_standing_spawn", spawn),
            _spawn(f"{key}_standingReverse_spawn", (0.0, -5.0, 10.0)),
        ]
    )
    return f"""//--- OBJECT WRITE BEGIN ---
$ThisPrefab = new SimGroup() {{
   canSave = "1";{body}
}};
//--- OBJECT WRITE END ---
"""


RACE_JSON = {
    "name": "Test Ring",
    "classification": {"closed": True, "reversible": True},
    "defaultStartPosition": 24,
    "startPositions": [
        # Deliberately not first: the loader must pick by oldId, not by position.
        {"name": "Start Position 25", "oldId": 25, "pos": [0.0, 0.0, 5.0], "rot": [0, 0, 1, 0]},
        {"name": "Start Position 24", "oldId": 24, "pos": [10.0, 0.0, 5.0], "rot": [0, 0, 0, 1]},
    ],
    "startNode": 2,
    "endNode": -1,
    "pathnodes": [
        {"name": "Pathnode 2", "oldId": 2, "pos": [10.0, 0.0, 5.0], "radius": 16},
        {"name": "Pathnode 3", "oldId": 3, "pos": [200.0, 0.0, 5.0], "radius": 16},
        {"name": "Pathnode 4", "oldId": 4, "pos": [200.0, 200.0, 5.0], "radius": 16},
    ],
}


def _write_level(root: Path, map_name: str, files: dict[str, str]) -> Path:
    """Write a level archive shaped like content/levels/<map>.zip."""
    levels = root / "content" / "levels"
    levels.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(levels / f"{map_name}.zip", "w") as z:
        for name, text in files.items():
            z.writestr(f"levels/{map_name}/quickrace/{name}", text)
    return root


@pytest.fixture
def home(tmp_path):
    """A BeamNG install holding one map with a legacy circuit, sprint and a ring."""
    return _write_level(
        tmp_path,
        "testmap",
        {
            "circuit.json": json.dumps(LEGACY_JSON),
            "circuit.prefab": _prefab("circuit"),
            "sprint1.json": json.dumps(LEGACY_SPRINT_JSON),
            "sprint1.prefab": _prefab("sprint1"),
            "ring.race.json": json.dumps(RACE_JSON),
        },
    )


# --------------------------------------------------------------------------- #
class TestAvailable:
    def test_lists_both_formats(self, home):
        assert quickrace.available("testmap", home) == ["circuit", "ring", "sprint1"]

    def test_unknown_map_lists_nothing(self, home):
        assert quickrace.available("nosuchmap", home) == []

    def test_missing_install_lists_nothing(self, tmp_path):
        assert quickrace.available("testmap", tmp_path / "nope") == []

    def test_map_name_matches_case_insensitively(self, home):
        # Shipped archives are lowercase; a map name's case must not decide this.
        assert quickrace.available("TestMap", home) == ["circuit", "ring", "sprint1"]

    def test_a_json_without_its_prefab_is_not_offered(self, tmp_path):
        # The positions live in the prefab, so json alone is not a usable track.
        home = _write_level(tmp_path, "m", {"orphan.json": json.dumps(LEGACY_JSON)})
        assert quickrace.available("m", home) == []


class TestLegacyFormat:
    def test_reads_closed_flag_and_ordered_checkpoints(self, home):
        race = quickrace.load("testmap", "circuit", home)
        assert race.closed is True
        assert race.kind == "lap"
        # lapConfig order, then the finish line appended to close the loop —
        # NOT the order the objects happen to appear in the prefab.
        assert race.checkpoints == [
            (100.0, 0.0, 11.0),
            (100.0, 100.0, 12.0),
            (0.0, 5.0, 10.0),
        ]

    def test_reads_authored_radii(self, home):
        race = quickrace.load("testmap", "circuit", home)
        assert race.radii == [6.8, 6.8, 5.0]

    def test_uses_the_standing_forward_spawn(self, home):
        # Not the reverse spawn, which would run the track backwards.
        assert quickrace.load("testmap", "circuit", home).spawn_pos == (0.0, 0.0, 10.0)

    def test_sprint_is_not_closed(self, home):
        race = quickrace.load("testmap", "sprint1", home)
        assert race.closed is False
        assert race.kind == "sprint"

    def test_trailing_commas_and_comments_are_tolerated(self, tmp_path):
        # The shipped files are not valid JSON: they carry trailing commas and
        # sometimes // comments.
        raw = """{
          // a comment the game's parser allows
          "closed": false,
          "lapConfig": [
            "quickrace_wp1",
            "quickrace_wp2",
          ],
        }"""
        home = _write_level(tmp_path, "m", {"t.json": raw, "t.prefab": _prefab("t")})
        assert quickrace.load("m", "t", home).closed is False


class TestRaceJsonFormat:
    def test_reads_classification_and_pathnodes(self, home):
        race = quickrace.load("testmap", "ring", home)
        assert race.closed is True
        assert race.title == "Test Ring"
        # Closed, so the first node is appended to shut the loop.
        assert race.checkpoints[-1] == race.checkpoints[0]
        assert len(race.checkpoints) == 4

    def test_picks_the_start_position_named_by_default_start(self, home):
        # oldId 24 sits second in the list; picking by order would take 25.
        assert quickrace.load("testmap", "ring", home).spawn_pos == (10.0, 0.0, 5.0)

    def test_rotates_the_chain_to_begin_at_start_node(self, tmp_path):
        spec = json.loads(json.dumps(RACE_JSON))
        spec["startNode"] = 3  # start mid-chain
        home = _write_level(tmp_path, "m", {"r.race.json": json.dumps(spec)})
        race = quickrace.load("m", "r", home)
        assert race.checkpoints[0] == (200.0, 0.0, 5.0)


class TestLoadAll:
    def test_longest_track_first(self, home):
        races = quickrace.load_all("testmap", home)
        lengths = [r.length_m() for r in races]
        assert lengths == sorted(lengths, reverse=True)

    def test_every_track_is_returned(self, home):
        assert {r.key for r in quickrace.load_all("testmap", home)} == {
            "circuit",
            "ring",
            "sprint1",
        }


class TestLoadErrors:
    def test_unknown_key_names_what_is_available(self, home):
        with pytest.raises(ValueError, match="circuit, ring, sprint1"):
            quickrace.load("testmap", "nope", home)

    def test_missing_map_is_a_value_error(self, home):
        with pytest.raises(ValueError):
            quickrace.load("nosuchmap", "circuit", home)


class TestToTrajectory:
    def test_checkpoints_become_the_sparse_waypoints(self, home):
        traj = quickrace.to_trajectory(quickrace.load("testmap", "circuit", home))
        assert traj.sparse_waypoints == [
            (100.0, 0.0, 11.0),
            (100.0, 100.0, 12.0),
            (0.0, 5.0, 10.0),
        ]

    def test_dense_waypoints_are_the_same_line_resampled(self, home):
        traj = quickrace.to_trajectory(quickrace.load("testmap", "circuit", home))
        assert len(traj.dense_waypoints) > len(traj.sparse_waypoints)

    def test_spawn_faces_its_first_checkpoint(self, home):
        # The stored rotation is not trusted; the heading is derived from the line
        # so it obeys the convention measured in-sim for this project.
        traj = quickrace.to_trajectory(quickrace.load("testmap", "circuit", home))
        fx, fy = _quat_to_forward(traj.spawn_rot)
        cp = traj.sparse_waypoints[0]
        dx, dy = cp[0] - traj.spawn_pos[0], cp[1] - traj.spawn_pos[1]
        norm = math.hypot(dx, dy)
        assert (fx * dx + fy * dy) / norm == pytest.approx(1.0, abs=1e-6)

    def test_source_records_the_track_and_its_kind(self, home):
        traj = quickrace.to_trajectory(quickrace.load("testmap", "circuit", home))
        assert traj.source == "quickrace:circuit:lap"
        sprint = quickrace.to_trajectory(quickrace.load("testmap", "sprint1", home))
        assert sprint.source == "quickrace:sprint1:sprint"

    def test_checkpoints_on_top_of_the_spawn_are_dropped(self, tmp_path):
        # A circuit's start line and its first checkpoint are the same place; kept,
        # it would score as cleared before the car moved.
        spec = json.loads(json.dumps(RACE_JSON))
        home = _write_level(tmp_path, "m", {"r.race.json": json.dumps(spec)})
        race = quickrace.load("m", "r", home)
        traj = quickrace.to_trajectory(race)
        # Node 2 IS the start position, so it must not be the first waypoint.
        assert traj.sparse_waypoints[0] != race.spawn_pos
        assert traj.sparse_waypoints[0] == (200.0, 0.0, 5.0)
