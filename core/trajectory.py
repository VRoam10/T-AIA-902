"""Automatic per-map trajectory generation for BeamNG environments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class TrajectoryData:
    """A spawn pose plus two pre-sampled waypoint sequences for one map."""

    spawn_pos: Vec3
    spawn_rot: Quat
    sparse_waypoints: list[Vec3]
    dense_waypoints: list[Vec3]
    map_name: str
    generated_at: str
    source: str

    def to_json(self) -> str:
        d = asdict(self)
        d["spawn_pos"] = list(self.spawn_pos)
        d["spawn_rot"] = list(self.spawn_rot)
        d["sparse_waypoints"] = [list(p) for p in self.sparse_waypoints]
        d["dense_waypoints"] = [list(p) for p in self.dense_waypoints]
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, payload: str) -> "TrajectoryData":
        d = json.loads(payload)
        return cls(
            spawn_pos=tuple(d["spawn_pos"]),
            spawn_rot=tuple(d["spawn_rot"]),
            sparse_waypoints=[tuple(p) for p in d["sparse_waypoints"]],
            dense_waypoints=[tuple(p) for p in d["dense_waypoints"]],
            map_name=d["map_name"],
            generated_at=d["generated_at"],
            source=d["source"],
        )
