"""The RoadsSensor must not be polled before the sim has stepped past a teleport.

Measured (docs/romain.md, seventh issue): a poll with no intervening physics step
hangs the simulator's game-engine side forever on road-dense maps, and Python
blocks in the socket recv. The guard makes the invariant explicit rather than
relying on the order reset() happens to call things in.

The lockstep paths (reset/step) close the gate on a teleport/scenario load and
reopen it only once the sim has actually advanced. The realtime paths
(human_play and BeamNGRaceEnv's realtime advance) never call bng.step() at
all — the simulator runs on its own once resumed — so for those the gate must
be *open*, including across their in-session teleports (human-play respawns).
Closing it there would leave it dead for the rest of the session, since
nothing else would ever call _advance() to reopen it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from environments.beamng import BeamNGDrivingEnv


class _CountingRoads:
    def __init__(self):
        self.polls = 0

    def poll(self):
        self.polls += 1
        return {"halfWidth": 4.0, "dist2Left": 4.0, "dist2Right": 4.0}


def _env():
    env = BeamNGDrivingEnv(beamng_home="unused", road_info=True)
    env.roads_sensor = _CountingRoads()
    return env


class TestRoadPollGuard:
    def test_no_poll_before_the_first_step(self):
        env = _env()
        env._road_pollable = False
        out = env._road_info_features((0.0, 0.0, 0.0), 0.0)
        assert env.roads_sensor.polls == 0
        np.testing.assert_allclose(out, [0.0] * 6, atol=1e-6)

    def test_polls_once_the_sim_has_stepped(self):
        env = _env()
        env._road_pollable = True
        env._road_info_features((0.0, 0.0, 0.0), 0.0)
        assert env.roads_sensor.polls == 1

    def test_advance_opens_the_gate(self):
        env = _env()
        env._road_pollable = False

        class _Bng:
            def __init__(self):
                self.steps = 0

            def step(self, n):
                self.steps += n

        env.bng = _Bng()
        env._advance(5)
        assert env.bng.steps == 5
        assert env._road_pollable is True

    def test_a_fresh_env_starts_closed(self):
        assert BeamNGDrivingEnv(beamng_home="unused", road_info=True)._road_pollable is False


class TestRealtimeGateStaysOpen:
    """The trap the brief walked into: realtime paths never step(), so a gate
    that only opens in a step wrapper would read neutral zeros forever in
    human play — precisely where a human is supposed to verify the feature.
    """

    def test_human_play_resume_opens_the_gate(self, monkeypatch):
        env = BeamNGDrivingEnv(beamng_home="unused", road_info=True)
        env._road_pollable = False
        monkeypatch.setattr(env, "_load_scenario", lambda human_control=False: None)
        monkeypatch.setattr("environments.beamng.stop_requested", lambda: True)

        class _Bng:
            def __init__(self):
                self.resumed = False

            def resume(self):
                self.resumed = True

        env.bng = _Bng()

        env.human_play()

        assert env.bng.resumed is True
        assert env._road_pollable is True

    def test_human_play_respawn_leaves_the_gate_open(self):
        # _reset_human_episode is the shared teleport used by both the
        # crash-respawn and path-completion handlers inside human_play's loop.
        env = BeamNGDrivingEnv(beamng_home="unused", road_info=True)
        env.bng = None  # keeps _update_active_marker a no-op
        env.vehicle = MagicMock()
        env.trajectory = SimpleNamespace(
            spawn_pos=(0.0, 0.0, 0.0), spawn_rot=(0.0, 0.0, 0.0, 1.0)
        )
        env._road_pollable = True  # the session is resumed and running

        env._reset_human_episode()

        assert env._road_pollable is True
