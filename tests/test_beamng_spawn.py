"""Tests for environments.beamng_spawn — the fine spawn height.

SPAWN_Z_OFFSET_M puts a cached spawn on the road surface; this module measures the
remaining centimetres, the car's ride height, so a teleport (which has no cling and
places the reference point exactly) lands the car at rest instead of dropping it
onto race suspension. The numbers below are that scale — tens of centimetres.

Failure modes matter as much as the happy path: the contract is "return 0.0 and
keep the cached height", because a wrong shift applied to every spawn on the map
would be worse than no shift at all.
"""

from unittest.mock import MagicMock

import pytest

from environments.beamng_spawn import (
    MAX_CORRECTION_M,
    SETTLE_STEPS,
    SETTLE_TRIES,
    corrected_spawn,
    measure_spawn_z_correction,
)


class _FakeVehicle:
    """Replays a scripted sequence of vehicle states, one per poll.

    The last state repeats once the script runs out, so a test only has to script
    the states it cares about.
    """

    def __init__(self, states, raise_on_poll=None):
        self._states = list(states)
        self._raise = raise_on_poll
        self.state = None
        self.polls = 0

    def poll_sensors(self):
        self.polls += 1
        if self._raise is not None:
            raise self._raise
        self.state = self._states[min(self.polls - 1, len(self._states) - 1)]


def _resting(z):
    return {"pos": (10.0, 20.0, z), "vel": (0.0, 0.0, 0.0)}


def _falling(z, vz=-3.0):
    return {"pos": (10.0, 20.0, z), "vel": (0.0, 0.0, vz)}


class TestMeasureSpawnZCorrection:
    def test_resting_car_yields_the_delta_to_the_cached_height(self):
        # East coast's real road surface, 51.92, is now the cached spawn; the car
        # rests a ride height above it.
        vehicle = _FakeVehicle([_resting(52.28)])
        corr = measure_spawn_z_correction(MagicMock(), vehicle, 51.92)
        assert corr == pytest.approx(0.36)

    def test_correction_lands_a_teleport_where_the_car_rests(self):
        vehicle = _FakeVehicle([_resting(52.28)])
        corr = measure_spawn_z_correction(MagicMock(), vehicle, 51.92)
        assert corrected_spawn((700.0, -6.7, 51.92), corr)[2] == pytest.approx(52.28)

    def test_steps_until_the_car_stops_moving(self):
        # Polled mid-bounce twice, so the height must not be read yet: a moving
        # car's z is not the height it will come to rest at.
        bng = MagicMock()
        vehicle = _FakeVehicle([_falling(52.60), _falling(52.35), _resting(52.28)])
        corr = measure_spawn_z_correction(bng, vehicle, 51.92)
        assert corr == pytest.approx(0.36)
        assert bng.step.call_count == 2
        bng.step.assert_called_with(SETTLE_STEPS)

    def test_missing_velocity_is_read_as_resting(self):
        # Older beamngpy state payloads carry no "vel"; refusing to measure would
        # leave the drop in place forever, so an absent velocity is trusted.
        vehicle = _FakeVehicle([{"pos": (0.0, 0.0, 10.0)}])
        assert measure_spawn_z_correction(MagicMock(), vehicle, 10.5) == pytest.approx(-0.5)

    def test_car_that_never_settles_keeps_the_cached_height(self):
        bng = MagicMock()
        vehicle = _FakeVehicle([_falling(50.0)])
        assert measure_spawn_z_correction(bng, vehicle, 52.93) == 0.0
        assert bng.step.call_count == SETTLE_TRIES

    def test_implausible_correction_is_rejected(self):
        # The car is falling through the map: shifting every spawn by this would
        # put them underground.
        vehicle = _FakeVehicle([_resting(20.0)])
        assert measure_spawn_z_correction(MagicMock(), vehicle, 52.93) == 0.0

    def test_a_metre_scale_correction_is_rejected(self):
        # Guards the global lowering: with the coarse offset at 0 a real correction
        # is a ride height, so a metre-scale reading is a bad measurement and must
        # not be allowed to quietly put the spawn back up where it was.
        vehicle = _FakeVehicle([_resting(52.93)])
        assert measure_spawn_z_correction(MagicMock(), vehicle, 51.92) == 0.0

    def test_correction_at_the_limit_is_still_accepted(self):
        vehicle = _FakeVehicle([_resting(50.0)])
        corr = measure_spawn_z_correction(MagicMock(), vehicle, 50.0 + MAX_CORRECTION_M)
        assert corr == pytest.approx(-MAX_CORRECTION_M)

    def test_failed_poll_keeps_the_cached_height(self):
        vehicle = _FakeVehicle([], raise_on_poll=RuntimeError("not connected"))
        assert measure_spawn_z_correction(MagicMock(), vehicle, 52.93) == 0.0

    def test_failed_step_keeps_the_cached_height(self):
        bng = MagicMock()
        bng.step.side_effect = RuntimeError("sim busy")
        vehicle = _FakeVehicle([_falling(52.5)])
        assert measure_spawn_z_correction(bng, vehicle, 52.93) == 0.0

    def test_state_without_a_position_keeps_the_cached_height(self):
        vehicle = _FakeVehicle([{"vel": (0.0, 0.0, 0.0)}])
        assert measure_spawn_z_correction(MagicMock(), vehicle, 52.93) == 0.0

    def test_non_numeric_position_keeps_the_cached_height(self):
        vehicle = _FakeVehicle([{"pos": ("nope", "nope", "nope")}])
        assert measure_spawn_z_correction(MagicMock(), vehicle, 52.93) == 0.0

    def test_nan_position_keeps_the_cached_height(self):
        vehicle = _FakeVehicle([{"pos": (0.0, 0.0, float("nan"))}])
        assert measure_spawn_z_correction(MagicMock(), vehicle, 52.93) == 0.0

    def test_unset_state_keeps_the_cached_height(self):
        # A vehicle polled before the scenario is running has state None.
        vehicle = _FakeVehicle([None])
        assert measure_spawn_z_correction(MagicMock(), vehicle, 52.93) == 0.0


class TestCorrectedSpawn:
    def test_shifts_only_the_height(self):
        assert corrected_spawn((1.0, 2.0, 3.0), -0.6) == (1.0, 2.0, pytest.approx(2.4))

    def test_zero_correction_is_the_cached_spawn(self):
        assert corrected_spawn((1.0, 2.0, 3.0), 0.0) == (1.0, 2.0, 3.0)

    def test_accepts_a_list_spawn_from_json(self):
        # Caches deserialize to lists; teleport needs a tuple of floats.
        assert corrected_spawn([1, 2, 3], 0.5) == (1.0, 2.0, 3.5)
