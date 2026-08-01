"""Unit tests for environments.beamng_spec — the sensor/output axes and sizes."""

import pytest

from environments.beamng_spec import (
    CAM_OUT_SIZE,
    CONTINUOUS_ALGOS,
    FIXED_ALGOS,
    LIDAR_RAYS,
    PERCEPTION_FEATURES,
    RACE_CAR,
    SENSORS,
    action_size,
    is_lidar,
    lidar_geometry,
    obs_size,
    output_for_algo,
    perception_features,
)


class TestOutputForAlgo:
    @pytest.mark.parametrize("algo", FIXED_ALGOS)
    def test_discrete_algos_are_fixed(self, algo):
        assert output_for_algo(algo) == "fixed"

    @pytest.mark.parametrize("algo", CONTINUOUS_ALGOS)
    def test_continuous_algos_are_continuous(self, algo):
        assert output_for_algo(algo) == "continuous"

    def test_unknown_algo_raises(self):
        # Guessing here would silently build the wrong action head.
        with pytest.raises(ValueError, match="unknown algorithm"):
            output_for_algo("q_learning")

    def test_every_algo_is_classified_exactly_once(self):
        assert not set(FIXED_ALGOS) & set(CONTINUOUS_ALGOS)


class TestObsSize:
    def test_reproduces_historical_layouts(self):
        """The observation contract must not move: these are the pre-refactor
        lengths of beamng/beamng_continuous (14), beamng_lidar (38) and
        beamng_camera (262), so existing obs slicing keeps working."""
        assert obs_size("lidar") == 14
        assert obs_size("adv_lidar") == 38
        assert obs_size("camera") == 262

    def test_hints_add_two_dims_each(self):
        assert obs_size("lidar", trajectory_hints=3) == 14 + 6

    def test_body_orientation_adds_two(self):
        assert obs_size("lidar", body_orientation=True) == 16

    def test_wheel_terrain_adds_two(self):
        assert obs_size("lidar", wheel_terrain=True) == 16

    def test_options_stack(self):
        assert (
            obs_size("adv_lidar", trajectory_hints=2, body_orientation=True, wheel_terrain=True)
            == 38 + 4 + 2 + 2
        )

    def test_flags_off_matches_bare_call(self):
        for sensor in SENSORS:
            assert (
                obs_size(sensor, 0, False, False) == obs_size(sensor)
            ), f"{sensor} flag-off length drifted"

    def test_unknown_sensor_raises(self):
        with pytest.raises(ValueError, match="unknown sensor"):
            obs_size("radar")


class TestPerceptionFeatures:
    def test_every_sensor_has_a_feature_count(self):
        assert set(SENSORS) == set(PERCEPTION_FEATURES)

    def test_lidar_is_one_row_of_rays(self):
        assert perception_features("lidar") == LIDAR_RAYS

    def test_adv_lidar_is_v_bins_times_rays(self):
        geom = lidar_geometry("adv_lidar")
        assert perception_features("adv_lidar") == int(geom["v_bins"]) * LIDAR_RAYS

    def test_camera_is_the_flattened_output_frame(self):
        assert perception_features("camera") == CAM_OUT_SIZE[0] * CAM_OUT_SIZE[1]


class TestActionSize:
    def test_fixed_matches_the_action_table(self):
        from environments.beamng import BeamNGDrivingEnv

        assert action_size("fixed") == len(BeamNGDrivingEnv.ACTIONS)

    def test_continuous_is_throttle_steering_brake(self):
        assert action_size("continuous") == 3

    def test_unknown_output_raises(self):
        with pytest.raises(ValueError, match="unknown output"):
            action_size("discrete")


class TestLidarGeometry:
    @pytest.mark.parametrize("sensor", ["lidar", "adv_lidar"])
    def test_lidar_sensors_have_geometry(self, sensor):
        geom = lidar_geometry(sensor)
        assert {"v_bins", "vert_res", "vert_angle"} <= set(geom)
        assert is_lidar(sensor)

    def test_camera_has_no_lidar_geometry(self):
        assert not is_lidar("camera")
        with pytest.raises(ValueError, match="not a LiDAR sensor"):
            lidar_geometry("camera")


class TestRaceCar:
    def test_carries_the_beamngpy_vehicle_kwargs(self):
        # The multi/race envs build Vehicle(name, **RACE_CAR | {"color": ...}).
        assert {"model", "licence", "color", "part_config"} == set(RACE_CAR)

    def test_is_the_sunburst2_race_config(self):
        assert RACE_CAR["model"] == "sunburst2"
        assert RACE_CAR["part_config"] == "vehicles/sunburst2/trackday_M.pc"
