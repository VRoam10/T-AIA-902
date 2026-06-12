"""Tests for the multi-agent CLI builder helper."""

from unittest.mock import MagicMock, patch

from core.cli import build_multi_session


def test_build_multi_session_builds_one_agent_per_spec():
    specs = [
        {
            "algo": "dqn",
            "env": "beamng",
            "vehicle_id": "taxi",
            "color": "Yellow",
            "save_path": "outputs/multi-agents/dqn.pth",
        },
        {
            "algo": "ddpg",
            "env": "beamng_continuous",
            "vehicle_id": "taxi",
            "color": "Red",
            "save_path": "outputs/multi-agents/ddpg.pth",
        },
    ]
    # Patch the env so no BeamNG launch happens.
    with patch("core.cli.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        env, slots = build_multi_session(specs, map_name="gridmap_v2")

    assert len(slots) == 2
    assert slots[0].action_space == "discrete"
    assert slots[1].action_space == "continuous"
    # every slot has a concrete agent instance attached
    assert all(s.agent is not None for s in slots)


def test_build_multi_session_passes_map_to_env():
    specs = [
        {
            "algo": "dqn",
            "env": "beamng",
            "vehicle_id": "taxi",
            "color": "Yellow",
            "save_path": "outputs/multi-agents/dqn.pth",
        },
    ]
    with patch("core.cli.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        build_multi_session(specs, map_name="italy")
        _, kwargs = EnvCls.call_args
        assert kwargs["map_name"] == "italy"


def test_build_multi_session_sizes_agent_to_each_env():
    # A camera env (262 states) and a lidar env (14 states) -> agents built with
    # the matching observation size for each.
    specs = [
        {
            "algo": "dqn",
            "env": "beamng",
            "vehicle_id": "taxi",
            "color": "Yellow",
            "save_path": "outputs/multi-agents/dqn.pth",
        },
        {
            "algo": "ddpg",
            "env": "beamng_camera",
            "vehicle_id": "taxi",
            "color": "Red",
            "save_path": "outputs/multi-agents/ddpg.pth",
        },
    ]
    with patch("core.cli.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        _, slots = build_multi_session(specs, map_name="gridmap_v2")

    assert slots[0].n_states == 14
    assert slots[1].n_states == 262
    assert slots[1].perception == "camera"
