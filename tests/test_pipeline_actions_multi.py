"""Tests for the multi-agent action builder helper."""

from unittest.mock import MagicMock, patch

from core.pipeline_actions import build_multi_session


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
    with patch("core.pipeline_actions.BeamNGMultiEnv") as EnvCls:
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
    with patch("core.pipeline_actions.BeamNGMultiEnv") as EnvCls:
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
    with patch("core.pipeline_actions.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        _, slots = build_multi_session(specs, map_name="gridmap_v2")

    assert slots[0].n_states == 14
    assert slots[1].n_states == 262
    assert slots[1].perception == "camera"


def test_build_multi_session_sizes_agent_with_flags():
    # Body orientation + wheel terrain on a lidar env -> 14 + 2 + 2 = 18 states.
    specs = [
        {
            "algo": "dqn",
            "env": "beamng",
            "vehicle_id": "taxi",
            "color": "Yellow",
            "save_path": "outputs/multi-agents/dqn.pth",
            "body_orientation": True,
            "wheel_terrain": True,
        },
    ]
    with patch("core.pipeline_actions.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        _, slots = build_multi_session(specs, map_name="gridmap_v2")
    assert slots[0].n_states == 18
    assert slots[0].body_orientation is True
    assert slots[0].wheel_terrain is True
    # The built DQN agent's network must also be sized to 18 inputs.
    assert slots[0].agent.q_net.feature[0].in_features == 18
