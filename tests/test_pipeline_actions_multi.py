"""Tests for the multi-agent action builder helper."""

from unittest.mock import MagicMock, patch

from core.pipeline_actions import build_multi_session


def test_build_multi_session_builds_one_agent_per_spec():
    specs = [
        {
            "algo": "dqn",
            "sensor": "lidar",
            "color": "Yellow",
            "save_path": "outputs/multi-agents/dqn.pth",
        },
        {
            "algo": "ddpg",
            "sensor": "lidar",
            "color": "Red",
            "save_path": "outputs/multi-agents/ddpg.pth",
        },
    ]
    # Patch the env so no BeamNG launch happens.
    with patch("core.pipeline_actions.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        env, slots = build_multi_session(specs, map_name="gridmap_v2")

    assert len(slots) == 2
    assert slots[0].output == "fixed"
    assert slots[1].output == "continuous"
    # every slot has a concrete agent instance attached
    assert all(s.agent is not None for s in slots)


def _multi_env_kwargs(**session_kwargs):
    """Kwargs build_multi_session hands the env, with no BeamNG launch."""
    specs = [{"algo": "dqn", "sensor": "lidar", "color": "Red", "save_path": "x.pth"}]
    with patch("core.pipeline_actions.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        build_multi_session(specs, map_name="italy", **session_kwargs)
    return EnvCls.call_args.kwargs


def test_a_chosen_game_track_reaches_the_multi_env():
    kwargs = _multi_env_kwargs(track="mixedCircuit1")
    assert kwargs["track"] == "mixedCircuit1"


def test_no_track_trains_on_the_generated_paths():
    assert _multi_env_kwargs()["track"] is None


def test_a_game_track_disables_random_path():
    # There is one authored line, so there is no random path to deal per episode;
    # leaving it on would make the env try to pick among a single path.
    assert _multi_env_kwargs(random_path=True, track="mixedCircuit1")["random_path"] is False
    assert _multi_env_kwargs(random_path=True)["random_path"] is True


def test_build_multi_session_passes_map_to_env():
    specs = [
        {
            "algo": "dqn",
            "sensor": "lidar",
            "color": "Yellow",
            "save_path": "outputs/multi-agents/dqn.pth",
        },
    ]
    with patch("core.pipeline_actions.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        build_multi_session(specs, map_name="italy")
        _, kwargs = EnvCls.call_args
        assert kwargs["map_name"] == "italy"


def test_build_multi_session_sizes_agent_to_each_sensor():
    # A camera slot (262 states) and a lidar slot (14 states) -> agents built with
    # the matching observation size for each.
    specs = [
        {
            "algo": "dqn",
            "sensor": "lidar",
            "color": "Yellow",
            "save_path": "outputs/multi-agents/dqn.pth",
        },
        {
            "algo": "ddpg",
            "sensor": "camera",
            "color": "Red",
            "save_path": "outputs/multi-agents/ddpg.pth",
        },
    ]
    with patch("core.pipeline_actions.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        _, slots = build_multi_session(specs, map_name="gridmap_v2")

    assert slots[0].n_states == 14
    assert slots[1].n_states == 262
    assert slots[1].sensor == "camera"


def test_build_multi_session_sizes_agent_with_flags():
    # Body orientation + road info on a lidar slot -> 14 + 2 + 6 = 22 states.
    specs = [
        {
            "algo": "dqn",
            "sensor": "lidar",
            "color": "Yellow",
            "save_path": "outputs/multi-agents/dqn.pth",
            "body_orientation": True,
            "road_info": True,
        },
    ]
    with patch("core.pipeline_actions.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        _, slots = build_multi_session(specs, map_name="gridmap_v2")
    assert slots[0].n_states == 22
    assert slots[0].body_orientation is True
    assert slots[0].road_info is True
    # The built DQN agent's network must also be sized to 22 inputs.
    assert slots[0].agent.q_net.feature[0].in_features == 22
