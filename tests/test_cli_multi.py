"""Tests for the multi-agent CLI builder helper."""

from unittest.mock import MagicMock, patch

from core.cli import build_multi_session


def test_build_multi_session_builds_one_agent_per_spec():
    specs = [
        {"algo": "dqn", "vehicle_id": "taxi", "color": "Yellow", "save_path": "outputs/dqn.pth"},
        {"algo": "ddpg", "vehicle_id": "taxi", "color": "Red", "save_path": "outputs/ddpg.pth"},
    ]
    # Patch the env so no BeamNG launch happens; capture the slots passed in.
    with patch("core.cli.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock(n_states=14)
        env, slots = build_multi_session(specs, map_name="gridmap_v2", trajectory_hints=0)

    assert len(slots) == 2
    assert slots[0].action_space == "discrete"
    assert slots[1].action_space == "continuous"
    # every slot has a concrete agent instance attached
    assert all(s.agent is not None for s in slots)


def test_build_multi_session_passes_map_and_hints_to_env():
    specs = [
        {"algo": "dqn", "vehicle_id": "taxi", "color": "Yellow", "save_path": "outputs/dqn.pth"},
    ]
    with patch("core.cli.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock(n_states=16)
        build_multi_session(specs, map_name="italy", trajectory_hints=1)
        _, kwargs = EnvCls.call_args
        assert kwargs["map_name"] == "italy"
        assert kwargs["trajectory_hints"] == 1
