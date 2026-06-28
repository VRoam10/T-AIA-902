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
    with patch("core.cli.BeamNGMultiEnv") as EnvCls:
        EnvCls.return_value = MagicMock()
        _, slots = build_multi_session(specs, map_name="gridmap_v2")
    assert slots[0].n_states == 18
    assert slots[0].body_orientation is True
    assert slots[0].wheel_terrain is True
    # The built DQN agent's network must also be sized to 18 inputs.
    assert slots[0].agent.q_net.feature[0].in_features == 18


def test_ask_bool_parses_yes_no():
    from core.cli import _ask_bool

    with patch("builtins.input", return_value="y"):
        assert _ask_bool("?") is True
    with patch("builtins.input", return_value=""):
        assert _ask_bool("?", default=False) is False
    with patch("builtins.input", return_value="yes"):
        assert _ask_bool("?") is True
    with patch("builtins.input", return_value="n"):
        assert _ask_bool("?", default=True) is False


def test_human_play_menu_omits_wheel_terrain_and_offers_random_path(monkeypatch):
    import core.cli as cli

    asked_prompts = []

    def fake_ask_bool(prompt, default=False):
        asked_prompts.append(prompt)
        # Enable only the "random path / checkpoints" toggle.
        return "random" in prompt.lower()

    monkeypatch.setattr(cli, "_pick_beamng_options", lambda: ("italy", "taxi"))
    monkeypatch.setattr(cli, "_pick", lambda options, prompt="Select": "None")
    monkeypatch.setattr(cli, "_ask_int", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "_ask_bool", fake_ask_bool)
    monkeypatch.setattr(cli.registry, "list_environments", lambda: ["beamng", "beamng_camera"])

    captured = {}

    def fake_factory(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(cli.registry, "get_environment", lambda name: {"factory": fake_factory})

    cli._human_play_menu()

    # The per-wheel road position prompt is gone from human play.
    assert not any("wheel" in p.lower() for p in asked_prompts)
    # A random path / checkpoints option is offered and forwarded to the env.
    assert captured["random_path"] is True
    # wheel_terrain is forced off (its RoadsSensor poll can freeze BeamNG).
    assert captured["wheel_terrain"] is False


def test_obs_suffix_encodes_predicted_checkpoint_count_and_body_orientation():
    from core.cli import _obs_suffix

    # No hints and no body orientation -> no suffix, so existing filenames are unchanged.
    assert _obs_suffix(0, False) == ""
    # Predicted-checkpoint hints -> a "_{n}hints" fragment that distinguishes the model.
    assert _obs_suffix(1, False) == "_1hints"
    assert _obs_suffix(3, False) == "_3hints"
    # Body orientation (pitch + roll) in obs -> a "_body" fragment.
    assert _obs_suffix(0, True) == "_body"
    # Both options -> hints first, then body, so the name is stable.
    assert _obs_suffix(2, True) == "_2hints_body"
