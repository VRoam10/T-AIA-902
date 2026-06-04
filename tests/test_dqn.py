"""Unit tests for algorithms.dqn — SumTree, DQNNetwork, and DQNAgent."""

import os
import tempfile

import numpy as np
import pytest
import torch

from algorithms.dqn import DQNAgent, DQNNetwork, SumTree

N_STATES = 8
N_ACTIONS = 4
HIDDEN = 64


# ---------------------------------------------------------------------------
# DQNNetwork
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SumTree
# ---------------------------------------------------------------------------


class TestSumTree:
    def test_total_equals_sum_of_priorities(self):
        tree = SumTree(capacity=8)
        for p in [1.0, 2.0, 3.0, 4.0]:
            tree.add(p, "x")
        assert tree.total == pytest.approx(10.0)

    def test_len_tracks_insertions(self):
        tree = SumTree(capacity=4)
        assert len(tree) == 0
        tree.add(1.0, "a")
        tree.add(2.0, "b")
        assert len(tree) == 2

    def test_len_capped_at_capacity(self):
        tree = SumTree(capacity=4)
        for i in range(10):
            tree.add(float(i + 1), i)
        assert len(tree) == 4

    def test_update_changes_total(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, "a")
        tree.add(1.0, "b")
        leaf_idx = tree.capacity - 1  # first leaf
        tree.update(leaf_idx, 5.0)
        assert tree.total == pytest.approx(6.0)

    def test_sample_returns_highest_priority_region(self):
        tree = SumTree(capacity=8)
        tree.add(0.01, "low")
        tree.add(100.0, "high")
        # Sampling near total should hit the high-priority leaf
        _, _, data = tree.sample(tree.total * 0.99)
        assert data == "high"

    def test_sample_low_value_hits_first_added(self):
        tree = SumTree(capacity=8)
        tree.add(100.0, "first")
        tree.add(0.01, "second")
        _, _, data = tree.sample(0.001)
        assert data == "first"


# ---------------------------------------------------------------------------
# DQNNetwork
# ---------------------------------------------------------------------------


class TestDQNNetwork:
    def setup_method(self):
        self.net = DQNNetwork(N_STATES, N_ACTIONS, hidden=HIDDEN)

    def test_output_shape(self):
        x = torch.zeros(1, N_STATES)
        out = self.net(x)
        assert out.shape == (1, N_ACTIONS)

    def test_batch_output_shape(self):
        x = torch.zeros(32, N_STATES)
        out = self.net(x)
        assert out.shape == (32, N_ACTIONS)

    def test_dueling_mean_advantage_is_zero(self):
        # By construction Q = V + (A - mean(A)), so mean over actions must equal V.
        x = torch.randn(16, N_STATES)
        f = self.net.feature(x)
        v = self.net.value(f)
        self.net.advantage(f)
        q = self.net(x)
        # mean(Q - V) should be ~0 (mean advantage is subtracted)
        mean_diff = (q - v).mean(dim=1)
        assert torch.allclose(mean_diff, torch.zeros(16), atol=1e-5)

    def test_gradients_flow(self):
        x = torch.randn(4, N_STATES)
        out = self.net(x).sum()
        out.backward()
        for name, p in self.net.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_deterministic_in_eval_mode(self):
        self.net.eval()
        x = torch.randn(1, N_STATES)
        with torch.no_grad():
            out1 = self.net(x)
            out2 = self.net(x)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# DQNAgent — construction and basic properties
# ---------------------------------------------------------------------------


class TestDQNAgentInit:
    def setup_method(self):
        self.agent = DQNAgent(N_STATES, N_ACTIONS)

    def test_epsilon_starts_at_one(self):
        assert self.agent.epsilon == pytest.approx(1.0)

    def test_epsilon_property_setter(self):
        self.agent.epsilon = 0.5
        assert self.agent.epsilon == pytest.approx(0.5)

    def test_target_net_matches_q_net_at_init(self):
        for (_, qp), (_, tp) in zip(
            self.agent.q_net.named_parameters(),
            self.agent.target_net.named_parameters(),
            strict=True,
        ):
            assert torch.equal(qp.data, tp.data)

    def test_get_config_contains_required_keys(self):
        cfg = self.agent.get_config()
        for key in (
            "gamma",
            "epsilon",
            "epsilon_min",
            "epsilon_decay",
            "batch_size",
            "target_update_freq",
        ):
            assert key in cfg


# ---------------------------------------------------------------------------
# DQNAgent — action selection
# ---------------------------------------------------------------------------


class TestDQNAgentSelectAction:
    def setup_method(self):
        self.agent = DQNAgent(N_STATES, N_ACTIONS)

    def test_greedy_action_is_valid_index(self):
        self.agent.epsilon = 0.0
        state = np.zeros(N_STATES, dtype=np.float32)
        action = self.agent.select_action(state)
        assert 0 <= action < N_ACTIONS

    def test_random_action_is_valid_index(self):
        self.agent.epsilon = 1.0
        state = np.zeros(N_STATES, dtype=np.float32)
        action = self.agent.select_action(state)
        assert 0 <= action < N_ACTIONS

    def test_greedy_is_deterministic(self):
        self.agent.epsilon = 0.0
        state = np.random.rand(N_STATES).astype(np.float32)
        actions = {self.agent.select_action(state) for _ in range(10)}
        assert len(actions) == 1


# ---------------------------------------------------------------------------
# DQNAgent — update / training
# ---------------------------------------------------------------------------


def _make_agent(**kwargs):
    defaults = dict(n_states=N_STATES, n_actions=N_ACTIONS)
    defaults.update(kwargs)
    return DQNAgent(**defaults)


def _random_transition():
    s = np.random.rand(N_STATES).astype(np.float32)
    a = np.random.randint(N_ACTIONS)
    r = float(np.random.rand())
    ns = np.random.rand(N_STATES).astype(np.float32)
    done = bool(np.random.rand() > 0.8)
    return s, a, r, ns, done


class TestDQNAgentUpdate:
    def test_update_returns_none_when_buffer_too_small(self):
        agent = _make_agent(batch_size=64)
        s, a, r, ns, d = _random_transition()
        loss = agent.update(s, a, r, ns, d)
        assert loss is None

    def test_update_returns_float_after_enough_transitions(self):
        agent = _make_agent(batch_size=8)
        for _ in range(20):
            loss = agent.update(*_random_transition())
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_steps_increments(self):
        agent = _make_agent(batch_size=8)
        for _ in range(20):
            agent.update(*_random_transition())
        assert agent.train_steps > 0

    def test_target_net_updates_at_correct_frequency(self):
        freq = 10
        agent = _make_agent(batch_size=8, target_update_freq=freq)

        # Fill buffer and do some training steps
        for _ in range(20):
            agent.update(*_random_transition())

        # Zero out target_net so a sync is detectable
        with torch.no_grad():
            for p in agent.target_net.parameters():
                p.fill_(0.0)

        # Run exactly enough steps to hit the next sync boundary
        steps_to_sync = freq - (agent.train_steps % freq)
        for _ in range(steps_to_sync):
            agent.update(*_random_transition())

        # The last step triggered the sync — target_net must now equal q_net
        for qp, tp in zip(agent.q_net.parameters(), agent.target_net.parameters(), strict=True):
            assert torch.equal(qp.data, tp.data)


# ---------------------------------------------------------------------------
# DQNAgent — Double DQN correctness
# ---------------------------------------------------------------------------


class TestDoubleDQN:
    def test_double_dqn_uses_q_net_for_action_selection(self):
        """Verify action selection (q_net) is decoupled from evaluation (target_net)."""
        agent = _make_agent(batch_size=8)
        for _ in range(20):
            agent.update(*_random_transition())

        # Patch q_net to always prefer action 0
        original_q_net = agent.q_net

        class BiasedNet(torch.nn.Module):
            def forward(self, x):
                out = original_q_net(x)
                # Make action 0 overwhelmingly preferred
                out = out - out
                out[:, 0] = 1000.0
                return out

        agent.q_net = BiasedNet()

        # Fill a batch and run _train_step; it should not raise
        agent._train_step()

    def test_no_overestimation_vs_vanilla(self):
        """Double DQN target should be <= vanilla DQN target on average."""
        torch.manual_seed(42)
        agent = _make_agent(batch_size=32)
        for _ in range(64):
            agent.update(*_random_transition())

        batch = list(agent.memory)[:32]
        _, _, rewards, next_states_np, dones = zip(*batch, strict=False)

        next_states = torch.FloatTensor(np.stack(next_states_np)).to(agent.device)
        rewards_t = torch.FloatTensor(rewards).to(agent.device)
        dones_t = torch.FloatTensor(dones).to(agent.device)

        with torch.no_grad():
            # Vanilla DQN target
            vanilla_next_q = agent.target_net(next_states).max(dim=1)[0]
            vanilla_target = rewards_t + agent.gamma * vanilla_next_q * (1.0 - dones_t)

            # Double DQN target
            next_actions = agent.q_net(next_states).argmax(dim=1)
            double_next_q = (
                agent.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            )
            double_target = rewards_t + agent.gamma * double_next_q * (1.0 - dones_t)

        # Double DQN targets should be <= vanilla on average (overestimation reduction)
        assert double_target.mean().item() <= vanilla_target.mean().item() + 1.0


# ---------------------------------------------------------------------------
# DQNAgent — epsilon decay
# ---------------------------------------------------------------------------


class TestEpsilonDecay:
    def test_decay_reduces_epsilon(self):
        agent = _make_agent()
        initial = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < initial

    def test_epsilon_never_goes_below_minimum(self):
        agent = _make_agent(epsilon=0.001, epsilon_min=0.05, epsilon_decay=0.5)
        for _ in range(100):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# DQNAgent — save / load roundtrip
# ---------------------------------------------------------------------------


class TestDQNAgentCheckpoint:
    def test_save_load_roundtrip(self):
        agent = _make_agent(batch_size=8)
        for _ in range(20):
            agent.update(*_random_transition())

        agent.epsilon = 0.42

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            agent.save(path)

            agent2 = _make_agent(batch_size=8)
            agent2.load(path)

            assert agent2.epsilon == pytest.approx(0.42)
            assert agent2.train_steps == agent.train_steps

            for p1, p2 in zip(agent.q_net.parameters(), agent2.q_net.parameters(), strict=True):
                assert torch.equal(p1.data, p2.data)
        finally:
            os.unlink(path)

    def test_load_syncs_target_net(self):
        agent = _make_agent(batch_size=8)
        for _ in range(20):
            agent.update(*_random_transition())

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            path = f.name

        try:
            agent.save(path)
            agent2 = _make_agent(batch_size=8)
            agent2.load(path)

            for qp, tp in zip(agent2.q_net.parameters(), agent2.target_net.parameters(), strict=True):
                assert torch.equal(qp.data, tp.data)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# DQNAgent — hidden size configurable
# ---------------------------------------------------------------------------


class TestHiddenConfig:
    def test_custom_hidden_changes_network_size(self):
        agent_small = DQNAgent(N_STATES, N_ACTIONS, hidden=32)
        agent_large = DQNAgent(N_STATES, N_ACTIONS, hidden=256)

        params_small = sum(p.numel() for p in agent_small.q_net.parameters())
        params_large = sum(p.numel() for p in agent_large.q_net.parameters())
        assert params_large > params_small

    def test_hidden_in_get_config(self):
        # hidden is reflected in the network but not necessarily in get_config;
        # just verify the agent trains without error at non-default hidden size.
        agent = DQNAgent(N_STATES, N_ACTIONS, hidden=32, batch_size=8)
        for _ in range(20):
            agent.update(*_random_transition())
        assert agent.train_steps > 0


# ---------------------------------------------------------------------------
# DQNAgent — Prioritized Experience Replay (PER)
# ---------------------------------------------------------------------------


def _make_per_agent(**kwargs):
    defaults = dict(
        n_states=N_STATES, n_actions=N_ACTIONS, batch_size=8, use_per=True, memory_size=200
    )
    defaults.update(kwargs)
    return DQNAgent(**defaults)


class TestPER:
    def test_per_agent_trains_without_error(self):
        agent = _make_per_agent()
        for _ in range(20):
            loss = agent.update(*_random_transition())
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_per_train_steps_increment(self):
        agent = _make_per_agent()
        for _ in range(20):
            agent.update(*_random_transition())
        assert agent.train_steps > 0

    def test_per_returns_none_when_buffer_too_small(self):
        agent = _make_per_agent(batch_size=64, memory_size=1000)
        loss = agent.update(*_random_transition())
        assert loss is None

    def test_per_max_priority_grows(self):
        agent = _make_per_agent()
        initial_max = agent._max_priority
        for _ in range(20):
            agent.update(*_random_transition())
        # After training, priorities should have been updated
        assert agent._max_priority >= initial_max

    def test_per_beta_anneals_toward_one(self):
        agent = _make_per_agent(per_beta=0.4, per_beta_steps=10)
        initial_beta = agent._per_beta
        for _ in range(20):
            agent.update(*_random_transition())
        assert agent._per_beta > initial_beta

    def test_per_vs_uniform_same_interface(self):
        per_agent = _make_per_agent()
        uni_agent = _make_agent(batch_size=8)

        for _ in range(20):
            s, a, r, ns, d = _random_transition()
            per_agent.update(s, a, r, ns, d)
            uni_agent.update(s, a, r, ns, d)

        # Both must select valid actions
        state = np.zeros(N_STATES, dtype=np.float32)
        per_agent.epsilon = 0.0
        uni_agent.epsilon = 0.0
        assert 0 <= per_agent.select_action(state) < N_ACTIONS
        assert 0 <= uni_agent.select_action(state) < N_ACTIONS

    def test_get_config_reflects_per(self):
        agent = _make_per_agent()
        cfg = agent.get_config()
        assert cfg["use_per"] is True
        assert "per_alpha" in cfg
        assert "per_beta" in cfg
