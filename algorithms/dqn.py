import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent


class SumTree:
    """Binary tree where every internal node stores the sum of its children.

    Supports O(log n) priority-weighted sampling — used by PER to sample
    transitions proportional to their TD error without a full sort each step.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self._ptr = 0
        self._size = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, transition) -> None:
        leaf = self._ptr + self.capacity - 1
        self.data[self._ptr] = transition
        self._update(leaf, priority)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update(self, leaf_idx: int, priority: float) -> None:
        self._update(leaf_idx, priority)

    def sample(self, s: float):
        """Return (leaf_idx, priority, transition) for cumulative value s."""
        idx = self._retrieve(0, s)
        return idx, self.tree[idx], self.data[idx - self.capacity + 1]

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])

    def _update(self, idx: int, priority: float) -> None:
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def __len__(self) -> int:
        return self._size


class DQNNetwork(nn.Module):
    """Dueling Q-network: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))."""

    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + (a - a.mean(dim=1, keepdim=True))


class DQNAgent(BaseAgent):
    """Double DQN + Dueling architecture with experience replay and a target network.

    Optionally uses Prioritized Experience Replay (use_per=True) to sample
    transitions weighted by their TD error, improving sample efficiency.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 20_000,
        target_update_freq: int = 100,
        hidden: int = 128,
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_steps: int = 50_000,
        per_epsilon: float = 1e-6,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self._epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0
        self.use_per = use_per

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Using device: {self.device}")

        self.q_net = DQNNetwork(n_states, n_actions, hidden).to(self.device)
        self.target_net = DQNNetwork(n_states, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        if use_per:
            self._tree = SumTree(memory_size)
            self._per_alpha = per_alpha
            self._per_beta = per_beta
            self._per_beta_increment = (1.0 - per_beta) / per_beta_steps
            self._per_epsilon = per_epsilon
            self._max_priority = 1.0
        else:
            self.memory: deque = deque(maxlen=memory_size)

    # -- BaseAgent interface -------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self._epsilon:
            return random.randrange(self.n_actions)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax(dim=1).item())

    def update(self, state, action: int, reward: float, next_state, done: bool) -> float | None:
        self._store(state, action, reward, next_state, done)
        return self._train_step()

    def decay_epsilon(self):
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    def save(self, path: str = "beamng_dqn.pth"):
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self._epsilon,
                "train_steps": self.train_steps,
            },
            path,
        )
        print(f"[DQNAgent] Saved -> {path}")

    def load(self, path: str = "beamng_dqn.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["q_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._epsilon = ckpt.get("epsilon", self.epsilon_min)
        self.train_steps = ckpt.get("train_steps", 0)
        print(f"[DQNAgent] Loaded <- {path}  (eps={self._epsilon:.3f}, steps={self.train_steps})")

    def get_config(self) -> dict:
        cfg = {
            "gamma": self.gamma,
            "epsilon": self._epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
            "use_per": self.use_per,
        }
        if self.use_per:
            cfg.update(
                {
                    "per_alpha": self._per_alpha,
                    "per_beta": self._per_beta,
                }
            )
        return cfg

    # -- Internal ------------------------------------------------------------

    def _store(self, state, action, reward, next_state, done):
        transition = (
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )
        if self.use_per:
            self._tree.add(self._max_priority, transition)
        else:
            self.memory.append(transition)

    def _train_step(self) -> float | None:
        if self.use_per:
            return self._train_step_per()
        return self._train_step_uniform()

    def _train_step_uniform(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)

        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def _train_step_per(self) -> float | None:
        if len(self._tree) < self.batch_size:
            return None

        segment = self._tree.total / self.batch_size
        leaf_indices, priorities, transitions = [], [], []

        for i in range(self.batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, transition = self._tree.sample(s)
            if transition is None:
                continue
            leaf_indices.append(idx)
            priorities.append(priority)
            transitions.append(transition)

        if len(transitions) < self.batch_size:
            return None

        # Importance-sampling weights to correct for the sampling bias.
        # β is annealed from its initial value toward 1 over training.
        priorities_np = np.array(priorities, dtype=np.float32)
        probs = priorities_np / self._tree.total
        weights = (len(self._tree) * probs) ** (-self._per_beta)
        weights /= weights.max()
        weights_t = torch.FloatTensor(weights).to(self.device)
        self._per_beta = min(1.0, self._per_beta + self._per_beta_increment)

        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # Weight the per-element loss by IS weights before reducing
        elementwise_loss = self.loss_fn(current_q, target_q)
        loss = (weights_t * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities with fresh TD errors
        with torch.no_grad():
            td_errors = (target_q - current_q).abs().cpu().numpy()
        new_priorities = (td_errors + self._per_epsilon) ** self._per_alpha
        self._max_priority = max(self._max_priority, float(new_priorities.max()))
        for idx, p in zip(leaf_indices, new_priorities):
            self._tree.update(idx, float(p))

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
