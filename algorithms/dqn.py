import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent


class DQNNetwork(nn.Module):
    """Dueling fully-connected Q-network.

    Splits the last layer into a Value stream V(s) and an Advantage stream A(s,a).
    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    This helps the agent learn the state value independently of action selection.
    """

    def __init__(self, n_states: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions

        self.feature = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Value stream: how good is this state?
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Advantage stream: how much better is each action vs average?
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        # Subtract mean advantage for identifiability
        return v + (a - a.mean(dim=1, keepdim=True))


class DQNAgent(BaseAgent):
    """Double DQN agent with experience replay and a target network.

    Uses CUDA automatically when available.
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
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self._epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNAgent] Using device: {self.device}")

        self.q_net = DQNNetwork(n_states, n_actions).to(self.device)
        self.target_net = DQNNetwork(n_states, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory: deque = deque(maxlen=memory_size)

    # -- BaseAgent interface -------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self._epsilon:
            return random.randrange(self.n_actions)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax(dim=1).item())

    def update(self, state, action: int, reward: float,
               next_state, done: bool) -> float | None:
        """Store transition and run one training step."""
        self._store(state, action, reward, next_state, done)
        return self._train_step()

    def decay_epsilon(self):
        self._epsilon = max(
            self.epsilon_min, self._epsilon * self.epsilon_decay)

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
        print(
            f"[DQNAgent] Loaded <- {path}  (eps={self._epsilon:.3f}, steps={self.train_steps})")

    def get_config(self) -> dict:
        return {
            "gamma": self.gamma,
            "epsilon": self._epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
        }

    # -- Internal ------------------------------------------------------------

    def _store(self, state, action, reward, next_state, done):
        self.memory.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    def _train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.memory, self.batch_size)
        )

        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.q_net(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: q_net selects the action, target_net evaluates it.
        # Prevents overestimation bias of vanilla DQN.
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1

        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
