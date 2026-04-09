import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent

# Continuous action dimension: [acceleration, steering] in [-1, 1]
ACTION_DIM = 2


class Actor(nn.Module):
    """Deterministic policy: state -> continuous action in [-1, 1]."""

    def __init__(self, state_dim: int, action_dim: int = ACTION_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """Q-network: (state, action) -> Q-value."""

    def __init__(self, state_dim: int, action_dim: int = ACTION_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""

    def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(size, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(self.size, dtype=np.float32)

    def sample(self) -> np.ndarray:
        dx = self.theta * -self.state + self.sigma * np.random.randn(self.size).astype(np.float32)
        self.state += dx
        return self.state


class DDPGAgent(BaseAgent):
    """Deep Deterministic Policy Gradient (DDPG) agent.

    Outputs continuous actions [acceleration, steering] in [-1, 1].
    The environment is responsible for interpreting these values
    (positive accel = throttle, negative = brake).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int = ACTION_DIM,
        state_type: str = "continuous",
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 100_000,
        noise_theta: float = 0.15,
        noise_sigma: float = 0.2,
        warmup_steps: int = 128,
        updates_per_step: int = 4,
    ):
        self.n_states = n_states
        self.n_actions = ACTION_DIM  # always 2 for continuous control
        self.gamma = gamma
        self.tau = tau
        self._epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.train_steps = 0
        self.episode = 0

        self.discrete_states = state_type == "discrete"
        net_input_size = n_states

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DDPGAgent] Using device: {self.device}")

        # Actor (policy) -> outputs [accel, steering]
        self.actor = Actor(net_input_size, self.n_actions).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic (Q-value)
        self.critic = Critic(net_input_size, self.n_actions).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Exploration noise (one per action dimension)
        self.noise = OUNoise(self.n_actions, theta=noise_theta, sigma=noise_sigma)

        # Replay buffer
        self.memory: deque = deque(maxlen=memory_size)

    # -- BaseAgent interface -------------------------------------------------

    def select_action(self, state) -> np.ndarray:
        """Return continuous action [accel, steering] in [-1, 1]."""
        s = self._encode_state(state)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]

        # Add OU noise scaled by epsilon for exploration
        if self._epsilon > 0:
            noise = self.noise.sample() * self._epsilon
            action = np.clip(action + noise, -1.0, 1.0)

        return action

    def update(self, state, action, reward: float, next_state, done: bool) -> float | None:
        self._store(state, action, reward, next_state, done)
        # Multiple gradient updates per environment step — learns faster from each experience
        last_loss = None
        for _ in range(self.updates_per_step):
            loss = self._train_step()
            if loss is not None:
                last_loss = loss
        return last_loss

    def decay_epsilon(self):
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)
        self.noise.reset()

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    def save(self, path: str = "ddpg_model.pth"):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "epsilon": self._epsilon,
                "train_steps": self.train_steps,
                "episode": self.episode,
            },
            path,
        )
        print(f"[DDPGAgent] Saved -> {path}  (episode={self.episode})")

    def load(self, path: str = "ddpg_model.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self._epsilon = ckpt.get("epsilon", self.epsilon_min)
        self.train_steps = ckpt.get("train_steps", 0)
        self.episode = ckpt.get("episode", 0)
        print(f"[DDPGAgent] Loaded <- {path}  (episode={self.episode}, eps={self._epsilon:.3f})")

    def get_config(self) -> dict:
        return {
            "gamma": self.gamma,
            "tau": self.tau,
            "epsilon": self._epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
        }

    # -- Internal ------------------------------------------------------------

    def _encode_state(self, state) -> torch.Tensor:
        if self.discrete_states:
            one_hot = np.zeros(self.n_states, dtype=np.float32)
            one_hot[int(state)] = 1.0
            return torch.FloatTensor(one_hot).unsqueeze(0).to(self.device)
        return torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)

    def _encode_state_np(self, state) -> np.ndarray:
        if self.discrete_states:
            one_hot = np.zeros(self.n_states, dtype=np.float32)
            one_hot[int(state)] = 1.0
            return one_hot
        return np.array(state, dtype=np.float32)

    def _store(self, state, action, reward, next_state, done):
        self.memory.append(
            (
                self._encode_state_np(state),
                np.array(action, dtype=np.float32),
                float(reward),
                self._encode_state_np(next_state),
                float(done),
            )
        )

    def _train_step(self) -> float | None:
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return None

        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.memory, self.batch_size), strict=False
        )

        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.FloatTensor(np.stack(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * target_q * (1.0 - dones)

        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Actor update: maximize Q-value
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.train_steps += 1
        return critic_loss.item()

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters(), strict=True):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
