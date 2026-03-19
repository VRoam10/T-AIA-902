import copy
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.base_agent import BaseAgent


class Actor(nn.Module):
    """Deterministic policy: state -> continuous action in [-1, 1]."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
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
    """Twin Q-networks: (state, action) -> Q1, Q2."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


class TD3Agent(BaseAgent):
    """Twin Delayed DDPG (TD3) agent for continuous action spaces.

    Key features over DDPG:
    - Twin critics to reduce overestimation bias
    - Delayed policy updates (actor updated less frequently than critics)
    - Target policy smoothing (noise added to target actions)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        hidden: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        exploration_noise: float = 0.3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 100_000,
        warmup_steps: int = 1000,
        device: str = "auto",
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self._epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.total_steps = 0
        self.train_steps = 0

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"[TD3Agent] Using device: {self.device}")

        # Actor
        self.actor = Actor(n_states, n_actions, hidden=hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Twin critics
        self.critic = Critic(n_states, n_actions, hidden=hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory: deque = deque(maxlen=memory_size)

    # -- BaseAgent interface -------------------------------------------------

    def select_action(self, state) -> np.ndarray:
        """Select action. Pure random during warmup, then actor + noise."""
        self.total_steps += 1

        # Warmup: pure random actions to fill replay buffer with diverse experience
        if self.total_steps <= self.warmup_steps:
            return np.random.uniform(-1.0, 1.0, size=self.n_actions).astype(np.float32)

        s = torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy().flatten()

        noise = np.random.normal(0, self.exploration_noise * self._epsilon, size=self.n_actions)
        action = np.clip(action + noise, -1.0, 1.0)

        return action

    def update(self, state, action, reward: float, next_state, done: bool) -> float | None:
        """Store transition and run one training step (skip training during warmup)."""
        self._store(state, action, reward, next_state, done)
        if self.total_steps <= self.warmup_steps:
            return None
        return self._train_step()

    def decay_epsilon(self):
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    def save(self, path: str = "td3_model.pth"):
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
                "total_steps": self.total_steps,
            },
            path,
        )
        print(f"[TD3Agent] Saved -> {path}")

    def load(self, path: str = "td3_model.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self._epsilon = ckpt.get("epsilon", self.epsilon_min)
        self.train_steps = ckpt.get("train_steps", 0)
        self.total_steps = ckpt.get("total_steps", self.warmup_steps + 1)
        print(f"[TD3Agent] Loaded <- {path}  (eps={self._epsilon:.3f}, steps={self.train_steps})")

    def get_config(self) -> dict:
        return {
            "gamma": self.gamma,
            "tau": self.tau,
            "policy_delay": self.policy_delay,
            "policy_noise": self.policy_noise,
            "noise_clip": self.noise_clip,
            "exploration_noise": self.exploration_noise,
            "epsilon": self._epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "warmup_steps": self.warmup_steps,
        }

    # -- Internal ------------------------------------------------------------

    def _store(self, state, action, reward, next_state, done):
        self.memory.append(
            (
                np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                float(reward),
                np.array(next_state, dtype=np.float32),
                float(done),
            )
        )

    def _train_step(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.memory, self.batch_size), strict=False
        )

        states = torch.FloatTensor(np.stack(states)).to(self.device)
        actions = torch.FloatTensor(np.stack(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            # Twin critics: take the minimum Q-value
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * target_q * (1.0 - dones)

        # Update critics
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q1, target_q) + nn.functional.mse_loss(
            current_q2, target_q
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        self.train_steps += 1

        # Delayed policy update
        if self.train_steps % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        return critic_loss.item()

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
