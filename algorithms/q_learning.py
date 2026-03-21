import numpy as np

from core.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Q-Learning agent for discrete environments like Taxi-v3."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        state_type: str = "discrete",
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self._epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        if np.random.random() < self._epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        current_q = self.q_table[state, action]
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - current_q)
        return None

    def decay_epsilon(self):
        self._epsilon = max(self.epsilon_min, self._epsilon * self.epsilon_decay)

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        self._epsilon = value

    def save(self, path: str):
        np.save(path, self.q_table)

    def load(self, path: str):
        self.q_table = np.load(path)

    def get_config(self) -> dict:
        return {
            "learning_rate": self.lr,
            "discount_factor": self.gamma,
            "epsilon": self._epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
