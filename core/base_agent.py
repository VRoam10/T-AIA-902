from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for all RL agents in the pipeline."""

    @abstractmethod
    def select_action(self, state: Any) -> int:
        """Choose an action given the current state (may be exploratory)."""
        ...

    @abstractmethod
    def update(
        self, state: Any, action: int, reward: float, next_state: Any, done: bool
    ) -> float | None:
        """Process one transition. Returns loss/metric or None.

        For tabular agents: directly updates the value table.
        For DQN-style agents: stores the transition and runs a training step.
        """
        ...

    @abstractmethod
    def decay_epsilon(self) -> None:
        """Decay exploration rate. Called once per episode."""
        ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...

    def get_config(self) -> dict:
        """Return a dict of hyperparameters for logging."""
        return {}

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return 0.0

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        pass
