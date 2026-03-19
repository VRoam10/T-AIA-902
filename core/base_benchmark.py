from abc import ABC, abstractmethod
from typing import Callable, Type


class BaseBenchmark(ABC):
    """Abstract base class for benchmarks that evaluate agent/env combos."""

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def run(self, agent_cls: Type, env_factory: Callable, config: dict) -> dict:
        """Run the benchmark. Returns a results dict."""
        ...

    def report(self, results: dict) -> str:
        """Format results as a human-readable string."""
        lines = [f"=== {self.name} ==="]
        for k, v in results.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
