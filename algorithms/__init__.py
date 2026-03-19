"""Register all algorithms with the pipeline registry."""

from algorithms.dqn import DQNAgent
from algorithms.q_learning import QLearningAgent
from core.registry import registry

registry.register_algorithm(
    "q_learning",
    QLearningAgent,
    default_config={
        "learning_rate": 0.85,
        "discount_factor": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.9975,
    },
    compatible_envs=["taxi"],
)

registry.register_algorithm(
    "dqn",
    DQNAgent,
    default_config={
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.995,
        "batch_size": 64,
        "memory_size": 20_000,
        "target_update_freq": 100,
    },
    compatible_envs=None,
)
