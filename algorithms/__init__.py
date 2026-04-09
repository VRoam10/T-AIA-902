"""Register all algorithms with the pipeline registry."""

from algorithms.ddpg import DDPGAgent
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
    "ddpg",
    DDPGAgent,
    default_config={
        "actor_lr": 1e-4,
        "critic_lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.99,
        "batch_size": 128,
        "memory_size": 100_000,
        "noise_theta": 0.15,
        "noise_sigma": 0.2,
        "warmup_steps": 128,
        "updates_per_step": 4,
    },
    compatible_envs=None,
)

registry.register_algorithm(
    "dqn",
    DQNAgent,
    default_config={
        "lr": 1e-3,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.95,
        "batch_size": 64,
        "memory_size": 20_000,
        "target_update_freq": 100,
    },
    compatible_envs=None,
)
