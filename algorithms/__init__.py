"""Register all algorithms with the pipeline registry."""

from algorithms.ddpg import DDPGAgent
from algorithms.dqn import DQNAgent
from algorithms.q_learning import QLearningAgent
from algorithms.td3 import TD3Agent
from core.registry import registry

# Continuous-action algorithms (DDPG/TD3) can only act in continuous action
# spaces, so they are restricted to the BeamNG driving envs and must never be
# offered for a discrete-action env like Taxi-v3.
_CONTINUOUS_ENVS = ["beamng", "beamng_lidar", "beamng_continuous", "beamng_camera"]

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
    compatible_envs=_CONTINUOUS_ENVS,
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
        "hidden": 128,
        "use_per": False,
    },
    compatible_envs=["taxi", "beamng", "beamng_lidar", "beamng_predicted"],
)

registry.register_algorithm(
    "dqn_per",
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
        "hidden": 128,
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "per_beta_steps": 50_000,
    },
    compatible_envs=[
        "taxi",
        "beamng",
        "beamng_lidar",
    ],
)

registry.register_algorithm(
    "td3",
    TD3Agent,
    default_config={
        "n_actions": 2,
        "hidden": 128,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "exploration_noise": 0.3,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.995,
        "batch_size": 64,
        "memory_size": 100_000,
        "warmup_steps": 1000,
        "device": "auto",
    },
    compatible_envs=_CONTINUOUS_ENVS,
)
