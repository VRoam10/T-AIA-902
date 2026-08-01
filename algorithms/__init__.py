"""Register all algorithms with the pipeline registry.

Every algorithm runs on the one ``beamng`` env; which action head it gets is
derived from its name by ``beamng_spec.output_for_algo`` (dqn/dqn_per -> the
discrete action table, ddpg/td3 -> continuous controls), so ``compatible_envs``
no longer varies between them.
"""

from algorithms.ddpg import DDPGAgent
from algorithms.dqn import DQNAgent
from algorithms.td3 import TD3Agent
from core.registry import registry

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
    compatible_envs=["beamng"],
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
    compatible_envs=["beamng"],
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
    compatible_envs=["beamng"],
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
    compatible_envs=["beamng"],
)
