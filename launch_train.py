"""Direct launcher for DQN + BeamNG training (bypasses interactive menu)."""
import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

import algorithms  # noqa: F401
import environments  # noqa: F401

from core.registry import registry
from core.runner import PipelineRunner

algo_name = "dqn"
env_name = "beamng"
n_episodes = 500
save_path = "outputs/dqn_beamng.pth"
plot_path = "outputs/dqn_beamng_training.png"

algo_info = registry.get_algorithm(algo_name)
env_info = registry.get_environment(env_name)

# Build agent with default config
cls = algo_info["class"]
config = dict(algo_info["default_config"])
meta = env_info["metadata"]
config["n_states"] = meta.get("n_states", 5)
config["n_actions"] = meta.get("n_actions", 6)
agent = cls(**config)

# Resume if checkpoint exists
if os.path.exists(save_path):
    print(f"Resuming from '{save_path}'")
    agent.load(save_path)

os.makedirs("outputs", exist_ok=True)
env = env_info["factory"]()

print(f"\n--- Training {algo_name} on {env_name} ({n_episodes} episodes) ---\n")
runner = PipelineRunner()
try:
    runner.train(agent, env, n_episodes=n_episodes, save_path=save_path, plot_path=plot_path)
finally:
    env.close()
