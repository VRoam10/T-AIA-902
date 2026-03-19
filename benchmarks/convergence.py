import numpy as np

from core.base_benchmark import BaseBenchmark
from core.runner import PipelineRunner


class ConvergenceBenchmark(BaseBenchmark):
    """Measures how many episodes until average reward exceeds a threshold."""

    name = "convergence"
    description = "Episodes to reach a reward threshold (configurable)"

    def run(self, agent_cls, env_factory, config: dict) -> dict:
        env = env_factory()
        metadata = config.get("env_metadata", {})

        agent_params = dict(config.get("agent_params", {}))
        agent_params["n_states"] = metadata.get("n_states", 5)
        agent_params["n_actions"] = metadata.get("n_actions", 6)
        agent = agent_cls(**agent_params)

        max_episodes = config.get("max_episodes", 2000)
        threshold = config.get("threshold", 7.0)
        window = config.get("window", 100)

        runner = PipelineRunner()
        history = runner.train(agent, env, n_episodes=max_episodes)

        rewards = history["rewards"]
        convergence_ep = None
        for i in range(window, len(rewards)):
            avg = np.mean(rewards[i - window : i])
            if avg >= threshold:
                convergence_ep = i
                break

        env.close()

        return {
            "converged": convergence_ep is not None,
            "convergence_episode": convergence_ep,
            "final_avg_reward": float(np.mean(rewards[-window:])) if len(rewards) >= window else float(np.mean(rewards)),
            "total_episodes": len(rewards),
        }
