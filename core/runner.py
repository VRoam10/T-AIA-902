import os
import time

import numpy as np
import matplotlib.pyplot as plt

from core.base_agent import BaseAgent


class PipelineRunner:
    """Generic train / evaluate / plot runner for any BaseAgent + Gymnasium env."""

    def train(
        self,
        agent: BaseAgent,
        env,
        n_episodes: int,
        save_path: str = None,
        save_every: int = 50,
        log_interval: int = 20,
        time_limit: float = None,
        plot_path: str = None,
    ) -> dict:
        """Run the training loop. Returns history dict."""
        rewards = []
        steps = []
        start = time.time()

        try:
            for ep in range(1, n_episodes + 1):
                if time_limit and (time.time() - start) >= time_limit:
                    print(f"\nTime limit reached after {ep - 1} episodes.")
                    break

                state = self._reset_env(env)
                ep_reward = 0.0
                ep_losses = []
                done = False

                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, info = self._step_env(env, action)
                    loss = agent.update(state, action, reward, next_state, done)
                    if loss is not None:
                        ep_losses.append(loss)
                    state = next_state
                    ep_reward += reward

                agent.decay_epsilon()
                rewards.append(ep_reward)
                ep_steps = info.get("steps", 0) if isinstance(info, dict) else 0
                steps.append(ep_steps)

                if ep % log_interval == 0:
                    avg_r = np.mean(rewards[-log_interval:])
                    avg_loss = np.mean(ep_losses) if ep_losses else float("nan")
                    print(
                        f"Ep {ep:4d}/{n_episodes} | "
                        f"Reward {ep_reward:8.1f} | "
                        f"Avg{log_interval} {avg_r:8.1f} | "
                        f"eps {agent.epsilon:.3f} | "
                        f"Loss {avg_loss:.4f}"
                    )

                if save_path and ep % save_every == 0:
                    agent.save(save_path)
                    if plot_path:
                        self._save_plot(rewards, steps, "Training", plot_path, ep)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

        finally:
            if save_path:
                agent.save(save_path)
            if plot_path:
                self._save_plot(rewards, steps, "Training", plot_path, len(rewards))

        elapsed = time.time() - start
        print(f"\nTraining complete in {elapsed:.1f}s ({len(rewards)} episodes).")
        return {"rewards": rewards, "steps": steps}

    def evaluate(
        self,
        agent: BaseAgent,
        env,
        n_episodes: int = 10,
    ) -> dict:
        """Run the agent in pure exploitation mode (epsilon=0)."""
        old_eps = agent.epsilon
        agent.epsilon = 0.0

        rewards = []
        steps = []

        try:
            for ep in range(1, n_episodes + 1):
                state = self._reset_env(env)
                ep_reward = 0.0
                done = False

                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, info = self._step_env(env, action)
                    state = next_state
                    ep_reward += reward

                ep_steps = info.get("steps", 0) if isinstance(info, dict) else 0
                rewards.append(ep_reward)
                steps.append(ep_steps)
                print(f"  Eval {ep}/{n_episodes} | Reward {ep_reward:.1f} | Steps {ep_steps}")

        finally:
            agent.epsilon = old_eps

        avg = np.mean(rewards)
        print(f"\nEvaluation: avg reward = {avg:.2f} over {n_episodes} episodes.")
        return {"rewards": rewards, "steps": steps, "avg_reward": float(avg)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_env(env):
        """Normalise Gymnasium reset (obs, info) vs old-style (obs)."""
        result = env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result

    @staticmethod
    def _step_env(env, action):
        """Normalise 5-tuple to 4-tuple (obs, reward, done, info)."""
        result = env.step(action)
        if len(result) == 5:
            obs, rew, terminated, truncated, info = result
            return obs, rew, terminated or truncated, info
        return result

    @staticmethod
    def _save_plot(rewards, steps, title, path, episode=None):
        if not rewards:
            return
        window = min(20, len(rewards))
        eps = range(1, len(rewards) + 1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        subtitle = f" — Episode {episode}" if episode else ""
        fig.suptitle(f"{title}{subtitle}")

        ax1.plot(eps, rewards, alpha=0.3, label="Reward")
        if len(rewards) >= window:
            roll = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax1.plot(range(window, len(rewards) + 1), roll, label=f"Rolling avg ({window})")
        ax1.set_ylabel("Total Reward")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(eps, steps, alpha=0.3, label="Steps", color="orange")
        if len(steps) >= window:
            roll = np.convolve(steps, np.ones(window) / window, mode="valid")
            ax2.plot(range(window, len(steps) + 1), roll,
                     label=f"Rolling avg ({window})", color="darkorange")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps per Episode")
        ax2.legend()
        ax2.grid(True)

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Plot saved -> {path}")
