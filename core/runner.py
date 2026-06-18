import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from core.base_agent import BaseAgent
from core.seeding import seed_action_space, set_global_seed


class PipelineRunner:
    """Generic train / evaluate / plot runner for any BaseAgent + Gymnasium env."""

    def train(
        self,
        agent: BaseAgent,
        env,
        n_episodes: int,
        save_path: str = None,
        save_every: int = 50,
        time_limit: float = None,
        plot_path: str = None,
        start_episode: int = 0,
        seed: int | None = None,
    ) -> dict:
        """Run the training loop. Returns history dict.

        When ``seed`` is given, every global RNG is seeded and the seed is
        forwarded to the environment's first reset, making the run reproducible.
        """
        if seed is not None:
            set_global_seed(seed)

        rewards = []
        steps = []
        speeds = []
        distances = []
        start = time.time()

        ep_begin = start_episode + 1
        ep_end = start_episode + n_episodes + 1
        pbar = tqdm(range(ep_begin, ep_end), desc="Training", unit="ep")
        first_reset = seed

        try:
            for ep in pbar:
                if time_limit and (time.time() - start) >= time_limit:
                    pbar.write(f"Time limit reached after {ep - 1} episodes.")
                    break

                state = self._reset_env(env, seed=first_reset)
                first_reset = None
                ep_reward = 0.0
                ep_losses = []
                ep_speeds = []
                ep_step_count = 0
                done = False

                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, info = self._step_env(env, action)
                    ep_step_count += 1
                    # obs[0] = speed/50 in BeamNG; skip for discrete envs (int state)
                    if isinstance(state, np.ndarray) and len(state) > 0:
                        ep_speeds.append(float(state[0]) * 50.0)
                    loss = agent.update(state, action, reward, next_state, done)
                    if loss is not None:
                        ep_losses.append(loss)
                    state = next_state
                    ep_reward += reward

                agent.decay_epsilon()
                if hasattr(agent, "episode"):
                    agent.episode = ep
                rewards.append(ep_reward)
                ep_steps = self._episode_steps(info, ep_step_count)
                steps.append(ep_steps)
                speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
                ep_distance = info.get("waypoint_idx", 0) if isinstance(info, dict) else 0
                distances.append(int(ep_distance))

                # Update progress bar
                avg_r = np.mean(rewards[-20:])
                avg_loss = np.mean(ep_losses) if ep_losses else float("nan")
                pbar.set_postfix(
                    reward=f"{ep_reward:.1f}",
                    avg20=f"{avg_r:.1f}",
                    eps=f"{agent.epsilon:.3f}",
                    loss=f"{avg_loss:.4f}",
                )

                if save_path and ep % save_every == 0:
                    agent.save(save_path)
                    if plot_path:
                        self._save_plot(
                            rewards, steps, "Training", plot_path, ep, speeds, distances
                        )

        except KeyboardInterrupt:
            pbar.write("Training interrupted by user.")

        finally:
            pbar.close()
            if save_path:
                agent.save(save_path)
            if plot_path:
                self._save_plot(
                    rewards, steps, "Training", plot_path, len(rewards), speeds, distances
                )

        elapsed = time.time() - start
        print(f"\nTraining complete in {elapsed:.1f}s ({len(rewards)} episodes).")
        return {"rewards": rewards, "steps": steps, "speeds": speeds, "distances": distances}

    def evaluate(
        self,
        agent: BaseAgent,
        env,
        n_episodes: int = 10,
        seed: int | None = None,
    ) -> dict:
        """Run the agent in pure exploitation mode (epsilon=0).

        When ``seed`` is given, the global RNGs and the environment's first
        reset are seeded so the evaluation is reproducible.
        """
        if seed is not None:
            set_global_seed(seed)

        old_eps = agent.epsilon
        agent.epsilon = 0.0

        rewards = []
        steps = []

        pbar = tqdm(range(1, n_episodes + 1), desc="Evaluating", unit="ep")
        first_reset = seed

        try:
            for _ep in pbar:
                state = self._reset_env(env, seed=first_reset)
                first_reset = None
                ep_reward = 0.0
                ep_step_count = 0
                done = False

                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, info = self._step_env(env, action)
                    ep_step_count += 1
                    state = next_state
                    ep_reward += reward

                ep_steps = self._episode_steps(info, ep_step_count)
                rewards.append(ep_reward)
                steps.append(ep_steps)

                avg = np.mean(rewards)
                pbar.set_postfix(reward=f"{ep_reward:.1f}", avg=f"{avg:.1f}")

        except KeyboardInterrupt:
            pbar.write("Evaluation interrupted by user.")

        finally:
            pbar.close()
            agent.epsilon = old_eps

        avg = np.mean(rewards)
        print(f"\nEvaluation: avg reward = {avg:.2f} over {n_episodes} episodes.")
        return {"rewards": rewards, "steps": steps, "avg_reward": float(avg)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_env(env, seed: int | None = None):
        """Normalise Gymnasium reset (obs, info) vs old-style (obs).

        When ``seed`` is provided, it is forwarded to the environment's reset
        (and to the action space) so the episode starts deterministically.
        """
        if seed is not None:
            seed_action_space(env, seed)
            try:
                result = env.reset(seed=seed)
            except TypeError:
                result = env.reset()
        else:
            result = env.reset()
        if isinstance(result, tuple):
            return result[0]
        return result

    @staticmethod
    def _episode_steps(info, fallback_count: int) -> int:
        """Return the episode step count.

        Prefers an explicit ``steps`` entry in the env's info dict (e.g. BeamNG)
        and falls back to the number of loop iterations, so step tracking works
        for environments like Taxi-v3 that do not report steps in info.
        """
        if isinstance(info, dict) and "steps" in info:
            return int(info["steps"])
        return fallback_count

    @staticmethod
    def _step_env(env, action):
        """Normalise 5-tuple to 4-tuple (obs, reward, done, info)."""
        result = env.step(action)
        if len(result) == 5:
            obs, rew, terminated, truncated, info = result
            return obs, rew, terminated or truncated, info
        return result

    @staticmethod
    def _save_plot(rewards, steps, title, path, episode=None, speeds=None, distances=None):
        if not rewards:
            return
        window = min(20, len(rewards))
        eps = range(1, len(rewards) + 1)

        # Only show speed/distance panels when real (non-zero) data was collected
        has_speed = bool(speeds) and any(s != 0.0 for s in speeds)
        has_dist = bool(distances) and any(d != 0 for d in distances)
        has_extra = has_speed or has_dist

        if has_extra:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            ax_reward = axes[0, 0]
            ax_steps = axes[0, 1]
            ax_speed = axes[1, 0]
            ax_dist = axes[1, 1]
        else:
            fig, (ax_reward, ax_steps) = plt.subplots(2, 1, figsize=(10, 8))

        subtitle = f" - Episode {episode}" if episode else ""
        fig.suptitle(f"{title}{subtitle}")

        # -- Reward
        ax_reward.plot(eps, rewards, alpha=0.3, label="Reward")
        if len(rewards) >= window:
            roll = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax_reward.plot(range(window, len(rewards) + 1), roll, label=f"Rolling avg ({window})")
        ax_reward.set_ylabel("Total Reward")
        ax_reward.legend()
        ax_reward.grid(True)

        # -- Steps
        ax_steps.plot(eps, steps, alpha=0.3, label="Steps", color="orange")
        if len(steps) >= window:
            roll = np.convolve(steps, np.ones(window) / window, mode="valid")
            ax_steps.plot(
                range(window, len(steps) + 1),
                roll,
                label=f"Rolling avg ({window})",
                color="darkorange",
            )
        ax_steps.set_ylabel("Steps per Episode")
        ax_steps.legend()
        ax_steps.grid(True)

        if has_extra:
            # -- Avg speed (m/s)
            if has_speed:
                ax_speed.plot(eps, speeds, alpha=0.3, label="Avg Speed", color="green")
                if len(speeds) >= window:
                    roll = np.convolve(speeds, np.ones(window) / window, mode="valid")
                    ax_speed.plot(
                        range(window, len(speeds) + 1),
                        roll,
                        label=f"Rolling avg ({window})",
                        color="darkgreen",
                    )
                ax_speed.set_xlabel("Episode")
                ax_speed.set_ylabel("Avg Speed (m/s)")
                ax_speed.legend()
                ax_speed.grid(True)

            # -- Distance / waypoints reached
            if has_dist:
                ax_dist.plot(eps, distances, alpha=0.3, label="Waypoints Reached", color="purple")
                if len(distances) >= window:
                    roll = np.convolve(distances, np.ones(window) / window, mode="valid")
                    ax_dist.plot(
                        range(window, len(distances) + 1),
                        roll,
                        label=f"Rolling avg ({window})",
                        color="indigo",
                    )
                ax_dist.set_xlabel("Episode")
                ax_dist.set_ylabel("Waypoints Reached")
                ax_dist.legend()
                ax_dist.grid(True)

            ax_reward.set_xlabel("")
            ax_steps.set_xlabel("")
        else:
            ax_steps.set_xlabel("Episode")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
