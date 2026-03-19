import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from core.base_benchmark import BaseBenchmark
from core.runner import PipelineRunner


class ConvergenceBenchmark(BaseBenchmark):
    """Full convergence analysis: speed, stability, reward distribution, loss tracking."""

    name = "convergence"
    description = "Comprehensive convergence analysis with detailed metrics and reports"

    def run(self, agent_cls, env_factory, config: dict) -> dict:
        env = env_factory()
        metadata = config.get("env_metadata", {})

        agent_params = dict(config.get("agent_params", {}))
        agent_params["n_states"] = metadata.get("n_states", 5)
        if "n_actions" not in agent_params:
            agent_params["n_actions"] = metadata.get("n_actions", 6)
        agent = agent_cls(**agent_params)

        max_episodes = config.get("max_episodes", 2000)
        threshold = config.get("threshold", 7.0)
        window = config.get("window", 100)

        # --- Training with detailed tracking ---
        runner = PipelineRunner()
        start_time = time.time()
        history = runner.train(agent, env, n_episodes=max_episodes)
        training_time = time.time() - start_time

        rewards = history["rewards"]
        steps = history["steps"]

        # --- Convergence detection ---
        convergence_ep = None
        rolling_avgs = []
        for i in range(window, len(rewards)):
            avg = float(np.mean(rewards[i - window : i]))
            rolling_avgs.append(avg)
            if convergence_ep is None and avg >= threshold:
                convergence_ep = i

        # --- Stability: std of reward in last 20% of episodes ---
        tail_size = max(1, len(rewards) // 5)
        tail_rewards = rewards[-tail_size:]
        stability_std = float(np.std(tail_rewards))
        stability_mean = float(np.mean(tail_rewards))

        # --- Reward distribution stats ---
        rewards_arr = np.array(rewards)
        quartiles = np.percentile(rewards_arr, [25, 50, 75]).tolist()

        # --- Per-episode step stats ---
        steps_arr = np.array(steps)

        # --- Peak performance ---
        best_reward = float(np.max(rewards_arr))
        best_episode = int(np.argmax(rewards_arr)) + 1
        worst_reward = float(np.min(rewards_arr))
        worst_episode = int(np.argmin(rewards_arr)) + 1

        # --- Improvement rate (linear regression slope on rolling avg) ---
        improvement_rate = 0.0
        if len(rolling_avgs) > 1:
            x = np.arange(len(rolling_avgs))
            coeffs = np.polyfit(x, rolling_avgs, 1)
            improvement_rate = float(coeffs[0])

        env.close()

        return {
            # Summary
            "converged": convergence_ep is not None,
            "convergence_episode": convergence_ep,
            "total_episodes": len(rewards),
            "training_time_s": round(training_time, 2),
            "threshold": threshold,
            "window": window,
            # Reward stats
            "final_avg_reward": round(stability_mean, 4),
            "final_std_reward": round(stability_std, 4),
            "best_reward": round(best_reward, 4),
            "best_episode": best_episode,
            "worst_reward": round(worst_reward, 4),
            "worst_episode": worst_episode,
            "mean_reward": round(float(np.mean(rewards_arr)), 4),
            "median_reward": round(float(np.median(rewards_arr)), 4),
            "q25_reward": round(quartiles[0], 4),
            "q75_reward": round(quartiles[2], 4),
            "improvement_rate": round(improvement_rate, 6),
            # Steps stats
            "mean_steps": round(float(np.mean(steps_arr)), 2),
            "max_steps": int(np.max(steps_arr)) if len(steps_arr) > 0 else 0,
            "min_steps": int(np.min(steps_arr)) if len(steps_arr) > 0 else 0,
            # Agent config
            "agent_config": agent.get_config(),
            # Raw data (for plots, not printed)
            "rewards": rewards,
            "steps": steps,
            "rolling_avgs": rolling_avgs,
        }

    def _save_plots(self, results: dict, run_dir: str, algo_name: str, env_name: str):
        """Generate comprehensive plot suite."""
        rewards = results.get("rewards", [])
        steps = results.get("steps", [])
        rolling_avgs = results.get("rolling_avgs", [])
        window = results.get("window", 100)
        threshold = results.get("threshold", 7.0)
        convergence_ep = results.get("convergence_episode")

        if not rewards:
            return

        eps = range(1, len(rewards) + 1)

        # --- 1. Full training overview (4 subplots) ---
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Convergence Benchmark — {algo_name} / {env_name}", fontsize=14)

        # 1a. Reward curve + rolling avg + threshold
        ax = axes[0, 0]
        ax.plot(eps, rewards, alpha=0.2, color="steelblue", label="Reward")
        if rolling_avgs:
            ra_eps = range(window, window + len(rolling_avgs))
            ax.plot(ra_eps, rolling_avgs, color="navy", linewidth=1.5, label=f"Rolling avg ({window})")
        ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.7, label=f"Threshold ({threshold})")
        if convergence_ep:
            ax.axvline(x=convergence_ep, color="green", linestyle="--", alpha=0.7, label=f"Converged (ep {convergence_ep})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Reward Curve")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1b. Steps per episode
        ax = axes[0, 1]
        ax.plot(eps, steps, alpha=0.3, color="orange")
        step_window = min(50, len(steps))
        if len(steps) >= step_window:
            roll = np.convolve(steps, np.ones(step_window) / step_window, mode="valid")
            ax.plot(range(step_window, len(steps) + 1), roll, color="darkorange", linewidth=1.5, label=f"Rolling avg ({step_window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Steps")
        ax.set_title("Steps per Episode")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1c. Reward distribution histogram
        ax = axes[1, 0]
        ax.hist(rewards, bins=50, color="steelblue", alpha=0.7, edgecolor="navy")
        ax.axvline(x=np.mean(rewards), color="red", linestyle="--", label=f"Mean ({np.mean(rewards):.1f})")
        ax.axvline(x=np.median(rewards), color="green", linestyle="--", label=f"Median ({np.median(rewards):.1f})")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Frequency")
        ax.set_title("Reward Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 1d. Reward boxplot by segment (split training into 5 phases)
        ax = axes[1, 1]
        n_segments = 5
        seg_size = max(1, len(rewards) // n_segments)
        segments = []
        labels = []
        for i in range(n_segments):
            start = i * seg_size
            end = start + seg_size if i < n_segments - 1 else len(rewards)
            segments.append(rewards[start:end])
            labels.append(f"Ep {start+1}-{end}")
        ax.boxplot(segments, labels=labels)
        ax.set_ylabel("Reward")
        ax.set_title("Reward by Training Phase")
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "overview.png"), dpi=150)
        plt.close()

        # --- 2. Cumulative reward ---
        fig, ax = plt.subplots(figsize=(10, 5))
        cum_rewards = np.cumsum(rewards)
        ax.plot(eps, cum_rewards, color="steelblue")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(f"Cumulative Reward — {algo_name} / {env_name}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "cumulative_reward.png"), dpi=150)
        plt.close()

        # --- 3. Reward heatmap (episodes grouped by blocks) ---
        block_size = max(1, len(rewards) // 20)
        if block_size > 0 and len(rewards) >= block_size:
            n_blocks = len(rewards) // block_size
            blocked = np.array(rewards[:n_blocks * block_size]).reshape(n_blocks, block_size)
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(blocked.T, aspect="auto", cmap="RdYlGn", interpolation="nearest")
            ax.set_xlabel("Block")
            ax.set_ylabel("Episode within block")
            ax.set_title(f"Reward Heatmap ({block_size} ep/block) — {algo_name} / {env_name}")
            plt.colorbar(im, ax=ax, label="Reward")
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "reward_heatmap.png"), dpi=150)
            plt.close()

    def _save_markdown(self, results: dict, path: str, algo_name: str, env_name: str):
        """Generate detailed markdown report."""
        convergence_ep = results.get("convergence_episode")
        converged = results.get("converged", False)

        lines = [
            f"# Convergence Benchmark Report",
            "",
            f"**Algorithm:** `{algo_name}`  ",
            f"**Environment:** `{env_name}`  ",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Training time:** {results.get('training_time_s', 0)}s  ",
            "",
            "---",
            "",
            "## Convergence",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Converged | {'Yes' if converged else 'No'} |",
            f"| Convergence episode | {convergence_ep if convergence_ep else 'N/A'} |",
            f"| Threshold | {results.get('threshold')} |",
            f"| Window size | {results.get('window')} |",
            f"| Total episodes | {results.get('total_episodes')} |",
            f"| Improvement rate | {results.get('improvement_rate', 0):.6f} reward/ep |",
            "",
            "## Reward Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean | {results.get('mean_reward')} |",
            f"| Median | {results.get('median_reward')} |",
            f"| Q25 | {results.get('q25_reward')} |",
            f"| Q75 | {results.get('q75_reward')} |",
            f"| Best | {results.get('best_reward')} (ep {results.get('best_episode')}) |",
            f"| Worst | {results.get('worst_reward')} (ep {results.get('worst_episode')}) |",
            f"| Final avg (last 20%) | {results.get('final_avg_reward')} |",
            f"| Final std (last 20%) | {results.get('final_std_reward')} |",
            "",
            "## Steps Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean steps/ep | {results.get('mean_steps')} |",
            f"| Max steps | {results.get('max_steps')} |",
            f"| Min steps | {results.get('min_steps')} |",
            "",
            "## Agent Configuration",
            "",
            "```json",
        ]

        import json
        lines.append(json.dumps(results.get("agent_config", {}), indent=2))
        lines.extend([
            "```",
            "",
            "## Plots",
            "",
            "### Training Overview",
            "![Overview](overview.png)",
            "",
            "### Cumulative Reward",
            "![Cumulative](cumulative_reward.png)",
            "",
            "### Reward Heatmap",
            "![Heatmap](reward_heatmap.png)",
            "",
        ])

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
