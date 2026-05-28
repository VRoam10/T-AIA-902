"""Side-by-side comparison benchmark for multiple DQN variants."""

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from core.base_benchmark import BaseBenchmark
from core.runner import PipelineRunner


class ComparisonBenchmark(BaseBenchmark):
    """Train several agent variants and compare convergence side by side.

    Each entry in config["variants"] is a dict with:
      - "name": display label
      - "agent_cls": the agent class to instantiate
      - "agent_params": kwargs forwarded to agent_cls.__init__

    Example config
    --------------
    {
        "max_episodes": 1500,
        "threshold": 7.0,
        "window": 100,
        "env_metadata": {"n_states": 500, "n_actions": 6},
        "variants": [
            {"name": "DQN", "agent_cls": DQNAgent, "agent_params": {...}},
            {"name": "DQN+PER", "agent_cls": DQNAgent, "agent_params": {..., "use_per": True}},
        ],
    }
    """

    name = "comparison"
    description = "Side-by-side convergence comparison for multiple agent variants"

    def run(self, agent_cls, env_factory, config: dict) -> dict:
        variants = config.get("variants", [])
        if not variants:
            raise ValueError("config must contain at least one entry in 'variants'")

        max_episodes = config.get("max_episodes", 1500)
        threshold = config.get("threshold", 7.0)
        window = config.get("window", 100)
        metadata = config.get("env_metadata", {})

        all_results = {}

        for variant in variants:
            label = variant["name"]
            cls = variant.get("agent_cls", agent_cls)
            params = dict(variant.get("agent_params", {}))
            params.setdefault("n_states", metadata.get("n_states", 5))
            params.setdefault("n_actions", metadata.get("n_actions", 6))

            print(f"\n[ComparisonBenchmark] Running variant: {label}")
            env = env_factory()
            agent = cls(**params)

            runner = PipelineRunner()
            start = time.time()
            history = runner.train(agent, env, n_episodes=max_episodes)
            elapsed = time.time() - start
            env.close()

            rewards = history["rewards"]
            rewards_arr = np.array(rewards)

            rolling = []
            convergence_ep = None
            for i in range(window, len(rewards)):
                avg = float(np.mean(rewards[i - window : i]))
                rolling.append(avg)
                if convergence_ep is None and avg >= threshold:
                    convergence_ep = i

            tail = rewards[-max(1, len(rewards) // 5) :]
            all_results[label] = {
                "rewards": rewards,
                "rolling_avgs": rolling,
                "convergence_episode": convergence_ep,
                "converged": convergence_ep is not None,
                "training_time_s": round(elapsed, 2),
                "final_avg_reward": round(float(np.mean(tail)), 4),
                "final_std_reward": round(float(np.std(tail)), 4),
                "mean_reward": round(float(np.mean(rewards_arr)), 4),
                "best_reward": round(float(np.max(rewards_arr)), 4),
            }

            status = f"converged at ep {convergence_ep}" if convergence_ep else "did not converge"
            print(f"  → {label}: {status}, final avg = {all_results[label]['final_avg_reward']:.2f}")

        return {
            "variants": all_results,
            "max_episodes": max_episodes,
            "threshold": threshold,
            "window": window,
        }

    def _save_plots(self, results: dict, run_dir: str, algo_name: str, env_name: str):
        variants = results.get("variants", {})
        if not variants:
            return

        window = results.get("window", 100)
        threshold = results.get("threshold", 7.0)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        # -- 1. Reward curves (rolling avg overlay) --
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (label, data) in enumerate(variants.items()):
            rewards = data.get("rewards", [])
            rolling = data.get("rolling_avgs", [])
            color = colors[i % len(colors)]
            ax.plot(rewards, alpha=0.12, color=color)
            if rolling:
                ra_eps = range(window, window + len(rolling))
                ax.plot(ra_eps, rolling, label=label, color=color, linewidth=2)
            conv = data.get("convergence_episode")
            if conv:
                ax.axvline(conv, color=color, linestyle="--", alpha=0.6)

        ax.axhline(threshold, color="black", linestyle=":", linewidth=1, label=f"Threshold ({threshold})")
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Rolling avg reward (window={window})")
        ax.set_title(f"DQN Variant Comparison — {env_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "comparison_curves.png"), dpi=150)
        plt.close()

        # -- 2. Bar chart: convergence episode (lower = faster) --
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        labels = list(variants.keys())
        conv_eps = [variants[l]["convergence_episode"] or results["max_episodes"] for l in labels]
        final_avgs = [variants[l]["final_avg_reward"] for l in labels]

        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

        axes[0].bar(labels, conv_eps, color=bar_colors, alpha=0.8)
        axes[0].set_ylabel("Episode")
        axes[0].set_title("Convergence speed (lower = better)")
        axes[0].grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(conv_eps):
            axes[0].text(i, v + 5, str(v), ha="center", va="bottom", fontsize=9)

        axes[1].bar(labels, final_avgs, color=bar_colors, alpha=0.8)
        axes[1].set_ylabel("Avg reward (last 20%)")
        axes[1].set_title("Final performance (higher = better)")
        axes[1].grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(final_avgs):
            axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        plt.suptitle(f"DQN Variant Comparison — {env_name}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "comparison_bars.png"), dpi=150)
        plt.close()

    def _save_markdown(self, results: dict, path: str, algo_name: str, env_name: str):
        variants = results.get("variants", {})
        threshold = results.get("threshold", 7.0)
        window = results.get("window", 100)

        lines = [
            "# DQN Variant Comparison Report",
            "",
            f"**Environment:** `{env_name}`  ",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Threshold:** {threshold}  ",
            f"**Rolling window:** {window}  ",
            "",
            "## Results",
            "",
            "| Variant | Converged | Conv. episode | Final avg reward | Final std | Time (s) |",
            "|---------|-----------|--------------|-----------------|-----------|----------|",
        ]

        for label, data in variants.items():
            conv = data.get("convergence_episode", "—")
            lines.append(
                f"| {label} | {'✓' if data['converged'] else '✗'} | {conv} "
                f"| {data['final_avg_reward']:.4f} | {data['final_std_reward']:.4f} "
                f"| {data['training_time_s']} |"
            )

        lines += [
            "",
            "## Plots",
            "",
            "### Reward curves",
            "![Curves](comparison_curves.png)",
            "",
            "### Summary bars",
            "![Bars](comparison_bars.png)",
            "",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
