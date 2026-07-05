"""Algorithm-agnostic, multi-seed side-by-side comparison benchmark."""

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from core.base_benchmark import BaseBenchmark
from core.registry import registry
from core.runner import PipelineRunner
from core.stats import aggregate, summary_line


class ComparisonBenchmark(BaseBenchmark):
    """Train several variants over multiple seeds and compare them side by side.

    Each entry in ``config["variants"]`` is a dict with:
      - ``name``: display label.
      - ``algo``: a registered algorithm name (resolved via the registry), OR
        ``agent_cls`` to pass an explicit agent class.
      - ``agent_params``: kwargs merged over the algorithm's default config.

    Every variant is trained once per seed in ``config["seeds"]`` (default
    ``[0]``) and its scalar metrics are aggregated into mean / std / ci95, so
    the comparison reflects statistical robustness rather than a single run.

    Example config
    --------------
    {
        "max_episodes": 1500,
        "threshold": 7.0,
        "window": 100,
        "seeds": [0, 1, 2],
        "env_metadata": {"n_states": 500, "n_actions": 6},
        "variants": [
            {"name": "Q-Learning", "algo": "q_learning"},
            {"name": "DQN", "algo": "dqn"},
            {"name": "DQN+PER", "algo": "dqn_per"},
        ],
    }
    """

    name = "comparison"
    description = "Algorithm-agnostic, multi-seed side-by-side comparison"

    def run(self, agent_cls, env_factory, config: dict) -> dict:
        variants = config.get("variants", [])
        if not variants:
            raise ValueError("config must contain at least one entry in 'variants'")

        max_episodes = config.get("max_episodes", 1500)
        threshold = config.get("threshold", 7.0)
        window = config.get("window", 100)
        metadata = config.get("env_metadata", {})
        seeds = config.get("seeds", [0])
        eval_episodes = config.get("eval_episodes", 100)
        success_threshold = config.get("success_threshold", 0.0)

        all_results = {}

        for variant in variants:
            label, cls, params = self._resolve_variant(variant, agent_cls, metadata)
            print(f"\n[ComparisonBenchmark] Variant '{label}' over {len(seeds)} seed(s)")

            per_seed = []
            rolling_curves = []
            for seed in seeds:
                metrics, rolling = self._run_single(
                    cls,
                    params,
                    env_factory,
                    max_episodes=max_episodes,
                    threshold=threshold,
                    window=window,
                    seed=seed,
                    eval_episodes=eval_episodes,
                    success_threshold=success_threshold,
                )
                per_seed.append(metrics)
                rolling_curves.append(rolling)

            agg = aggregate(per_seed)
            mean_rolling, std_rolling = self._aggregate_curves(rolling_curves)
            converged_rate = float(np.mean([m["converged"] for m in per_seed]))

            all_results[label] = {
                "aggregate": agg,
                "converged_rate": round(converged_rate, 4),
                "n_seeds": len(seeds),
                "mean_rolling": mean_rolling,
                "std_rolling": std_rolling,
            }

            eval_stat = agg.get("eval_mean_reward", {})
            print(
                f"  -> {label}: eval reward {eval_stat.get('mean', 0)} "
                f"+/- {eval_stat.get('std', 0)} | converged {converged_rate:.0%}"
            )

        return {
            "variants": all_results,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "max_episodes": max_episodes,
            "threshold": threshold,
            "window": window,
        }

    @staticmethod
    def _resolve_variant(variant: dict, default_cls, metadata: dict):
        """Resolve a variant entry to (label, agent_cls, agent_params).

        Supports registry algorithm names via ``algo`` (defaults merged with
        ``agent_params``) and explicit classes via ``agent_cls``.
        """
        label = variant["name"]
        if "algo" in variant:
            info = registry.get_algorithm(variant["algo"])
            cls = info["class"]
            params = {**info["default_config"], **variant.get("agent_params", {})}
        else:
            cls = variant.get("agent_cls", default_cls)
            params = dict(variant.get("agent_params", {}))
        BaseBenchmark._finalize_agent_params(cls, params, metadata)
        return label, cls, params

    def _run_single(
        self,
        cls,
        params: dict,
        env_factory,
        *,
        max_episodes: int,
        threshold: float,
        window: int,
        seed: int,
        eval_episodes: int,
        success_threshold: float,
    ):
        """Train one variant for one seed and return (metrics, rolling_avgs)."""
        env = env_factory()
        agent = cls(**params)
        runner = PipelineRunner()

        start = time.time()
        history = runner.train(agent, env, n_episodes=max_episodes, seed=seed)
        elapsed = time.time() - start
        env.close()

        eval_metrics = self.evaluate_policy(
            agent,
            env_factory,
            n_episodes=eval_episodes,
            seed=seed + 10_000,
            success_threshold=success_threshold,
        )

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
        metrics = {
            "convergence_episode": convergence_ep,
            "converged": convergence_ep is not None,
            "training_time_s": round(elapsed, 2),
            "final_avg_reward": round(float(np.mean(tail)), 4),
            "mean_reward": round(float(np.mean(rewards_arr)), 4),
            "best_reward": round(float(np.max(rewards_arr)), 4),
            **eval_metrics,
        }
        return metrics, rolling

    @staticmethod
    def _aggregate_curves(curves: list[list[float]]):
        """Return per-episode mean and std of the rolling-average curves."""
        curves = [c for c in curves if c]
        if not curves:
            return [], []
        length = min(len(c) for c in curves)
        if length == 0:
            return [], []
        matrix = np.array([c[:length] for c in curves], dtype=float)
        return matrix.mean(axis=0).tolist(), matrix.std(axis=0).tolist()

    def _save_csv(self, results: dict, run_dir: str):
        """Write summary.csv with one aggregated row per variant."""
        rows = []
        for label, data in results.get("variants", {}).items():
            row = {"variant": label, "converged_rate": data.get("converged_rate", 0.0)}
            for metric, stat in data["aggregate"].items():
                row[f"{metric}_mean"] = stat["mean"]
                row[f"{metric}_std"] = stat["std"]
            rows.append(row)
        self._write_csv_rows(os.path.join(run_dir, "summary.csv"), rows)

    def _save_plots(self, results: dict, run_dir: str, algo_name: str, env_name: str):
        variants = results.get("variants", {})
        if not variants:
            return

        window = results.get("window", 100)
        threshold = results.get("threshold", 7.0)
        n_seeds = results.get("n_seeds", 1)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (label, data) in enumerate(variants.items()):
            mean_rolling = data.get("mean_rolling", [])
            std_rolling = data.get("std_rolling", [])
            color = colors[i % len(colors)]
            if mean_rolling:
                ra_eps = range(window, window + len(mean_rolling))
                ax.plot(ra_eps, mean_rolling, label=label, color=color, linewidth=2)
                if std_rolling:
                    mean_arr = np.array(mean_rolling)
                    std_arr = np.array(std_rolling)
                    ax.fill_between(
                        ra_eps, mean_arr - std_arr, mean_arr + std_arr, color=color, alpha=0.15
                    )

        ax.axhline(
            threshold, color="black", linestyle=":", linewidth=1, label=f"Threshold ({threshold})"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Rolling avg reward (window={window})")
        ax.set_title(f"Variant comparison ({n_seeds} seeds) — {env_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "comparison_curves.png"), dpi=150)
        plt.close()

        self._save_bar_plots(results, run_dir, env_name)

    def _save_bar_plots(self, results: dict, run_dir: str, env_name: str):
        """Bar charts of convergence speed and greedy eval reward (mean ±std)."""
        variants = results.get("variants", {})
        max_episodes = results.get("max_episodes", 0)
        n_seeds = results.get("n_seeds", 1)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        labels = list(variants.keys())
        bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

        def stat(label, key, field, default=0.0):
            return variants[label]["aggregate"].get(key, {}).get(field, default)

        conv_means = [stat(lbl, "convergence_episode", "mean", max_episodes) for lbl in labels]
        conv_errs = [stat(lbl, "convergence_episode", "std") for lbl in labels]
        eval_means = [stat(lbl, "eval_mean_reward", "mean") for lbl in labels]
        eval_errs = [stat(lbl, "eval_mean_reward", "std") for lbl in labels]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(labels, conv_means, yerr=conv_errs, capsize=4, color=bar_colors, alpha=0.8)
        axes[0].set_ylabel("Episode")
        axes[0].set_title("Convergence speed (lower = better)")
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].bar(labels, eval_means, yerr=eval_errs, capsize=4, color=bar_colors, alpha=0.8)
        axes[1].set_ylabel("Greedy eval reward")
        axes[1].set_title("Policy performance (higher = better)")
        axes[1].grid(True, axis="y", alpha=0.3)

        plt.suptitle(f"Variant comparison ({n_seeds} seeds) — {env_name}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "comparison_bars.png"), dpi=150)
        plt.close()

    def _save_markdown(self, results: dict, path: str, algo_name: str, env_name: str):
        variants = results.get("variants", {})
        threshold = results.get("threshold", 7.0)
        window = results.get("window", 100)
        seeds = results.get("seeds", [])

        lines = [
            "# Variant Comparison Report",
            "",
            f"**Environment:** `{env_name}`  ",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Seeds:** {seeds} ({results.get('n_seeds', 1)} runs)  ",
            f"**Threshold:** {threshold}  ",
            f"**Rolling window:** {window}  ",
            "",
            "## Results (mean ± std across seeds)",
            "",
            "| Variant | Converged | Conv. episode | Eval reward | Eval success | Eval steps | Time (s) |",
            "|---------|-----------|--------------|-------------|--------------|-----------|----------|",
        ]

        for label, data in variants.items():

            def cell(key, data=data):
                stat = data["aggregate"].get(key)
                return summary_line(stat) if stat else "—"

            conv_rate = f"{data.get('converged_rate', 0):.0%}"
            lines.append(
                f"| {label} | {conv_rate} | {cell('convergence_episode')} "
                f"| {cell('eval_mean_reward')} | {cell('eval_success_rate')} "
                f"| {cell('eval_mean_steps')} | {cell('training_time_s')} |"
            )

        lines += [
            "",
            "## Plots",
            "",
            "### Reward curves (mean ± std band)",
            "![Curves](comparison_curves.png)",
            "",
            "### Summary bars",
            "![Bars](comparison_bars.png)",
            "",
        ]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
