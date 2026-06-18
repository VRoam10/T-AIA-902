import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from core.runner import PipelineRunner
from core.stats import aggregate, numeric_keys, summary_line


class BaseBenchmark(ABC):
    """Abstract base class for benchmarks that evaluate agent/env combos."""

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def run(self, agent_cls: type, env_factory: Callable, config: dict) -> dict:
        """Run the benchmark. Returns a results dict."""
        ...

    def evaluate_policy(
        self,
        agent,
        env_factory: Callable,
        n_episodes: int = 100,
        seed: int | None = None,
        success_threshold: float = 0.0,
    ) -> dict:
        """Evaluate a trained agent greedily (epsilon=0) on a fresh environment.

        Measures the true performance of the learned policy, as opposed to the
        noisy training rewards that mix in exploration.

        Args:
            agent: A trained agent exposing the BaseAgent interface.
            env_factory: Callable returning a fresh environment instance.
            n_episodes: Number of evaluation episodes to run.
            seed: Optional seed for reproducible evaluation.
            success_threshold: An episode counts as a success when its total
                reward is greater than or equal to this value.

        Returns:
            Dict with eval_episodes, eval_mean_reward, eval_std_reward,
            eval_mean_steps and eval_success_rate.
        """
        env = env_factory()
        runner = PipelineRunner()
        result = runner.evaluate(agent, env, n_episodes=n_episodes, seed=seed)
        env.close()

        rewards = np.array(result["rewards"], dtype=float)
        steps = np.array(result["steps"], dtype=float)
        has_data = rewards.size > 0

        success_rate = float(np.mean(rewards >= success_threshold)) if has_data else 0.0

        return {
            "eval_episodes": n_episodes,
            "eval_mean_reward": round(float(np.mean(rewards)), 4) if has_data else 0.0,
            "eval_std_reward": round(float(np.std(rewards)), 4) if has_data else 0.0,
            "eval_mean_steps": round(float(np.mean(steps)), 2) if steps.size else 0.0,
            "eval_success_rate": round(success_rate, 4),
        }

    def run_multi(self, agent_cls: type, env_factory: Callable, config: dict) -> dict:
        """Run the benchmark once per seed and aggregate the scalar metrics.

        For each seed in ``config["seeds"]`` (default ``[0, 1, 2, 3, 4]``) the
        benchmark's ``run`` is executed with that seed injected into the config.
        Scalar metrics are aggregated into mean / std / ci95; the per-seed
        results and a representative run (closest to the mean performance) are
        kept for reporting and plotting.

        Args:
            agent_cls: The agent class to benchmark.
            env_factory: Callable returning a fresh environment.
            config: Benchmark config; ``seeds`` selects the runs.

        Returns:
            Dict with seeds, n_seeds, per_seed, aggregate and representative.
        """
        seeds = config.get("seeds", [0, 1, 2, 3, 4])
        per_seed = []
        for seed in seeds:
            print(f"\n[{self.name}] seed {seed} ({len(per_seed) + 1}/{len(seeds)})")
            seed_config = {**config, "seed": seed}
            per_seed.append(self.run(agent_cls, env_factory, seed_config))

        scalar_keys = numeric_keys(per_seed[0])
        agg = aggregate(per_seed, scalar_keys)

        return {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "aggregate": agg,
            "per_seed": per_seed,
            "representative": self._representative(per_seed, agg),
        }

    @staticmethod
    def _representative(per_seed: list[dict], agg: dict) -> dict:
        """Pick the run whose headline metric is closest to the seed mean."""
        for key in ("eval_mean_reward", "final_avg_reward", "mean_reward"):
            if key in agg:
                target = agg[key]["mean"]
                return min(per_seed, key=lambda run: abs(run.get(key, 0.0) - target))
        return per_seed[0]

    def export_multi(
        self,
        multi_results: dict,
        algo_name: str,
        env_name: str,
        output_dir: str = "outputs/benchmarks",
    ):
        """Export aggregated multi-seed results: JSON, markdown, and plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(
            output_dir, f"{self.name}_{algo_name}_{env_name}_multiseed_{timestamp}"
        )
        os.makedirs(run_dir, exist_ok=True)

        self._save_json(multi_results, os.path.join(run_dir, "results_full.json"))
        self._save_json(
            {
                "seeds": multi_results["seeds"],
                "n_seeds": multi_results["n_seeds"],
                "aggregate": multi_results["aggregate"],
            },
            os.path.join(run_dir, "summary.json"),
        )

        self._save_plots(multi_results["representative"], run_dir, algo_name, env_name)
        self._save_seed_band_plot(multi_results, run_dir, algo_name, env_name)
        self._save_multi_markdown(
            multi_results, os.path.join(run_dir, "report.md"), algo_name, env_name
        )

        print(f"\n[Benchmark] Multi-seed reports saved to: {run_dir}/")
        return run_dir

    def _save_seed_band_plot(
        self, multi_results: dict, run_dir: str, algo_name: str, env_name: str
    ):
        """Plot the per-episode reward mean with a ±std band across seeds."""
        curves = [run.get("rewards") for run in multi_results["per_seed"] if run.get("rewards")]
        if not curves:
            return
        length = min(len(c) for c in curves)
        if length == 0:
            return
        matrix = np.array([c[:length] for c in curves], dtype=float)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        episodes = range(1, length + 1)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(episodes, mean, color="navy", linewidth=1.5, label="Mean reward")
        ax.fill_between(
            episodes, mean - std, mean + std, color="navy", alpha=0.2, label="±1 std (seeds)"
        )
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"Reward across {multi_results['n_seeds']} seeds — {algo_name} / {env_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "seed_band.png"), dpi=150)
        plt.close()

    def _save_multi_markdown(self, multi_results: dict, path: str, algo_name: str, env_name: str):
        """Write the aggregated multi-seed markdown report."""
        agg = multi_results["aggregate"]
        seeds = multi_results["seeds"]

        lines = [
            f"# {self.name} — Multi-seed Report",
            "",
            f"**Algorithm:** `{algo_name}`  ",
            f"**Environment:** `{env_name}`  ",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Seeds:** {seeds} ({multi_results['n_seeds']} runs)  ",
            "",
            "## Aggregated Metrics",
            "",
            "| Metric | Mean ± Std | CI95 | Min | Max |",
            "|--------|-----------|------|-----|-----|",
        ]
        for key, stat in agg.items():
            lines.append(
                f"| {key} | {summary_line(stat)} | ±{stat['ci95']} "
                f"| {stat['min']} | {stat['max']} |"
            )

        lines += ["", "## Plots", "", "![Reward band](seed_band.png)", ""]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def report(self, results: dict) -> str:
        """Format results as a human-readable string."""
        lines = [f"=== {self.name} ==="]
        for k, v in results.items():
            if isinstance(v, list | np.ndarray):
                continue
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def export(
        self, results: dict, algo_name: str, env_name: str, output_dir: str = "outputs/benchmarks"
    ):
        """Export full benchmark results: JSON, Markdown report, and plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"{self.name}_{algo_name}_{env_name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # JSON
        json_path = os.path.join(run_dir, "results.json")
        self._save_json(results, json_path)

        # Plots
        self._save_plots(results, run_dir, algo_name, env_name)

        # Markdown
        md_path = os.path.join(run_dir, "report.md")
        self._save_markdown(results, md_path, algo_name, env_name)

        print(f"\n[Benchmark] Reports saved to: {run_dir}/")
        return run_dir

    def _save_json(self, results: dict, path: str):
        """Save results as JSON (convert numpy types)."""

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        with open(path, "w", encoding="utf-8") as f:
            json.dump(_convert(results), f, indent=2, ensure_ascii=False)

    def _save_plots(self, results: dict, run_dir: str, algo_name: str, env_name: str):
        """Override in subclasses for custom plots. Default: reward curve."""
        rewards = results.get("rewards", [])
        if not rewards:
            return

        eps = range(1, len(rewards) + 1)
        window = min(50, len(rewards))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eps, rewards, alpha=0.3, label="Reward")
        if len(rewards) >= window:
            roll = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(range(window, len(rewards) + 1), roll, label=f"Rolling avg ({window})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(f"{self.name} — {algo_name} / {env_name}")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "reward_curve.png"), dpi=150)
        plt.close()

    def _save_markdown(self, results: dict, path: str, algo_name: str, env_name: str):
        """Override in subclasses for custom markdown. Default: summary table."""
        lines = [
            f"# Benchmark: {self.name}",
            f"**Algorithm:** {algo_name}  ",
            f"**Environment:** {env_name}  ",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            "",
            "## Results",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for k, v in results.items():
            if isinstance(v, list | np.ndarray):
                continue
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")

        lines.append("")
        lines.append("## Plots")
        lines.append("![Reward Curve](reward_curve.png)")
        lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
