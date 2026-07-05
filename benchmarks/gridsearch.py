"""Hyperparameter grid-search benchmark.

Sweeps a cartesian product of hyperparameters, evaluates each configuration
over multiple seeds with a greedy evaluation, and ranks them. Produces a
leaderboard, a 2D heatmap (when exactly two parameters are swept), and JSON.
"""

import itertools
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from core.base_benchmark import BaseBenchmark
from core.runner import PipelineRunner
from core.stats import aggregate, summary_line


class GridSearchBenchmark(BaseBenchmark):
    """Grid-search over hyperparameters, ranked by greedy eval reward.

    Config keys
    -----------
    - ``param_grid``: dict mapping a hyperparameter name to a list of values.
    - ``agent_params``: base params merged under each grid combination.
    - ``seeds``: seeds evaluated per combination (default ``[0, 1, 2]``).
    - ``max_episodes``, ``eval_episodes``, ``success_threshold``.

    Example
    -------
    {
        "param_grid": {
            "learning_rate": [0.1, 0.5, 0.85],
            "discount_factor": [0.9, 0.99],
        },
        "seeds": [0, 1, 2],
        "max_episodes": 1000,
    }
    """

    name = "gridsearch"
    description = "Hyperparameter grid-search ranked by greedy eval reward"

    def run(self, agent_cls, env_factory, config: dict) -> dict:
        param_grid = config.get("param_grid", {})
        if not param_grid:
            raise ValueError("config must contain a non-empty 'param_grid'")

        metadata = config.get("env_metadata", {})
        base_params = dict(config.get("agent_params", {}))
        self._finalize_agent_params(agent_cls, base_params, metadata)

        seeds = config.get("seeds", [0, 1, 2])
        max_episodes = config.get("max_episodes", 1000)
        eval_episodes = config.get("eval_episodes", 100)
        success_threshold = config.get("success_threshold", 0.0)

        param_names = list(param_grid.keys())
        value_lists = [param_grid[name] for name in param_names]
        combinations = list(itertools.product(*value_lists))
        print(
            f"[GridSearchBenchmark] {len(combinations)} combination(s) "
            f"x {len(seeds)} seed(s) = {len(combinations) * len(seeds)} runs"
        )

        entries = []
        for combo in combinations:
            combo_params = dict(zip(param_names, combo, strict=True))
            params = {**base_params, **combo_params}

            per_seed = []
            for seed in seeds:
                per_seed.append(
                    self._run_single(
                        agent_cls,
                        params,
                        env_factory,
                        max_episodes=max_episodes,
                        seed=seed,
                        eval_episodes=eval_episodes,
                        success_threshold=success_threshold,
                    )
                )

            agg = aggregate(per_seed)
            entries.append(
                {
                    "params": combo_params,
                    "aggregate": agg,
                    "eval_mean_reward": agg.get("eval_mean_reward", {}).get("mean", 0.0),
                }
            )
            print(f"  {combo_params} -> eval {entries[-1]['eval_mean_reward']}")

        entries.sort(key=lambda e: e["eval_mean_reward"], reverse=True)

        return {
            "param_names": param_names,
            "param_grid": param_grid,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "n_combinations": len(combinations),
            "entries": entries,
            "best": entries[0] if entries else None,
        }

    def _run_single(
        self,
        agent_cls,
        params: dict,
        env_factory,
        *,
        max_episodes: int,
        seed: int,
        eval_episodes: int,
        success_threshold: float,
    ) -> dict:
        """Train one configuration for one seed and return its scalar metrics."""
        env = env_factory()
        agent = agent_cls(**params)
        runner = PipelineRunner()

        start = time.time()
        runner.train(agent, env, n_episodes=max_episodes, seed=seed)
        elapsed = time.time() - start
        env.close()

        eval_metrics = self.evaluate_policy(
            agent,
            env_factory,
            n_episodes=eval_episodes,
            seed=seed + 10_000,
            success_threshold=success_threshold,
        )
        return {"training_time_s": round(elapsed, 2), **eval_metrics}

    def report(self, results: dict) -> str:
        best = results.get("best")
        lines = [f"=== {self.name} ==="]
        lines.append(f"  combinations: {results.get('n_combinations')}")
        lines.append(f"  seeds: {results.get('n_seeds')}")
        if best:
            lines.append(f"  best params: {best['params']}")
            lines.append(f"  best eval reward: {best['eval_mean_reward']}")
        return "\n".join(lines)

    def _save_csv(self, results: dict, run_dir: str):
        """Write leaderboard.csv with one ranked row per configuration."""
        rows = []
        for rank, entry in enumerate(results.get("entries", []), start=1):
            row = {"rank": rank, **entry["params"]}
            for metric, stat in entry["aggregate"].items():
                row[f"{metric}_mean"] = stat["mean"]
                row[f"{metric}_std"] = stat["std"]
            rows.append(row)
        self._write_csv_rows(os.path.join(run_dir, "leaderboard.csv"), rows)

    def _save_plots(self, results: dict, run_dir: str, algo_name: str, env_name: str):
        entries = results.get("entries", [])
        param_names = results.get("param_names", [])
        if not entries:
            return

        if len(param_names) == 2:
            self._save_heatmap(results, run_dir, algo_name, env_name)

        top = entries[: min(10, len(entries))]
        labels = [self._combo_label(e["params"]) for e in top]
        means = [e["aggregate"].get("eval_mean_reward", {}).get("mean", 0.0) for e in top]
        errs = [e["aggregate"].get("eval_mean_reward", {}).get("std", 0.0) for e in top]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(range(len(top)), means, xerr=errs, capsize=4, color="steelblue", alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Greedy eval reward (mean ±std)")
        ax.set_title(f"Top configurations — {algo_name} / {env_name}")
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "leaderboard.png"), dpi=150)
        plt.close()

    def _save_heatmap(self, results: dict, run_dir: str, algo_name: str, env_name: str):
        """Render a 2D heatmap of eval reward over the two swept parameters."""
        p1, p2 = results["param_names"]
        grid = results["param_grid"]
        v1, v2 = grid[p1], grid[p2]
        lookup = {
            (e["params"][p1], e["params"][p2]): e["eval_mean_reward"] for e in results["entries"]
        }
        matrix = np.array([[lookup.get((a, b), np.nan) for b in v2] for a in v1], dtype=float)

        fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(v2), 1.5 + 1.0 * len(v1)))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(v2)))
        ax.set_xticklabels(v2)
        ax.set_yticks(range(len(v1)))
        ax.set_yticklabels(v1)
        ax.set_xlabel(p2)
        ax.set_ylabel(p1)
        ax.set_title(f"Eval reward heatmap — {algo_name} / {env_name}")
        for i in range(len(v1)):
            for j in range(len(v2)):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.1f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
        plt.colorbar(im, ax=ax, label="Eval reward")
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "heatmap.png"), dpi=150)
        plt.close()

    @staticmethod
    def _combo_label(params: dict) -> str:
        """Compact label for a parameter combination."""
        return ", ".join(f"{k}={v}" for k, v in params.items())

    def _save_markdown(self, results: dict, path: str, algo_name: str, env_name: str):
        entries = results.get("entries", [])
        param_names = results.get("param_names", [])

        lines = [
            "# Grid-Search Report",
            "",
            f"**Algorithm:** `{algo_name}`  ",
            f"**Environment:** `{env_name}`  ",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"**Seeds:** {results.get('seeds')} ({results.get('n_seeds')} runs)  ",
            f"**Combinations:** {results.get('n_combinations')}  ",
            "",
            "## Leaderboard (ranked by greedy eval reward)",
            "",
            "| Rank | " + " | ".join(param_names) + " | Eval reward | Success | Time (s) |",
            "|------|"
            + "|".join(["----"] * len(param_names))
            + "|------------|---------|----------|",
        ]

        for rank, entry in enumerate(entries, start=1):
            agg = entry["aggregate"]
            param_cells = " | ".join(str(entry["params"][name]) for name in param_names)
            eval_cell = summary_line(agg["eval_mean_reward"]) if "eval_mean_reward" in agg else "—"
            succ_cell = (
                summary_line(agg["eval_success_rate"]) if "eval_success_rate" in agg else "—"
            )
            time_cell = summary_line(agg["training_time_s"]) if "training_time_s" in agg else "—"
            lines.append(f"| {rank} | {param_cells} | {eval_cell} | {succ_cell} | {time_cell} |")

        lines += ["", "## Plots", ""]
        if len(param_names) == 2:
            lines += ["### Heatmap", "![Heatmap](heatmap.png)", ""]
        lines += ["### Top configurations", "![Leaderboard](leaderboard.png)", ""]

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
