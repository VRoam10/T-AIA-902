import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class BaseBenchmark(ABC):
    """Abstract base class for benchmarks that evaluate agent/env combos."""

    name: str = "unnamed"
    description: str = ""

    @abstractmethod
    def run(self, agent_cls: type, env_factory: Callable, config: dict) -> dict:
        """Run the benchmark. Returns a results dict."""
        ...

    def report(self, results: dict) -> str:
        """Format results as a human-readable string."""
        lines = [f"=== {self.name} ==="]
        for k, v in results.items():
            if isinstance(v, (list, np.ndarray)):
                continue
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def export(self, results: dict, algo_name: str, env_name: str, output_dir: str = "outputs/benchmarks"):
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
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
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
            if isinstance(v, (list, np.ndarray)):
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
