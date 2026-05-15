"""Grid search over hyperparameters, launching one BeamNG instance per configuration."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np

from core.runner import PipelineRunner


def _expand_grid(param_grid: dict) -> list[dict]:
    """Return all combinations of values in param_grid."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]


def _run_single(
    idx: int,
    params: dict,
    algo_cls,
    make_env,
    n_states: int,
    n_actions: int,
    n_episodes: int,
    port: int,
    output_dir: str,
) -> dict:
    """Train one agent configuration in its own BeamNG instance."""
    label = f"run_{idx:02d}"
    print(f"[GridSearch] {label} starting on port {port} | params={params}")

    env = make_env(port)
    agent_params = {"n_states": n_states, "n_actions": n_actions, **params}
    agent = algo_cls(**agent_params)

    save_path = os.path.join(output_dir, f"{label}.pth")
    runner = PipelineRunner()

    try:
        history = runner.train(
            agent,
            env,
            n_episodes=n_episodes,
            save_path=save_path,
        )
        rewards = history["rewards"]
        result = {
            "run": label,
            "port": port,
            "params": params,
            "avg_reward": float(np.mean(rewards)) if rewards else float("-inf"),
            "best_reward": float(np.max(rewards)) if rewards else float("-inf"),
            "n_episodes": len(rewards),
            "save_path": save_path,
        }
    except Exception as exc:
        print(f"[GridSearch] {label} FAILED: {exc}")
        result = {
            "run": label,
            "port": port,
            "params": params,
            "avg_reward": float("-inf"),
            "best_reward": float("-inf"),
            "n_episodes": 0,
            "error": str(exc),
        }
    finally:
        env.close()

    print(f"[GridSearch] {label} done — avg_reward={result['avg_reward']:.2f}")
    return result


class GridSearch:
    """Run a hyperparameter grid search over multiple parallel BeamNG instances."""

    def run(
        self,
        algo_cls,
        make_env,
        param_grid: dict,
        n_states: int,
        n_actions: int,
        n_episodes: int = 100,
        base_port: int = 25252,
        max_parallel: int = 2,
        output_dir: str = "outputs/grid_search",
    ) -> list[dict]:
        """
        Args:
            algo_cls:      Agent class (e.g. DQNAgent).
            make_env:      Callable(port) -> env.  Must accept a port kwarg so
                           each instance binds to a different BeamNG process.
            param_grid:    Dict mapping hyperparameter names to lists of values.
                           e.g. {"lr": [1e-3, 1e-4], "gamma": [0.99, 0.95]}
            n_states:      State-space size passed to the agent constructor.
            n_actions:     Action-space size passed to the agent constructor.
            n_episodes:    Episodes per configuration.
            base_port:     First BeamNG port; each subsequent run gets +1.
            max_parallel:  Max concurrent BeamNG instances.
            output_dir:    Directory for per-run checkpoints and the summary.

        Returns:
            List of result dicts sorted by avg_reward descending.
        """
        configs = _expand_grid(param_grid)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[GridSearch] {len(configs)} configurations, max {max_parallel} in parallel.\n")

        results = []
        futures = {}

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            for idx, params in enumerate(configs):
                port = base_port + idx
                future = pool.submit(
                    _run_single,
                    idx,
                    params,
                    algo_cls,
                    make_env,
                    n_states,
                    n_actions,
                    n_episodes,
                    port,
                    output_dir,
                )
                futures[future] = idx

            for future in as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: r["avg_reward"], reverse=True)

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

        self._print_summary(results)
        return results

    @staticmethod
    def _print_summary(results: list[dict]):
        print("\n" + "=" * 60)
        print("  Grid Search Results (ranked by avg reward)")
        print("=" * 60)
        for rank, r in enumerate(results, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
            status = f"avg={r['avg_reward']:.2f}  best={r['best_reward']:.2f}"
            if "error" in r:
                status = f"ERROR: {r['error']}"
            print(f"  #{rank:02d}  {status}  |  {params_str}")
        print("=" * 60)
        if results:
            best = results[0]
            print(f"\n  Best config: {best['params']}")
            print(f"  Saved to:    {best.get('save_path', 'N/A')}\n")
