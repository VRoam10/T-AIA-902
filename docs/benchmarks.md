# Benchmarks — metrics reference & data schema

This document defines every metric the benchmark suite reports and the JSON
schema consumed by the web dashboard.

## Principles

- **Seeds** — each run seeds the global RNGs (`random`, `numpy`, `torch`) and
  the environment, so a given seed is exactly reproducible.
- **Multi-seed** — each configuration is replayed across several seeds and
  scalar metrics are aggregated (`mean`, `std`, `ci95`, `min`, `max`, `n`).
- **Greedy evaluation** — after training, the policy is evaluated at
  `epsilon=0` over N episodes to measure its true performance.

## Metric definitions

| Metric | Meaning |
|--------|---------|
| `convergence_episode` | First episode where the rolling-average reward (window `window`) reaches `threshold`. |
| `improvement_rate` | Linear-regression slope of the rolling-average reward (reward/episode). |
| `final_avg_reward` | Mean reward over the last 20 % of training episodes. |
| `mean_reward` / `best_reward` / `worst_reward` | Reward distribution over training. |
| `eval_mean_reward` | Mean reward of the greedy policy over the evaluation episodes. |
| `eval_std_reward` | Std of the greedy-policy reward. |
| `eval_success_rate` | Fraction of evaluation episodes with reward ≥ `success_threshold`. |
| `eval_mean_steps` | Mean steps per evaluation episode. |
| `training_time_s` | Wall-clock training time. |

Aggregated stats (`mean ± std`, `ci95`) are computed across seeds. `ci95` is the
95 % confidence-interval half-width: `1.96 · std / sqrt(n)`.

## Output files (per run)

| File | Content |
|------|---------|
| `report.md` | Human-readable report with tables, plots and a reproducibility section. |
| `metadata.json` | git commit, python/numpy/torch versions, device, seeds, algo, env. |
| `summary.json` | Aggregated metrics (multi-seed). |
| `results_full.json` | Everything, including per-seed reward curves. |
| `metrics.csv` | One row per seed (or per run). |
| `summary.csv` | One row per aggregated metric / variant. |
| `leaderboard.csv` | Ranked configurations (grid-search only). |
| `*.png` | Reward band, bars, heatmap. |

`outputs/benchmarks/index.json` is a manifest of all runs (built by
`core.benchmark_index`), used by the dashboard.

## Result shapes (index → dashboard)

The dashboard discriminates four shapes from the result JSON:

- **single** — flat scalar metrics + `rewards` (one convergence run).
- **multi-seed** — `aggregate` + `per_seed` (reward band across seeds).
- **comparison** — `variants` mapping a label to its `aggregate`, `converged_rate`
  and `mean_rolling`/`std_rolling` curves.
- **grid-search** — `entries` (ranked) + `param_names` + `best`.

## Continuous agents & BeamNG

The suite is algorithm-agnostic: DDPG and TD3 (continuous control) go through the
exact same path as the tabular/deep discrete agents. Greedy evaluation sets the
agent's `epsilon` to 0, which disables exploration noise (OU noise for DDPG,
Gaussian for TD3), so the policy is evaluated deterministically.

This continuous path is covered by `tests/test_benchmarks_continuous.py`, which
runs DDPG and TD3 through `convergence`, multi-seed aggregation and `comparison`
on a stub environment with the same contract as BeamNG (N-dim continuous
observation, 2-dim continuous action in [-1, 1]) — the simulator itself cannot
run in CI.

**Cost note (BeamNG):** each seed launches the environment twice (once for
training, once for the greedy evaluation), so an N-seed run starts the simulator
2·N times. On BeamNG, prefer a small number of seeds (e.g. `seeds=[0,1,2]`) and a
modest `eval_episodes`.

## Running

From the app (`python main.py` → OpenTUI → *Run a benchmark*) or programmatically:

```python
import benchmarks
from core.registry import registry

env = registry.get_environment("taxi")
algo = registry.get_algorithm("q_learning")
bench = registry.get_benchmark("convergence")["class"]()

multi = bench.run_multi(
    algo["class"], env["factory"],
    {
        "agent_params": algo["default_config"],
        "env_metadata": env["metadata"],
        "max_episodes": 1200,
        "seeds": [0, 1, 2],
        "eval_episodes": 100,
    },
)
bench.export_multi(multi, "q_learning", "taxi")
```

Then `python scripts/sync_web_data.py` to publish to the dashboard.
