# convergence — Multi-seed Report

**Algorithm:** `dqn`  
**Environment:** `taxi`  
**Date:** 2026-07-03 11:26:31  
**Seeds:** [0, 1] (2 runs)  

## Aggregated Metrics

| Metric | Mean ± Std | CI95 | Min | Max |
|--------|-----------|------|-----|-----|
| total_episodes | 500.0 ± 0.0 | ±0.0 | 500.0 | 500.0 |
| training_time_s | 1298.065 ± 214.385 | ±297.1225 | 1083.68 | 1512.45 |
| threshold | 7.0 ± 0.0 | ±0.0 | 7.0 | 7.0 |
| window | 100.0 ± 0.0 | ±0.0 | 100.0 | 100.0 |
| final_avg_reward | -181.495 ± 38.635 | ±53.5454 | -220.13 | -142.86 |
| final_std_reward | 119.6881 ± 10.172 | ±14.0977 | 109.5161 | 129.8602 |
| best_reward | 14.5 ± 0.5 | ±0.693 | 14.0 | 15.0 |
| best_episode | 339.5 ± 31.5 | ±43.6568 | 308.0 | 371.0 |
| worst_reward | -1028.0 ± 135.0 | ±187.1005 | -1163.0 | -893.0 |
| worst_episode | 439.5 ± 28.5 | ±39.499 | 411.0 | 468.0 |
| mean_reward | -224.168 ± 13.678 | ±18.9567 | -237.846 | -210.49 |
| median_reward | -224.75 ± 2.25 | ±3.1183 | -227.0 | -222.5 |
| q25_reward | -236.0 ± 0.0 | ±0.0 | -236.0 | -236.0 |
| q75_reward | -213.5 ± 4.5 | ±6.2367 | -218.0 | -209.0 |
| improvement_rate | 0.1892 ± 0.1214 | ±0.1682 | 0.0679 | 0.3106 |
| mean_steps | 182.21 ± 11.03 | ±15.2868 | 171.18 | 193.24 |
| max_steps | 200.0 ± 0.0 | ±0.0 | 200.0 | 200.0 |
| min_steps | 6.5 ± 0.5 | ±0.693 | 6.0 | 7.0 |
| eval_episodes | 50.0 ± 0.0 | ±0.0 | 50.0 | 50.0 |
| eval_mean_reward | -175.14 ± 24.86 | ±34.4542 | -200.0 | -150.28 |
| eval_std_reward | 44.2411 ± 44.2411 | ±61.315 | 0.0 | 88.4821 |
| eval_mean_steps | 177.66 ± 22.34 | ±30.9617 | 155.32 | 200.0 |
| eval_success_rate | 0.1 ± 0.1 | ±0.1386 | 0.0 | 0.2 |

## Plots

![Reward band](seed_band.png)

## Reproducibility

| Field | Value |
|-------|-------|
| git_commit | e278fd4 |
| python | 3.11.9 |
| platform | Windows-10-10.0.26200-SP0 |
| numpy | 1.26.4 |
| torch | 2.12.0+cpu |
| device | cpu |
| benchmark | convergence |
| algo | dqn |
| env | taxi |
| seeds | [0, 1] |
| n_seeds | 2 |
