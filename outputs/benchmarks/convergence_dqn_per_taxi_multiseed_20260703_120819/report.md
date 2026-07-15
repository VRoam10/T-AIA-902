# convergence — Multi-seed Report

**Algorithm:** `dqn_per`  
**Environment:** `taxi`  
**Date:** 2026-07-03 12:08:22  
**Seeds:** [0, 1] (2 runs)  

## Aggregated Metrics

| Metric | Mean ± Std | CI95 | Min | Max |
|--------|-----------|------|-----|-----|
| total_episodes | 500.0 ± 0.0 | ±0.0 | 500.0 | 500.0 |
| training_time_s | 1250.76 ± 90.21 | ±125.0247 | 1160.55 | 1340.97 |
| threshold | 7.0 ± 0.0 | ±0.0 | 7.0 | 7.0 |
| window | 100.0 ± 0.0 | ±0.0 | 100.0 | 100.0 |
| final_avg_reward | -160.79 ± 40.5 | ±56.1301 | -201.29 | -120.29 |
| final_std_reward | 87.1204 ± 22.3495 | ±30.9748 | 64.7709 | 109.4699 |
| best_reward | 15.0 ± 0.0 | ±0.0 | 15.0 | 15.0 |
| best_episode | 321.5 ± 159.5 | ±221.0557 | 162.0 | 481.0 |
| worst_reward | -888.5 ± 58.5 | ±81.0769 | -947.0 | -830.0 |
| worst_episode | 1.0 ± 0.0 | ±0.0 | 1.0 | 1.0 |
| mean_reward | -213.257 ± 28.527 | ±39.5364 | -241.784 | -184.73 |
| median_reward | -222.5 ± 4.5 | ±6.2367 | -227.0 | -218.0 |
| q25_reward | -240.5 ± 4.5 | ±6.2367 | -245.0 | -236.0 |
| q75_reward | -140.0 ± 78.0 | ±108.1025 | -218.0 | -62.0 |
| improvement_rate | 0.1796 ± 0.1149 | ±0.1592 | 0.0647 | 0.2945 |
| mean_steps | 172.445 ± 20.575 | ±28.5155 | 151.87 | 193.02 |
| max_steps | 200.0 ± 0.0 | ±0.0 | 200.0 | 200.0 |
| min_steps | 6.0 ± 0.0 | ±0.0 | 6.0 | 6.0 |
| eval_episodes | 50.0 ± 0.0 | ±0.0 | 50.0 | 50.0 |
| eval_mean_reward | -139.13 ± 35.33 | ±48.9649 | -174.46 | -103.8 |
| eval_std_reward | 86.7073 ± 17.5414 | ±24.3111 | 69.1659 | 104.2487 |
| eval_mean_steps | 145.22 ± 31.76 | ±44.0171 | 113.46 | 176.98 |
| eval_success_rate | 0.26 ± 0.14 | ±0.194 | 0.12 | 0.4 |

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
| algo | dqn_per |
| env | taxi |
| seeds | [0, 1] |
| n_seeds | 2 |
