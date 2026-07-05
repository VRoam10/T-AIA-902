# Five Algorithms on Taxi-v3: Q-Learning, DQN, DQN+PER, TD3, DDPG

Benchmarks every Taxi-compatible algorithm in the registry (`q_learning`,
`dqn`, `dqn_per`, `td3`, `ddpg`) on Gymnasium's `Taxi-v3` — 500 discrete
states, 6 actions — using the same convergence/comparison harness as the rest
of the pipeline.

## Headline results

| | Eval reward | Detail |
|---|---|---|
| **Q-Learning** | **+8.83** | 1200 ep, 1 seed, 90% success |
| **DQN** | **-175.1** | 500 ep, 2 seeds, 10% success |
| **DQN+PER** | **-139.1** | 500 ep, 2 seeds, 26% success |
| **TD3** | **-200.0** | 250 ep, 2 seeds, 0% success, zero variance |
| **DDPG** | **-200.0** | 80 ep, 1 seed, 0% success, zero variance |
| **Params vs Q-table (3,000)** | 1x → 130x | q_learning → dqn (38x) → td3 (81x) → ddpg (130x) |

### Reading these numbers: Taxi-v3's reward scale

Every step costs **-1**, a successful drop-off pays **+20**, an illegal
pickup/drop-off costs **-10**, and an episode is cut off at **200 steps**. So a
positive average (Q-Learning's **+8.83**) means the taxi reaches the passenger
and destination in efficient trips (~12 steps). A large negative average
(DQN's **-175**) is not "175 units of bad" in the abstract — arithmetically
it's close to `-200` (the full 200-step timeout penalty) with a few illegal-move
penalties mixed in: **the episode almost never finishes the trip before the timeout.**
TD3/DDPG's exact **-200.0 with zero variance** is a step further: not just a
timeout, but the *same* timeout on every single episode and seed.

## Methodology

All runs use `core.pipeline_actions.run_benchmark` against the `convergence`
and `comparison` benchmarks already in the pipeline. Each run trains with
epsilon-greedy exploration, then evaluates the frozen greedy policy (epsilon=0)
to measure what the agent actually learned. Convergence is flagged when the
100-episode rolling-average reward first reaches `7.0`, the conventional
"solved" bar for Taxi-v3.

| Run | Episodes | Seeds | Eval episodes | Purpose |
|---|---|---|---|---|
| Q-Learning convergence | 1200 | 1 | 30 | Baseline diagnostics |
| DQN convergence | 500 | 2 | 50 | Per-algorithm diagnostics |
| DQN+PER convergence | 500 | 2 | 50 | Per-algorithm diagnostics |
| TD3 convergence | 250 | 2 | 30 | Per-algorithm diagnostics |
| DDPG convergence | 80 | 1 | 30 | Per-algorithm diagnostics (reduced budget) |
| 3-way comparison | 250 | 2 | 30 | Equal-budget head-to-head |

Episode budgets differ across sections on purpose: the per-algorithm runs use
a larger budget for richer diagnostics, while the head-to-head comparison uses
a smaller, identical budget for all three so the ranking is apples-to-apples.
Wall-clock cost made anything larger impractical on this CPU-only machine —
DDPG in particular got a much smaller budget because its default
hyperparameters make it roughly 6x slower per episode than DQN.

`td3` and `ddpg` are continuous-control (actor-critic) algorithms built for
BeamNG's steering/throttle output, not discrete environments. The pipeline
bridges them onto Taxi by having the actor score all 6 actions and taking the
argmax — the same trick used for their BeamNG-vs-Taxi action space mismatch.
Running them through the benchmark suite surfaced a real gap in the harness
itself; see [process notes](#process-notes).

## Q-Learning — baseline, 1200 episodes, seed 0

Tabular Q-learning solves Taxi convincingly: eval reward **+8.83 ± 2.65** over
30 greedy episodes, **90% success rate**, averaging **12.2 steps** per trip.
Training 1200 episodes takes **0.65 seconds**.

![Fig. 1 — Q-Learning training overview](images/fig1-qlearning-overview.png)

**Fig. 1.** Reward curve (top-left) crosses into positive territory by
~episode 400; the phase boxplot (bottom-right) shows the spread tightening
and its median climbing across the 5 training phases.

![Fig. 2 — Q-Learning reward heatmap](images/fig2-qlearning-heatmap.png)

**Fig. 2.** Each column is a ~60-episode block; red (low-reward) cells
dominate the left, green cells the right — a direct visual signature of
convergence.

## DQN — Double + Dueling, 500 episodes, 2 seeds

With the registry's default hyperparameters, DQN does not solve Taxi within
this budget: eval reward **-175.1 ± 24.9**, success rate **10%**, episodes
averaging **178 of the 200-step cap**.

![Fig. 3 — DQN training overview](images/fig3-dqn-overview.png)

**Fig. 3.** Reward curve sits near its minimum for the entire 500 episodes;
the phase boxplot is flat — phase 5's median is no better than phase 1's.

![Fig. 4 — DQN reward across 2 seeds](images/fig4-dqn-seedband.png)

**Fig. 4.** Both seeds track the same flat, negative trajectory with a narrow
±1 std band — a consistent failure mode, not seed noise.

## DQN + PER — Prioritized replay, 500 episodes, 2 seeds

Prioritized experience replay recovers some ground: eval reward
**-139.1 ± 35.3**, success rate **26%** (vs DQN's 10%), at ~20% more
wall-clock cost for the sum-tree bookkeeping.

![Fig. 5 — DQN+PER training overview](images/fig5-dqnper-overview.png)

**Fig. 5.** Same axes as Fig. 3 — the reward floor is less severe and the
phase boxplot shows mild upward drift in the later phases.

![Fig. 6 — DQN+PER reward across 2 seeds](images/fig6-dqnper-seedband.png)

**Fig. 6.** Wider variance between seeds than Fig. 4, but a visibly less
negative mean band overall.

## TD3 — Twin Delayed DDPG, 250 episodes, 2 seeds

TD3 doesn't just fail to solve Taxi — it converges to a single, perfectly
reproducible dead end: eval reward exactly **-200.0 ± 0.0** across all 30 eval
episodes, both seeds, **0% success**, every episode running the full 200-step
cap with zero illegal-move penalties. The greedy policy isn't struggling —
it's stuck cycling through legal-but-pointless moves with mathematical certainty.

![Fig. 7 — TD3 training overview](images/fig7-td3-overview.png)

**Fig. 7.** Every panel is flat: the reward curve is a perfectly straight line
at -200, and the phase boxplot shows literally zero height (zero variance) in
every one of the 5 phases.

![Fig. 8 — TD3 reward across 2 seeds](images/fig8-td3-seedband.png)

**Fig. 8.** No visible band at all — both seeds produce the exact same -200
trajectory from episode 1, so there's nothing for a ±1 std shade to show.

## DDPG — OU noise, 80 episodes, 1 seed, reduced budget

DDPG's default config (`hidden=256`, `updates_per_step=4`) makes it the
slowest agent in the registry by a wide margin — **~15.8s/episode** here, so
this run uses a smaller 80-episode, single-seed budget purely to keep it
bounded. During training (with Ornstein-Uhlenbeck exploration noise active)
reward is noisy and often worse than -200 (best **-328**, worst **-893**,
mean **-602**) — but strip the noise away for the greedy eval and it lands on
the *exact same* dead end as TD3: **-200.0 ± 0.0**, 0% success, every episode
timing out.

![Fig. 9 — DDPG training overview](images/fig9-ddpg-overview.png)

**Fig. 9.** Unlike Fig. 7, the reward curve here does move — but downward and
noisily, driven by OU noise pushing the actor into illegal moves. The phase
boxplot shows wide spread, not convergence.

![Fig. 10 — DDPG reward heatmap](images/fig10-ddpg-heatmap.png)

**Fig. 10.** Almost entirely red/orange (low reward) throughout — no green
blocks ever appear, unlike Fig. 2's clear left-to-right improvement.

## Continuous-control algorithms are overkill for a 500-state, 6-action problem

TD3 and DDPG exist to handle continuous action spaces (BeamNG's steering and
throttle) — twin critics, delayed policy updates, target policy smoothing, and
Ornstein-Uhlenbeck noise are all machinery built to stabilize learning a
*continuous* control signal. None of that machinery has anything to do with
choosing among 6 discrete moves in a 500-state world small enough to fit in a
lookup table:

| Algorithm | Trainable params | vs Q-table | Network modules | s / episode |
|---|---|---|---|---|
| q_learning | 3,000 | 1x | 1 lookup table | 0.0005 |
| dqn | 114,567 | 38x | online + target net | ~2.6 |
| td3 | 244,488 | 81x | actor+target, twin-critic+target | ~5.7 |
| ddpg | 391,431 | 130x | actor+target, critic+target | ~15.8 |

Param counts are actor+critic only (excluding the target-network copies, which
roughly double the live tensors in memory). Q-learning's "table" is 500x6
floats, no gradients, no forward/backward pass at all.

## Head-to-head — equal 250-episode budget, 2 seeds

| Variant | Eval reward | Success | Eval steps | Train time (s) |
|---|---|---|---|---|
| q_learning | -71.23 ± 10.37 | 56.7% | 84.2 | 0.34 |
| dqn | -200.00 ± 0.00 | 0.0% | 200.0 | 162.9 |
| dqn_per | -189.52 ± 10.48 | 5.0% | 190.6 | 198.2 |

TD3/DDPG are deliberately left out of this specific chart: at this budget TD3
alone would add ~48 minutes and DDPG's default config makes an equal-episode
run impractically slow (their own dedicated sections above already make the
comparison point clearly).

![Fig. 11 — comparison reward curves](images/fig11-comparison-curves.png)

**Fig. 11.** Q-Learning's curve climbs steadily from episode ~30 onward; both
DQN curves stay pinned near the floor for the entire 250-episode budget.

![Fig. 12 — comparison summary bars](images/fig12-comparison-bars.png)

**Fig. 12.** Left: neither DQN variant converges within 250 episodes (bar sits
at the budget ceiling). Right: the eval-reward gap is large enough that the
error bars don't come close to overlapping.

## Observations from the plots

**Q-Learning's convergence is visible directly in the curve and the heatmap.**
Fig. 1's reward curve is the textbook convergence shape: noisy near-random
rewards for the first ~100 episodes, then a steady climb into positive
territory. Fig. 2 shows the same thing from a different angle — reward per
block goes from mostly red to mostly green, with no red re-appearing once it
clears.

**DQN's reward curve never leaves the floor — and the seed band shows it
isn't seed noise.** Fig. 3's curve sits near its minimum for all 500 episodes;
Fig. 4 rules out a single unlucky seed. The likely cause: `dqn` ships with
`epsilon_decay=0.95`, tuned for BeamNG's world of a few hundred long episodes.
On Taxi's short (≤200-step) episodes that decay collapses epsilon to its floor
(0.05) in **~60 episodes** — exploration stops before the agent has sampled
enough of the 500-state space. Q-Learning's own default
(`epsilon_decay=0.9975`, ~20x slower to decay) is the main variable that
differs between a curve that climbs (Fig. 1) and one that doesn't (Fig. 3).

**PER's plots show a real but partial recovery.** Fig. 5's phase boxplot shows
what Fig. 3's doesn't: a visible upward drift in later phases. Fig. 6's mean
line sits clearly above Fig. 4's for most of training, though its band is
wider — PER's priority sampling makes individual runs less predictable even as
it lifts the average.

**The head-to-head curves and bars agree with the per-algorithm plots.**
Fig. 11 reproduces the same shapes at a shorter, equal budget: one curve
(q_learning) climbs, two (dqn, dqn_per) stay flat. Fig. 12's right panel turns
that into non-overlapping error bars — the ranking isn't a close call at this budget.

**TD3's flat line is a stronger failure signature than DQN's noisy one.**
Fig. 3's reward curve is deeply negative but still has visible noise episode
to episode. Fig. 7 has none — every phase box has zero height, and Fig. 8
shows no shaded band because both seeds land on the identical trajectory. A
neural net that behaves identically across random seeds down to the exact
reward isn't "slow to learn," it's found one specific fixed point (a cycle of
legal moves that never reaches the destination) and the argmax-over-actor-scores
bridge never lets it escape.

**DDPG's plots show noise without progress, then the same dead end as TD3.**
Fig. 9's reward curve is the only one of the five that trends *down* during
training — visible directly as the OU-noise-driven dip toward -900 around
episode 15. Fig. 10 confirms there's no recovery: no green block ever appears,
unlike Fig. 2's clean left-to-right shift. Once the exploration noise is
switched off for the greedy eval, DDPG's actor converges to the exact same
-200.0 flatline as Fig. 7 — two different algorithms, two different network
sizes, the same degenerate fixed point.

## Process notes

Not plot-driven — tooling and environment findings from getting these runs working at all.

**Real harness bug found and fixed: continuous-control agents on discrete
envs.** Running `td3`/`ddpg` through `benchmarks.convergence` crashed
immediately: `TypeError: unhashable type: 'numpy.ndarray'` inside Taxi's own
`step()`. Unlike the DQN one-hot bug below, this one was **not** already fixed
on `main`. Root cause: `core.pipeline_actions.build_agent` (used for training)
injects `state_type="discrete"` so TD3/DDPG switch into one-hot-state /
argmax-over-scores mode — but `benchmarks/convergence.py`, `comparison.py` and
`gridsearch.py` each construct the agent directly from `agent_cls(**params)`
without ever setting it, so the agent silently defaulted to continuous mode
and emitted a float action into a `Discrete(6)` env. Fixed by adding
`BaseBenchmark._finalize_agent_params()` (mirrors `build_agent`'s guard: only
sets `state_type` if the agent class actually accepts it) and calling it from
all three benchmark classes. Verified against the existing 27-test
benchmark/taxi suite plus the full 157-test non-BeamNG suite — all pass.

**Local environment was broken before any of this could run.** `.venv`
pointed at an Anaconda install (`C:\Users\moham\anaconda3`) that no longer
exists on this machine. Rebuilt the venv against the standalone Python 3.11
install with `--system-site-packages` to reuse the already-installed
torch/numpy/matplotlib, then pinned `gymnasium==1.0.0` as
`requirements.txt` specifies — the globally installed gymnasium 1.3.0 has
removed `Taxi-v3` in favor of `Taxi-v4`, which would have broken every run silently.

**DQN-on-Taxi crash: already found and fixed upstream.** Independently hit a
matmul shape-mismatch crash feeding Taxi's raw integer state into a network
built for a 500-dim input. Pulling `main` showed this was already resolved 91
commits ago via `DQNAgent._encode` (one-hot encoding), covered by
`tests/test_taxi_algorithms.py` (18/18 passing). No code change needed here.

**Wall-clock cost gap widens sharply as the networks grow.** Tabular
Q-learning trains 1200 episodes in 0.65s. DQN averages ~2.6s/episode, TD3
~5.7s/episode, DDPG ~15.8s/episode by default — a **~29,000x** per-episode gap
between Q-learning and DDPG, on a CPU-only machine with no GPU available in
this session. Expected for a 500-state discrete problem (no forward/backward
pass beats a Q-table lookup), but the gap tracks the param-count table above
almost exactly: more network, more wall-clock, for a worse result on this
particular environment.

## Next steps

- Run a small `epsilon_decay` grid-search for `dqn` / `dqn_per` on Taxi to test the hypothesis above before writing DQN off for this environment.
- Re-run the head-to-head comparison at a matched, larger episode budget.
- For TD3/DDPG on Taxi specifically: the exploration mechanism (Gaussian/OU noise over continuous actor scores) is likely the wrong tool for a 6-way discrete choice — worth trying plain epsilon-greedy over the actor's scores instead of the current always-on noise, before concluding these architectures simply can't work here.
- Reuse this exact harness (`core.pipeline_actions.run_benchmark`) on BeamNG once GPU time is available — no script changes needed, just longer wall-clock budget. This is where TD3/DDPG's continuous-control design actually fits.
- New runs are synced to `web/public/data/` for the dashboard (`scripts/sync_web_data.py`) alongside the existing q_learning demo data.

---

git `235b413` · Python 3.11.9 · torch 2.12.0+cpu · device: cpu · Windows · 2026-07-03 / 2026-07-05
