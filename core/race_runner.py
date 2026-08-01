"""Race loop for BeamNGRaceEnv: N cars on one track, optionally still learning.

Two modes, one loop:

  * **Exhibition** (``learning=False``) — policies are frozen and exploration noise
    is off, so the race shows what the checkpoints actually learned. Nothing is
    written back.
  * **Race-training** (``learning=True``) — agents keep updating from the race, so
    the reward's gap term has teeth: the only way to earn it is to be ahead of the
    other car. Checkpoints are saved at the end of every race.

`MultiAgentRunner` stays the throughput trainer (N agents, N separate paths, no
contact); this is the head-to-head one.
"""

import time

import numpy as np
from tqdm import tqdm

from core.stop_signal import stop_requested
from environments import beamng_spec


class RaceRunner:
    """Drives a sequence of races on a shared track."""

    # Realtime pacing: a human cannot drive in lockstep, so in realtime mode the
    # simulator runs freely and we poll at this interval. It is derived from the
    # lockstep step duration so the agent's control rate is the same in both modes —
    # hardcoding 100 ms here would have driven the agent 3x faster in a human race
    # than in the training it learned from.
    REALTIME_TICK_S = beamng_spec.SECONDS_PER_ENV_STEP

    # Hard cap on ticks per race, so a stalled field cannot loop forever when every
    # car is wedged short of both the finish and its own step limit.
    MAX_TICKS_PER_RACE_FACTOR = 3

    def run(self, env, races: int = 1, learning: bool = False, save_every: int = 1) -> dict:
        """Run ``races`` races and return the per-race results.

        Args:
            env: A :class:`~environments.beamng_race.BeamNGRaceEnv`.
            races: How many races to run back to back.
            learning: Keep updating the agents from race experience.
            save_every: With ``learning``, save checkpoints every N races.
        """
        results = []
        frozen = [] if learning else self._freeze_exploration(env)

        pbar = tqdm(total=races, desc="Racing", unit="race")
        try:
            for race_idx in range(races):
                if stop_requested():
                    pbar.write("Stopped by user.")
                    break
                result = self._one_race(env, learning=learning)
                result["race"] = race_idx + 1
                results.append(result)

                pbar.update(1)
                pbar.set_postfix(winner=result.get("winner") or "-", margin=result.get("margin_m"))
                pbar.write(self._describe(result))

                if learning and save_every and (race_idx + 1) % save_every == 0:
                    self._save(env)
        except KeyboardInterrupt:
            pbar.write("Race interrupted by user.")
        finally:
            pbar.close()
            if learning:
                self._save(env)
            else:
                self._restore_exploration(frozen)

        return {
            "races": len(results),
            "results": results,
            "wins": self._win_counts(results),
        }

    # ------------------------------------------------------------------
    # One race
    # ------------------------------------------------------------------

    def _one_race(self, env, *, learning: bool) -> dict:
        env.reset_race()
        max_ticks = env.MAX_STEPS * self.MAX_TICKS_PER_RACE_FACTOR
        ticks = 0

        while not env.race_over() and not stop_requested() and ticks < max_ticks:
            tick_start = time.time()

            # 1. Every driven car acts. The human's car is left alone — the player
            #    is driving it — but is still stepped and observed with the rest.
            pending = []
            for slot in env.agent_slots():
                if slot.done:
                    continue
                state = slot.last_obs
                action = slot.agent.select_action(state)
                env.apply_action(slot, action)
                pending.append((slot, state, action))

            # 2. One advance for the whole field, so contact is symmetric.
            env.advance()
            ticks += 1

            # 3. Observe everyone (this is what advances positions and checkpoints).
            observations = env.observe_all()

            # 4. Reward and, optionally, learn.
            for slot, state, action in pending:
                next_obs = observations[slot.name]
                # Count the step before the reward so the MAX_STEPS termination check
                # matches the single-vehicle env, which increments before rewarding.
                slot.steps += 1
                reward, done = env.compute_race_reward_for(slot, next_obs)
                if learning:
                    loss = slot.agent.update(state, action, reward, next_obs, done)
                    if loss is not None:
                        slot.ep_losses.append(loss)
                slot.ep_reward += reward
                if isinstance(state, np.ndarray) and state.size:
                    slot.ep_speeds.append(float(state[0]) * 50.0)
                slot.last_obs = next_obs
                slot.done = slot.done or done

            # 5. A human cannot finish by reward (nothing rewards them), so their
            #    step count and completion are tracked directly.
            for slot in env.human_slots():
                slot.steps += 1
                if slot.waypoints and slot.waypoint_idx >= len(slot.waypoints) * env.laps:
                    slot.finished = True

            # 6. Refresh the gap baselines for the next tick, once, after all
            #    rewards — so no slot's update moves another's baseline mid-tick.
            env.snapshot_progress()

            if env.realtime:
                self._pace(tick_start)

        if learning:
            for slot in env.agent_slots():
                slot.agent.decay_epsilon()
                slot.episode += 1
                if hasattr(slot.agent, "episode"):
                    slot.agent.episode = slot.episode
                slot.reward_history.append(slot.ep_reward)
                slot.steps_history.append(slot.steps)

        result = env.result()
        result["timed_out"] = ticks >= max_ticks
        result["rewards"] = {s.name: round(float(s.ep_reward), 1) for s in env.agent_slots()}
        return result

    def _pace(self, tick_start: float) -> None:
        """Sleep out the remainder of a realtime tick (never negative)."""
        elapsed = time.time() - tick_start
        remaining = self.REALTIME_TICK_S - elapsed
        if remaining > 0:
            time.sleep(remaining)

    # ------------------------------------------------------------------
    # Exhibition mode: freeze the policies
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze_exploration(env) -> list[tuple[object, float]]:
        """Zero every agent's exploration noise, remembering the old values.

        An exhibition race should show the learned policy, not a policy still taking
        random actions — with epsilon left at its training value the cars would drive
        visibly worse than they actually are.
        """
        saved = []
        for slot in env.agent_slots():
            agent = slot.agent
            if hasattr(agent, "epsilon"):
                saved.append((agent, agent.epsilon))
                try:
                    agent.epsilon = 0.0
                except AttributeError:
                    # A read-only epsilon property backs onto _epsilon.
                    agent._epsilon = 0.0
        return saved

    @staticmethod
    def _restore_exploration(saved) -> None:
        for agent, value in saved:
            try:
                agent.epsilon = value
            except AttributeError:
                agent._epsilon = value

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _save(env) -> None:
        for slot in env.agent_slots():
            if slot.save_path:
                slot.agent.save(slot.save_path)

    @staticmethod
    def _win_counts(results: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in results:
            name = r.get("winner")
            if name:
                counts[name] = counts.get(name, 0) + 1
        return counts

    @staticmethod
    def _describe(result: dict) -> str:
        order = " | ".join(
            f"{e['name']} {e['progress_m']:.0f}m cp{e['checkpoints']}"
            f"{' FINISH' if e['finished'] else ''}"
            for e in result["entrants"]
        )
        head = f"race {result.get('race', '?')}: {result.get('winner') or 'nobody'} wins"
        margin = f" by {result['margin_m']:.1f} m" if result.get("margin_m") else ""
        tail = "  (timed out)" if result.get("timed_out") else ""
        return f"{head}{margin} — {order}{tail}"
