"""Parallel training loop for BeamNGMultiEnv: many agents, one shared step."""

import time

import numpy as np
from tqdm import tqdm

from core.stop_signal import stop_requested


class MultiAgentRunner:
    """Drives N agents on one shared BeamNG scenario.

    Each tick: collect every active agent's action, step physics once, then
    update every agent. A vehicle whose episode ends is teleported to spawn and
    continues immediately while the others keep driving. The session stops when
    every agent has completed ``n_episodes`` or the wall-clock ``time_limit``
    (seconds) is reached — whichever comes first.
    """

    def train(self, env, n_episodes, time_limit=None, save_every=50):
        start = time.time()
        env.reset_all()

        def time_up():
            return time_limit is not None and (time.time() - start) >= time_limit

        def all_done():
            return all(s.episode >= n_episodes for s in env.slots)

        # Progress spans only the episodes left to run this session, so resumed
        # runs (slots loaded past episode 0) still advance to 100%.
        remaining = sum(max(0, n_episodes - s.episode) for s in env.slots)
        pbar = tqdm(total=remaining, desc="Multi-agent", unit="ep")

        try:
            while not all_done() and not time_up() and not stop_requested():
                pending = []
                for slot in env.slots:
                    if slot.episode >= n_episodes:
                        continue
                    state = slot.last_obs
                    action = slot.agent.select_action(state)
                    env.apply_action(slot, action)
                    pending.append((slot, state, action))

                if not pending:
                    break

                env.step_physics()

                for slot, state, action in pending:
                    next_obs = env.observe(slot)
                    # Count the step before reward so the MAX_STEPS termination
                    # check in compute_reward matches the single-agent env, which
                    # increments _steps before computing its reward.
                    slot.steps += 1
                    reward, done = env.compute_reward(slot, next_obs)
                    loss = slot.agent.update(state, action, reward, next_obs, done)
                    if loss is not None:
                        slot.ep_losses.append(loss)
                    slot.ep_reward += reward
                    if isinstance(state, np.ndarray) and len(state) > 0:
                        slot.ep_speeds.append(float(state[0]) * 50.0)
                    slot.last_obs = next_obs

                    if done:
                        self._finish_episode(env, slot, save_every)
                        pbar.update(1)
                        pbar.set_postfix(active=sum(1 for s in env.slots if s.episode < n_episodes))
        except KeyboardInterrupt:
            pbar.write("Multi-agent training interrupted by user.")
        finally:
            pbar.close()
            for slot in env.slots:
                slot.agent.save(slot.save_path)
                self._save_slot_plot(slot)

        return {
            slot.name: {
                "episodes": slot.episode,
                "rewards": slot.reward_history,
                "steps": slot.steps_history,
            }
            for slot in env.slots
        }

    def _finish_episode(self, env, slot, save_every):
        avg_speed = float(np.mean(slot.ep_speeds)) if slot.ep_speeds else 0.0
        slot.reward_history.append(slot.ep_reward)
        slot.steps_history.append(slot.steps)
        slot.speed_history.append(avg_speed)
        # checkpoints_reached, not waypoint_idx: the finish zeroes the index, so a
        # completed episode logged 0 checkpoints — which blanked the plot panel.
        slot.distance_history.append(slot.checkpoints_reached)
        slot.agent.decay_epsilon()
        if hasattr(slot.agent, "episode"):
            slot.agent.episode = slot.episode + 1
        slot.episode += 1

        avg = np.mean(slot.reward_history[-20:])
        print(
            f"[{slot.name}] ep {slot.episode} reward={slot.ep_reward:.1f} "
            f"avg20={avg:.1f} eps={getattr(slot.agent, 'epsilon', 0.0):.3f} "
            f"speed={avg_speed:.1f}m/s wpts={slot.checkpoints_reached}"
        )

        if save_every and slot.episode % save_every == 0:
            slot.agent.save(slot.save_path)
            print(f"[{slot.name}] checkpoint saved -> {slot.save_path}")

        env.reset_vehicle(slot)

    def _save_slot_plot(self, slot):
        """Write a per-agent reward/steps plot beside that agent's checkpoint."""
        if not slot.reward_history:
            return
        import os

        from core.runner import PipelineRunner

        plot_dir = os.path.dirname(slot.save_path) or "."
        PipelineRunner._save_plot(
            slot.reward_history,
            slot.steps_history,
            slot.name,
            os.path.join(plot_dir, f"{slot.name}_training.png"),
            slot.episode,
            slot.speed_history,
            slot.distance_history,
        )
