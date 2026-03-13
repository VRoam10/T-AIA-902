"""
BeamNG.drive DQN Training Script
=================================
Usage:
    python train_beamng.py

Make sure BeamNG.drive is NOT already running — this script launches it automatically.
Edit BEAMNG_HOME below to point to your BeamNG.drive installation folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from beamng_env import BeamNGDrivingEnv
from dqn_agent import DQNAgent

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------
BEAMNG_HOME = r'C:\Program Files (x86)\Steam\steamapps\common\BeamNG.drive'
# BEAMNG_USER = r'C:\Users\YourName\AppData\Local\BeamNG.drive'  # optional

N_EPISODES   = 500    # total training episodes
SAVE_EVERY   = 50     # save checkpoint every N episodes
MODEL_PATH   = 'beamng_dqn.pth'
RESULTS_PATH = 'beamng_training_results.png'

# DQN hyper-parameters
LR             = 1e-3
GAMMA          = 0.99
EPSILON        = 1.0
EPSILON_MIN    = 0.05
EPSILON_DECAY  = 0.995
BATCH_SIZE     = 64
MEMORY_SIZE    = 20_000
TARGET_UPDATE  = 100
# ---------------------------------------------------------------------------


def train():
    env = BeamNGDrivingEnv(beamng_home=BEAMNG_HOME)
    agent = DQNAgent(
        n_states=BeamNGDrivingEnv.N_STATES,
        n_actions=BeamNGDrivingEnv.N_ACTIONS,
        lr=LR,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        memory_size=MEMORY_SIZE,
        target_update_freq=TARGET_UPDATE,
    )

    if os.path.exists(MODEL_PATH):
        print(f"Found existing model at '{MODEL_PATH}', resuming training.")
        agent.load(MODEL_PATH)

    episode_rewards = []
    episode_lengths = []

    try:
        for ep in range(1, N_EPISODES + 1):
            state     = env.reset()
            ep_reward = 0.0
            ep_loss   = []

            while True:
                action                          = agent.select_action(state)
                next_state, reward, done, info  = env.step(action)
                agent.store(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss is not None:
                    ep_loss.append(loss)

                state     = next_state
                ep_reward += reward

                if done:
                    break

            episode_rewards.append(ep_reward)
            episode_lengths.append(info['steps'])

            avg_r    = np.mean(episode_rewards[-20:])
            avg_loss = np.mean(ep_loss) if ep_loss else float('nan')
            print(
                f"Ep {ep:4d}/{N_EPISODES} | "
                f"Reward {ep_reward:8.1f} | "
                f"Avg20 {avg_r:8.1f} | "
                f"Steps {info['steps']:4d} | "
                f"ε {agent.epsilon:.3f} | "
                f"Loss {avg_loss:.4f}"
            )

            if ep % SAVE_EVERY == 0:
                agent.save(MODEL_PATH)
                _save_plot(episode_rewards, episode_lengths, ep)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        print("Saving final model and plot …")
        agent.save(MODEL_PATH)
        _save_plot(episode_rewards, episode_lengths, len(episode_rewards))
        env.close()


def _save_plot(rewards, lengths, episode):
    if not rewards:
        return
    window = min(20, len(rewards))
    eps    = range(1, len(rewards) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'BeamNG.drive DQN Training — Episode {episode}')

    ax1.plot(eps, rewards, alpha=0.3, label='Reward')
    if len(rewards) >= window:
        roll = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax1.plot(range(window, len(rewards) + 1), roll, label=f'Rolling avg ({window})')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(eps, lengths, alpha=0.3, label='Steps', color='orange')
    if len(lengths) >= window:
        roll = np.convolve(lengths, np.ones(window) / window, mode='valid')
        ax2.plot(range(window, len(lengths) + 1), roll,
                 label=f'Rolling avg ({window})', color='darkorange')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps per Episode')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(RESULTS_PATH)
    plt.close()
    print(f"Plot saved → {RESULTS_PATH}")


if __name__ == '__main__':
    train()
