"""
BeamNG.drive DQN Evaluation Script
====================================
Loads a trained model and runs it in pure exploitation mode (ε = 0).
Usage:
    python evaluate_beamng.py
"""

import numpy as np

from beamng_env import BeamNGDrivingEnv
from dqn_agent import DQNAgent
from config import BEAMNG_HOME, BEAMNG_USER

# ---------------------------------------------------------------------------
MODEL_PATH      = 'beamng_dqn.pth'
N_EVAL_EPISODES = 10
# ---------------------------------------------------------------------------


def evaluate():
    env = BeamNGDrivingEnv(beamng_home=BEAMNG_HOME, beamng_user=BEAMNG_USER)
    agent = DQNAgent(
        n_states=BeamNGDrivingEnv.N_STATES,
        n_actions=BeamNGDrivingEnv.N_ACTIONS,
        epsilon=0.0,  # pure exploitation — no random actions
    )
    agent.load(MODEL_PATH)
    agent.epsilon = 0.0

    rewards = []
    lengths = []
    laps    = 0

    try:
        for ep in range(1, N_EVAL_EPISODES + 1):
            state     = env.reset()
            ep_reward = 0.0
            steps     = 0
            completed = False

            while True:
                action                         = agent.select_action(state)
                state, reward, done, info      = env.step(action)
                ep_reward += reward
                steps     += 1

                if reward >= 150:   # lap completion bonus fired
                    completed = True

                if done:
                    break

            rewards.append(ep_reward)
            lengths.append(steps)
            laps += int(completed)
            status = 'LAP COMPLETE' if completed else 'crashed/timeout'
            print(f"Episode {ep:2d}: Reward={ep_reward:8.1f}  Steps={steps:4d}  [{status}]")

    finally:
        env.close()

    print()
    print("=" * 50)
    print(f"Evaluation over {N_EVAL_EPISODES} episodes")
    print(f"  Average Reward : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Average Steps  : {np.mean(lengths):.0f}")
    print(f"  Best Reward    : {max(rewards):.1f}")
    print(f"  Laps completed : {laps}/{N_EVAL_EPISODES}")
    print("=" * 50)


if __name__ == '__main__':
    evaluate()
