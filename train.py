import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent

OPTIMIZED_PARAMS = {
    "learning_rate": 0.85,
    "discount_factor": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.9975,
}


def train(
    n_episodes: int = 10000,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    time_limit: float = None,
):
    env = gym.make("Taxi-v3")

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    rewards_history = []
    steps_history = []
    start_time = time.time()

    for episode in range(n_episodes):
        if time_limit is not None and (time.time() - start_time) >= time_limit:
            print(f"\nTime limit reached after {episode} episodes.")
            break

        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)
        steps_history.append(steps)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_history[-1000:])
            avg_steps = np.mean(steps_history[-1000:])
            elapsed = time.time() - start_time
            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Avg Steps: {avg_steps:.1f} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

    env.close()
    agent.save("q_table.npy")
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s. Q-table saved to q_table.npy")

    plot_training(rewards_history, steps_history)
    return agent


def plot_training(rewards: list, steps: list, window: int = 100):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    smoothed_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smoothed_steps = np.convolve(steps, np.ones(window) / window, mode="valid")

    ax1.plot(smoothed_rewards)
    ax1.set_title(f"Rewards (rolling avg over {window} episodes)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True)

    ax2.plot(smoothed_steps, color="orange")
    ax2.set_title(f"Steps per Episode (rolling avg over {window} episodes)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()
    print("Training plot saved to training_results.png")


if __name__ == "__main__":
    train()
