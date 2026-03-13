import gymnasium as gym
import numpy as np
import pygame

from agent import QLearningAgent


def evaluate(q_table_path: str = "q_table.npy", n_episodes: int = 10, render: bool = True):
    render_mode = "human" if render else None
    env = gym.make("Taxi-v3", render_mode=render_mode)

    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
    )
    agent.load(q_table_path)
    agent.epsilon = 0.0  # Pure exploitation

    total_rewards = []
    total_steps = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if render:
                pygame.event.pump()
                pygame.time.wait(100)

        total_rewards.append(total_reward)
        total_steps.append(steps)
        print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

    env.close()

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Avg Reward : {np.mean(total_rewards):.2f}")
    print(f"  Avg Steps  : {np.mean(total_steps):.1f}")
    print(f"  Success rate: {sum(r > 0 for r in total_rewards) / n_episodes * 100:.1f}%")


if __name__ == "__main__":
    evaluate()
