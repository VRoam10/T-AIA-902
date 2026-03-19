import gymnasium as gym


def make_taxi():
    """Factory for the Taxi-v3 environment."""
    return gym.make("Taxi-v3")
