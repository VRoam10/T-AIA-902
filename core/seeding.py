"""Reproducible seeding helpers shared across the pipeline.

Agents in this project rely on the *global* RNGs (``random``, ``numpy``,
and — for the deep agents — ``torch``) rather than per-instance generators.
Seeding those global RNGs once, plus seeding the Gymnasium environment, is
therefore enough to make any algorithm/environment run fully reproducible.
"""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every global RNG used by the pipeline.

    Covers ``random``, ``numpy`` and, when available, ``torch`` (CPU + CUDA).
    Torch is imported lazily so that tabular runs (Q-Learning) keep working
    even when PyTorch is not installed.

    Args:
        seed: The seed value to apply to all RNGs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_action_space(env, seed: int) -> None:
    """Seed an environment's action space for reproducible ``sample()`` calls.

    Stays agnostic to the environment API: if the action space has no ``seed``
    method (or seeding fails), the call is a no-op.

    Args:
        env: The environment whose action space should be seeded.
        seed: The seed value to apply.
    """
    action_space = getattr(env, "action_space", None)
    if action_space is not None and hasattr(action_space, "seed"):
        try:
            action_space.seed(seed)
        except (AttributeError, TypeError):
            pass


def seed_env(env, seed: int) -> None:
    """Fully seed a Gymnasium (or legacy gym) environment.

    Seeds the environment through ``reset(seed=...)`` (falling back to a legacy
    ``env.seed`` method) and seeds the action space. Useful for standalone
    seeding outside the training loop.

    Args:
        env: The environment instance to seed.
        seed: The seed value to apply.
    """
    try:
        env.reset(seed=seed)
    except TypeError:
        try:
            env.seed(seed)
        except (AttributeError, TypeError):
            pass
    seed_action_space(env, seed)
