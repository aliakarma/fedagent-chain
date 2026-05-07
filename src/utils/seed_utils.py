"""Random seed control utilities for full reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set all random seeds for full deterministic reproducibility.

    Must be called at the very beginning of every script entry point before
    any stochastic operation is performed.

    Parameters
    ----------
    seed : int
        The random seed to set globally. The paper uses seed=42 for all
        reported results. Seeds 123 and 2024 are recommended for secondary
        verification runs.

    Notes
    -----
    This function sets seeds for: Python's random module, NumPy, PyTorch CPU,
    PyTorch CUDA (all devices), and the PYTHONHASHSEED environment variable.
    It also configures cuDNN for deterministic operation.

    Examples
    --------
    >>> set_global_seed(42)
    >>> import torch
    >>> x = torch.randn(3, 3)  # Deterministic across runs
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed; skip torch-specific seeding


def get_rng(seed: int) -> np.random.Generator:
    """Return a NumPy random generator with the given seed.

    Parameters
    ----------
    seed : int
        Seed for the random generator.

    Returns
    -------
    np.random.Generator
        A seeded NumPy random generator instance.
    """
    return np.random.default_rng(seed)
