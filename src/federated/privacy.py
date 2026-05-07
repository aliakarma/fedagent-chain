"""Differential privacy mechanisms for FedAgent-Chain.

Implements gradient clipping and Gaussian DP noise addition as described
in Section 4.5 of the FedAgent-Chain paper.

Privacy guarantee: The combination of clipping and noise addition provides
(ε, δ)-differential privacy, where ε and δ depend on the noise multiplier σ,
clipping threshold C, and the number of federated rounds T.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger("DifferentialPrivacy")


def clip_update(update: np.ndarray, C: float) -> np.ndarray:
    """Clip a model update to have L2 norm at most C.

    Implements the clipping operation from Section 4.5:
        Δw̃_k = Δw_k / max(1, ‖Δw_k‖_2 / C)

    Parameters
    ----------
    update : np.ndarray
        Flattened model update vector.
    C : float
        Clipping threshold (maximum allowed L2 norm). Must be positive.

    Returns
    -------
    np.ndarray
        Clipped update with L2 norm ≤ C. If norm ≤ C, update is unchanged.

    Raises
    ------
    ValueError
        If C is non-positive.

    Examples
    --------
    >>> update = np.array([3.0, 4.0])  # norm = 5.0
    >>> clipped = clip_update(update, C=5.0)  # Exactly at threshold
    >>> np.allclose(clipped, [3.0, 4.0])
    True
    >>> clipped = clip_update(update, C=2.5)  # Above threshold, will be clipped
    >>> np.isclose(np.linalg.norm(clipped), 2.5)
    True
    """
    if C <= 0:
        raise ValueError(f"Clipping threshold C must be positive, got {C}")
    l2_norm = float(np.linalg.norm(update))
    if l2_norm > C:
        scale = C / l2_norm
        return update * scale
    return update.copy()


def add_dp_noise(
    update: np.ndarray,
    sigma: float,
    C: float,
    seed: int | None = None,
) -> np.ndarray:
    """Add calibrated Gaussian noise for differential privacy.

    Adds noise N(0, (σ·C)²I) to the clipped update.

    Parameters
    ----------
    update : np.ndarray
        Clipped model update vector.
    sigma : float
        Noise multiplier. Larger values provide stronger privacy but
        reduce model utility.
    C : float
        Clipping threshold used in the previous clipping step.
    seed : int, optional
        Random seed for reproducible noise. In production, do not seed
        to maintain statistical guarantees.

    Returns
    -------
    np.ndarray
        Update with added Gaussian noise. Same shape as input.

    Raises
    ------
    ValueError
        If sigma or C is non-positive.

    Examples
    --------
    >>> update = np.zeros(100)
    >>> noisy = add_dp_noise(update, sigma=1.0, C=1.0, seed=42)
    >>> not np.allclose(update, noisy)  # Noise was added
    True
    """
    if sigma <= 0:
        raise ValueError(f"Noise multiplier sigma must be positive, got {sigma}")
    if C <= 0:
        raise ValueError(f"Clipping threshold C must be positive, got {C}")

    rng = np.random.default_rng(seed)
    noise_std = sigma * C
    noise = rng.normal(loc=0.0, scale=noise_std, size=update.shape)
    return update + noise


def protect_update(
    update: np.ndarray,
    C: float,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """Apply clipping followed by Gaussian DP noise in one step.

    Parameters
    ----------
    update : np.ndarray
        Raw model update vector.
    C : float
        Clipping threshold.
    sigma : float
        Noise multiplier.
    seed : int, optional
        Random seed for the noise.

    Returns
    -------
    np.ndarray
        Privacy-protected model update.
    """
    clipped = clip_update(update, C)
    noisy = add_dp_noise(clipped, sigma=sigma, C=C, seed=seed)

    logger.debug(
        "Update protected",
        original_norm=float(np.linalg.norm(update)),
        clipped_norm=float(np.linalg.norm(clipped)),
        noisy_norm=float(np.linalg.norm(noisy)),
        C=C,
        sigma=sigma,
    )
    return noisy


def protect_state_dict(
    state_dict: Dict[str, np.ndarray],
    C: float,
    sigma: float,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Apply DP protection to all parameters in a state dictionary.

    Parameters
    ----------
    state_dict : dict
        Model parameter dictionary mapping name → numpy array.
    C : float
        Per-parameter clipping threshold.
    sigma : float
        Noise multiplier.
    seed : int, optional
        Base seed for reproducible noise. Each parameter uses a derived seed.

    Returns
    -------
    dict
        Protected state dictionary with same keys.
    """
    protected = {}
    for i, (name, param) in enumerate(state_dict.items()):
        param_seed = None if seed is None else seed + i
        flat = param.flatten()
        protected_flat = protect_update(flat, C=C, sigma=sigma, seed=param_seed)
        protected[name] = protected_flat.reshape(param.shape)
    return protected
