"""Fairness-aware optimization and disparity metrics for FedAgent-Chain.

Implements the fairness disparity measure D_fair and the fairness regularization
objective Ω_fair from Section 4.6 of the FedAgent-Chain paper.
"""

from __future__ import annotations

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger("FairnessOptimizer")


def compute_fairness_disparity(
    group_metrics: dict[str, float],
    metric_name: str = "f1_score",
) -> float:
    """Compute fairness disparity across protected groups.

    Implements the fairness disparity measure from Section 4.6:
        D_fair = max_{g ∈ G} M_g - min_{g ∈ G} M_g

    Lower D_fair values indicate more equitable performance across groups.

    Parameters
    ----------
    group_metrics : dict
        Mapping from group identifier to performance metric value.
        Example: {"mobility": 0.82, "vision": 0.79, "hearing": 0.84}
    metric_name : str, optional
        Name of the metric for logging purposes.

    Returns
    -------
    float
        Fairness disparity value D_fair ∈ [0, 1]. Lower is better.

    Raises
    ------
    ValueError
        If group_metrics is empty.

    Examples
    --------
    >>> metrics = {"disability_A": 0.85, "disability_B": 0.72, "disability_C": 0.91}
    >>> compute_fairness_disparity(metrics)
    0.19
    """
    if not group_metrics:
        raise ValueError("group_metrics cannot be empty.")
    if len(group_metrics) == 1:
        return 0.0

    values = list(group_metrics.values())
    disparity = float(max(values) - min(values))

    logger.debug(
        "Fairness disparity computed",
        metric=metric_name,
        group_metrics=group_metrics,
        disparity=round(disparity, 4),
        max_group=max(group_metrics, key=group_metrics.get),  # type: ignore[arg-type]
        min_group=min(group_metrics, key=group_metrics.get),  # type: ignore[arg-type]
    )
    return disparity


def compute_fairness_penalty(
    group_metrics: dict[str, float],
    lambda_fairness: float = 0.1,
) -> float:
    """Compute the fairness regularization penalty Ω_fair.

    The penalty discourages models that exhibit high performance disparity
    across protected groups. It is added to the local training objective.

    Parameters
    ----------
    group_metrics : dict
        Per-group performance metrics.
    lambda_fairness : float
        Fairness regularization weight. Higher values enforce stricter fairness
        at the potential cost of overall performance.

    Returns
    -------
    float
        Fairness penalty Ω_fair = λ × D_fair.
    """
    disparity = compute_fairness_disparity(group_metrics)
    return lambda_fairness * disparity


def compute_all_disparities(
    per_group_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute fairness disparity across all protected attribute dimensions.

    Parameters
    ----------
    per_group_metrics : dict
        Nested dict mapping attribute name → {group_value → metric}.
        Example: {
            "disability": {"mobility": 0.82, "vision": 0.79},
            "language": {"en": 0.85, "ar": 0.74},
        }

    Returns
    -------
    dict
        Mapping from attribute name to disparity value.

    Examples
    --------
    >>> metrics = {
    ...     "disability": {"mobility": 0.80, "vision": 0.72},
    ...     "language": {"en": 0.85, "ar": 0.74},
    ... }
    >>> disparities = compute_all_disparities(metrics)
    >>> disparities["disability"]
    0.08
    """
    return {
        attr: compute_fairness_disparity(group_map, metric_name=attr)
        for attr, group_map in per_group_metrics.items()
    }


def group_performance_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
) -> dict[str, float]:
    """Compute per-group F1 scores from predictions and group membership.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    group_labels : np.ndarray
        Group membership array (categorical labels as strings or integers).

    Returns
    -------
    dict
        Mapping from group label to F1 score.
    """
    from sklearn.metrics import f1_score

    unique_groups = np.unique(group_labels)
    group_f1s: dict[str, float] = {}

    for group in unique_groups:
        mask = group_labels == group
        if mask.sum() == 0:
            continue
        f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0.0)
        group_f1s[str(group)] = float(f1)

    return group_f1s


def fairness_reweight_samples(
    group_labels: np.ndarray,
    target_distribution: dict[str, float] | None = None,
) -> np.ndarray:
    """Compute per-sample weights to rebalance group representation.

    Parameters
    ----------
    group_labels : np.ndarray
        Group membership labels for each sample.
    target_distribution : dict, optional
        Target proportion per group. If None, uses uniform distribution.

    Returns
    -------
    np.ndarray
        Per-sample weight array of shape (n_samples,).
    """
    unique_groups, counts = np.unique(group_labels, return_counts=True)
    n_samples = len(group_labels)
    n_groups = len(unique_groups)

    if target_distribution is None:
        target = {g: 1.0 / n_groups for g in unique_groups}
    else:
        total = sum(target_distribution.values())
        target = {g: v / total for g, v in target_distribution.items()}

    actual = {g: c / n_samples for g, c in zip(unique_groups, counts, strict=False)}
    weights = np.ones(n_samples, dtype=np.float32)

    for group, _count in zip(unique_groups, counts, strict=False):
        mask = group_labels == group
        target_prop = target.get(str(group), 1.0 / n_groups)
        actual_prop = actual[group]
        if actual_prop > 0:
            sample_weight = target_prop / actual_prop
            weights[mask] = sample_weight

    # Normalize weights to sum to n_samples
    weights = weights * (n_samples / weights.sum())
    return weights
