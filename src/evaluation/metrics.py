"""Evaluation metrics for FedAgent-Chain employment matching.

Implements the full evaluation suite from Section 5 of the paper:
- Accuracy, Precision, Recall, F1 (binary classification)
- Precision@K and Recall@K (ranking metrics)
- Per-round training loss tracking
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels (0 or 1).
    y_pred : np.ndarray
        Predicted binary labels (0 or 1).

    Returns
    -------
    dict
        Dictionary with keys: accuracy, precision, recall, f1.

    Examples
    --------
    >>> y_true = np.array([1, 1, 0, 1, 0])
    >>> y_pred = np.array([1, 0, 0, 1, 1])
    >>> metrics = compute_classification_metrics(y_true, y_pred)
    >>> 0 <= metrics["f1"] <= 1
    True
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0.0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0.0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0.0)),
    }


def precision_at_k(
    y_true_scores: np.ndarray,
    y_pred_scores: np.ndarray,
    k: int = 5,
) -> float:
    """Compute Precision@K for ranking evaluation.

    Measures the fraction of top-K predictions that are relevant.

    Parameters
    ----------
    y_true_scores : np.ndarray
        Ground truth relevance scores or binary labels.
    y_pred_scores : np.ndarray
        Predicted relevance scores (higher = more relevant).
    k : int
        Number of top items to consider.

    Returns
    -------
    float
        Precision@K in [0, 1].
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, len(y_true_scores))

    top_k_indices = np.argsort(y_pred_scores)[::-1][:k]
    top_k_relevance = y_true_scores[top_k_indices]
    return float(np.sum(top_k_relevance > 0) / k)


def recall_at_k(
    y_true_scores: np.ndarray,
    y_pred_scores: np.ndarray,
    k: int = 5,
) -> float:
    """Compute Recall@K for ranking evaluation.

    Measures the fraction of relevant items found in the top-K predictions.

    Parameters
    ----------
    y_true_scores : np.ndarray
        Ground truth relevance scores or binary labels.
    y_pred_scores : np.ndarray
        Predicted relevance scores.
    k : int
        Number of top items to consider.

    Returns
    -------
    float
        Recall@K in [0, 1].
    """
    total_relevant = float(np.sum(y_true_scores > 0))
    if total_relevant == 0:
        return 0.0

    k = min(k, len(y_true_scores))
    top_k_indices = np.argsort(y_pred_scores)[::-1][:k]
    top_k_relevance = y_true_scores[top_k_indices]
    return float(np.sum(top_k_relevance > 0) / total_relevant)


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute Precision@K and Recall@K for multiple K values.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_scores : np.ndarray
        Predicted probability or suitability scores.
    k_values : list of int
        K values to evaluate at.

    Returns
    -------
    dict
        Metrics with keys like 'precision_at_5', 'recall_at_10', etc.
    """
    if k_values is None:
        k_values = [5, 10]
    metrics = {}
    for k in k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_scores, k)
        metrics[f"recall_at_{k}"] = recall_at_k(y_true, y_scores, k)
    return metrics


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute the complete evaluation metric set for FedAgent-Chain.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_scores : np.ndarray, optional
        Predicted probability/suitability scores for ranking metrics.
    k_values : list of int
        K values for Precision@K and Recall@K.

    Returns
    -------
    dict
        All metrics including classification and ranking metrics.
    """
    if k_values is None:
        k_values = [5, 10]
    metrics = compute_classification_metrics(y_true, y_pred)

    if y_scores is not None:
        ranking = compute_ranking_metrics(y_true, y_scores, k_values)
        metrics.update(ranking)

    return metrics


def aggregate_metrics_across_nodes(
    node_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Aggregate per-node metrics to global statistics.

    Parameters
    ----------
    node_metrics : dict
        Mapping from node_id to metric dictionary.

    Returns
    -------
    dict
        Aggregated metrics with mean and std across nodes.
    """
    all_values: dict[str, list[float]] = {}
    for _node_id, m in node_metrics.items():
        for metric_name, value in m.items():
            all_values.setdefault(metric_name, []).append(value)

    aggregated = {}
    for metric_name, values in all_values.items():
        arr = np.array(values)
        aggregated[f"mean_{metric_name}"] = float(np.mean(arr))
        aggregated[f"std_{metric_name}"] = float(np.std(arr))
        aggregated[f"min_{metric_name}"] = float(np.min(arr))
        aggregated[f"max_{metric_name}"] = float(np.max(arr))

    return aggregated
