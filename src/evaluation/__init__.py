"""Evaluation metrics and reporting for FedAgent-Chain."""

from src.evaluation.fairness_evaluator import FairnessEvaluator
from src.evaluation.metrics import (
    aggregate_metrics_across_nodes,
    compute_classification_metrics,
    compute_full_metrics,
    compute_ranking_metrics,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "compute_classification_metrics",
    "compute_ranking_metrics",
    "compute_full_metrics",
    "precision_at_k",
    "recall_at_k",
    "aggregate_metrics_across_nodes",
    "FairnessEvaluator",
]
