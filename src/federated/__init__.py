"""Federated learning core module for FedAgent-Chain."""

from src.federated.aggregator import FairnessAwareFedAvgAggregator, FedAvgAggregator
from src.federated.fairness import (
    compute_all_disparities,
    compute_fairness_disparity,
    compute_fairness_penalty,
    fairness_reweight_samples,
    group_performance_from_predictions,
)
from src.federated.privacy import add_dp_noise, clip_update, protect_state_dict, protect_update

__all__ = [
    "FedAvgAggregator",
    "FairnessAwareFedAvgAggregator",
    "clip_update",
    "add_dp_noise",
    "protect_update",
    "protect_state_dict",
    "compute_fairness_disparity",
    "compute_fairness_penalty",
    "compute_all_disparities",
    "group_performance_from_predictions",
    "fairness_reweight_samples",
]
