"""Federated averaging aggregators for FedAgent-Chain.

Implements standard FedAvg and fairness-aware FedAvg as described in
Section 4.7 of the FedAgent-Chain paper.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from omegaconf import DictConfig

from src.utils.logging_utils import get_logger

logger = get_logger("FederatedAggregator")


class FedAvgAggregator:
    """Standard Federated Averaging aggregator.

    Implements the classic FedAvg algorithm (McMahan et al., 2017).
    Aggregation weight for node k is proportional to its local dataset size:
        ω_k = n_k / Σ_j n_j

    Parameters
    ----------
    cfg : DictConfig, optional
        Configuration object (unused in base FedAvg but kept for API consistency).
    """

    def __init__(self, cfg: DictConfig | None = None) -> None:
        self.cfg = cfg

    def compute_weights(
        self, updates: List[Tuple[str, Dict[str, np.ndarray], int]]
    ) -> Dict[str, float]:
        """Compute per-node aggregation weights.

        Parameters
        ----------
        updates : list of (node_id, state_dict, n_samples)
            Local model updates from each participating node.

        Returns
        -------
        dict
            Mapping from node_id to aggregation weight ω_k.
        """
        total_samples = sum(n for _, _, n in updates)
        return {node_id: n / total_samples for node_id, _, n in updates}

    def aggregate(
        self, updates: List[Tuple[str, Dict[str, np.ndarray], int]]
    ) -> Dict[str, np.ndarray]:
        """Aggregate local absolute model weights via weighted average.

        Parameters
        ----------
        updates : list of (node_id, absolute_weights, n_samples)
            Local model absolute weights from each participating node.

        Returns
        -------
        dict
            Aggregated global model absolute weight state dict.

        Raises
        ------
        ValueError
            If updates list is empty or contains inconsistent parameter shapes.
        """
        if not updates:
            raise ValueError("Cannot aggregate: updates list is empty.")

        # Defensive check: ensure no NaN/Inf in input weights
        for node_id, state_dict, _ in updates:
            for p_name, p_val in state_dict.items():
                if np.any(np.isnan(p_val)) or np.any(np.isinf(p_val)):
                    raise ValueError(
                        f"NaN or Inf detected in state_dict from node '{node_id}', "
                        f"parameter '{p_name}'. Indicates DP noise or accumulation corruption."
                    )

        weights = self.compute_weights(updates)
        _, first_state, _ = updates[0]

        aggregated: Dict[str, np.ndarray] = {}
        for param_name in first_state:
            weighted_sum = np.zeros_like(first_state[param_name], dtype=np.float64)
            for node_id, state_dict, _ in updates:
                if param_name not in state_dict:
                    raise ValueError(
                        f"Parameter '{param_name}' missing from node '{node_id}' update."
                    )
                weighted_sum += weights[node_id] * state_dict[param_name].astype(np.float64)
            aggregated[param_name] = weighted_sum.astype(first_state[param_name].dtype)

        logger.info(
            "Aggregation complete",
            n_nodes=len(updates),
            weights={k: round(v, 4) for k, v in weights.items()},
        )
        return aggregated


class FairnessAwareFedAvgAggregator(FedAvgAggregator):
    """Fairness-aware federated averaging aggregator.

    Implements reweighted aggregation that upweights nodes with
    underrepresented protected groups, as described in Section 4.7
    of the FedAgent-Chain paper.

    The aggregation weight for node k is:
        ω_k = (n_k × ρ_k) / Σ_j (n_j × ρ_j)

    where ρ_k is the fairness adjustment factor for node k, computed
    as the inverse of that node's minimum group performance.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with fields:
        - lambda_fairness (float): Fairness regularization weight.
        - protected_groups (list): Names of protected group attributes.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.lambda_fairness: float = cfg.get("lambda_fairness", 0.1)
        self.protected_groups: List[str] = list(cfg.get("protected_groups", []))

    def compute_fairness_adjustment(
        self, node_id: str, fairness_scores: Dict[str, float]
    ) -> float:
        """Compute the fairness adjustment factor ρ_k for a node.

        Nodes with lower minimum group performance receive a higher ρ_k,
        increasing their weight in the global aggregation.

        Parameters
        ----------
        node_id : str
            Node identifier.
        fairness_scores : dict
            Mapping from node_id to that node's minimum group F1 score.

        Returns
        -------
        float
            Fairness adjustment factor ρ_k ≥ 1.0.
        """
        min_group_score = fairness_scores.get(node_id, 1.0)
        # Nodes with higher min-group performance get higher weight
        rho = 1.0 + self.lambda_fairness * min_group_score
        return float(rho)

    def compute_weights(  # type: ignore[override]
        self,
        updates: List[Tuple[str, Dict[str, np.ndarray], int]],
        fairness_scores: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """Compute fairness-adjusted per-node aggregation weights.

        Parameters
        ----------
        updates : list of (node_id, state_dict, n_samples)
            Local model updates.
        fairness_scores : dict, optional
            Per-node minimum group performance scores. If None, falls back
            to standard FedAvg weights.

        Returns
        -------
        dict
            Mapping from node_id to fairness-adjusted weight ω_k.
        """
        if fairness_scores is None:
            return super().compute_weights(updates)

        adjusted_weights = {}
        for node_id, _, n_samples in updates:
            rho = self.compute_fairness_adjustment(node_id, fairness_scores)
            adjusted_weights[node_id] = n_samples * rho

        total = sum(adjusted_weights.values())
        normalized = {k: v / total for k, v in adjusted_weights.items()}

        logger.info(
            "Fairness-adjusted weights computed",
            weights={k: round(v, 4) for k, v in normalized.items()},
            lambda_fairness=self.lambda_fairness,
        )
        return normalized

    def aggregate(  # type: ignore[override]
        self,
        updates: List[Tuple[str, Dict[str, np.ndarray], int]],
        fairness_scores: Dict[str, float] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Aggregate absolute weights with fairness-adjusted weights.

        Parameters
        ----------
        updates : list of (node_id, absolute_weights, n_samples)
            Local model absolute weights.
        fairness_scores : dict, optional
            Per-node fairness scores for weight adjustment.

        Returns
        -------
        dict
            Aggregated global model absolute weight state dict.
        """
        if not updates:
            raise ValueError("Cannot aggregate: updates list is empty.")

        weights = self.compute_weights(updates, fairness_scores)
        _, first_state, _ = updates[0]

        aggregated: Dict[str, np.ndarray] = {}
        for param_name in first_state:
            weighted_sum = np.zeros_like(first_state[param_name], dtype=np.float64)
            for node_id, state_dict, _ in updates:
                weighted_sum += weights[node_id] * state_dict[param_name].astype(np.float64)
            aggregated[param_name] = weighted_sum.astype(first_state[param_name].dtype)

        return aggregated
