"""Federated Learning server (aggregator) for FedAgent-Chain.

Orchestrates multi-round federated training across all institutional nodes,
manages global model state, and coordinates fairness-aware aggregation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
from omegaconf import DictConfig

from src.federated.aggregator import FairnessAwareFedAvgAggregator, FedAvgAggregator
from src.federated.client import FederatedClient
from src.models.employment_model import EmploymentMatchingModel
from src.utils.io_utils import ensure_dir, save_json
from src.utils.logging_utils import get_logger

logger = get_logger("FederatedServer")


class FederatedServer:
    """Central coordination server for federated learning in FedAgent-Chain.

    Manages:
    - Global model initialization
    - Round-by-round aggregation (FedAvg or Fairness-Aware FedAvg)
    - MLflow experiment logging
    - Checkpoint saving
    - Per-round metric collection

    Parameters
    ----------
    cfg : DictConfig
        Full experiment configuration.
    use_fairness_aggregation : bool
        If True, use FairnessAwareFedAvgAggregator; otherwise standard FedAvg.
    output_dir : str or Path
        Directory to save checkpoints and per-round metrics.
    """

    def __init__(
        self,
        cfg: DictConfig,
        use_fairness_aggregation: bool = True,
        output_dir: str | Path = "experiments/runs/current",
    ) -> None:
        self.cfg = cfg
        self.output_dir = ensure_dir(output_dir)
        self.n_rounds: int = int(cfg.federated.get("n_rounds", 20))
        self.min_clients: int = int(cfg.federated.get("min_clients", 4))

        fed_cfg = cfg.get("federated", {})
        fairness_cfg = cfg.get("fairness", {})

        if use_fairness_aggregation:
            self.aggregator: FedAvgAggregator = FairnessAwareFedAvgAggregator(fairness_cfg)
            logger.info("Using Fairness-Aware FedAvg aggregation")
        else:
            self.aggregator = FedAvgAggregator()
            logger.info("Using standard FedAvg aggregation")

        self.use_fairness = use_fairness_aggregation

        # Initialise global model
        model_cfg = cfg.get("model", {})
        self.global_model = EmploymentMatchingModel.from_config(model_cfg)
        self.global_state: Dict[str, np.ndarray] = self.global_model.get_state_dict_numpy()

        self.round_history: List[Dict] = []
        logger.info(
            "FederatedServer initialised",
            n_rounds=self.n_rounds,
            min_clients=self.min_clients,
            model_params=self.global_model.count_parameters(),
        )

    def run(
        self,
        clients: List[FederatedClient],
        seed: int = 42,
        use_mlflow: bool = True,
    ) -> Dict:
        """Execute the full federated training loop.

        Parameters
        ----------
        clients : list of FederatedClient
            All participating institutional node clients.
        seed : int
            Base seed for reproducibility across rounds.
        use_mlflow : bool
            Whether to log metrics to MLflow.

        Returns
        -------
        dict
            Final evaluation results aggregated across all clients.

        Raises
        ------
        ValueError
            If fewer clients than min_clients are provided.
        """
        if len(clients) < self.min_clients:
            raise ValueError(
                f"Requires at least {self.min_clients} clients, got {len(clients)}."
            )

        if use_mlflow:
            mlflow.log_params({
                "n_rounds": self.n_rounds,
                "n_clients": len(clients),
                "use_fairness_aggregation": self.use_fairness,
                "seed": seed,
            })

        logger.info(
            "Federated training started",
            n_rounds=self.n_rounds,
            n_clients=len(clients),
            seed=seed,
        )

        for round_num in range(1, self.n_rounds + 1):
            round_start = time.time()
            round_metrics = self.run_round(clients, round_num=round_num, seed=seed)
            round_duration = time.time() - round_start

            round_metrics["round"] = round_num
            round_metrics["duration_seconds"] = round(round_duration, 2)
            self.round_history.append(round_metrics)

            if use_mlflow:
                mlflow_metrics = {
                    k: v for k, v in round_metrics.items()
                    if isinstance(v, (int, float))
                }
                mlflow.log_metrics(mlflow_metrics, step=round_num)

            logger.info(
                "Round complete",
                round=round_num,
                avg_f1=round(round_metrics.get("mean_f1", 0.0), 4),
                avg_fairness_disparity=round(
                    round_metrics.get("mean_fairness_disparity_disability", 0.0), 4
                ),
                duration_s=round(round_duration, 2),
            )

            # Save checkpoint every 5 rounds
            if round_num % 1 == 0:
                self._save_checkpoint(round_num)

        # Save final model
        self._save_checkpoint(self.n_rounds, final=True)
        self._save_round_history()

        logger.info("Federated training complete", total_rounds=self.n_rounds)
        return self._compute_final_results()

    def run_round(
        self,
        clients: List[FederatedClient],
        round_num: int,
        seed: int = 42,
    ) -> Dict:
        """Execute a single federated learning round.

        Parameters
        ----------
        clients : list of FederatedClient
            All participating clients.
        round_num : int
            Current round index (1-based).
        seed : int
            Base seed for this round.

        Returns
        -------
        dict
            Aggregated metrics for this round.
        """
        updates: List[Tuple[str, Dict[str, np.ndarray], int]] = []
        node_metrics: Dict[str, Dict] = {}
        fairness_scores: Dict[str, float] = {}

        # Phase 1: Local training at each node
        for client in clients:
            delta, n_samples, metrics = client.train_round(
                global_state=self.global_state.copy(),
                round_number=round_num,
                seed=seed,
            )
            updates.append((client.node_id, delta, n_samples))
            node_metrics[client.node_id] = metrics
            fairness_scores[client.node_id] = metrics.get("min_group_f1", 0.5)

        # Phase 2: Fairness-aware aggregation
        if self.use_fairness and isinstance(self.aggregator, FairnessAwareFedAvgAggregator):
            aggregated_delta = self.aggregator.aggregate(updates, fairness_scores)
        else:
            aggregated_delta = self.aggregator.aggregate(updates)

        # Phase 3: Apply aggregated delta to global model
        for name in self.global_state:
            if name in aggregated_delta:
                self.global_state[name] = self.global_state[name] + aggregated_delta[name]

        # Aggregate round metrics across nodes
        round_summary: Dict = {}
        for metric_name in ["f1", "accuracy", "precision", "recall",
                             "train_loss", "fairness_disparity_disability"]:
            values = [
                m[metric_name] for m in node_metrics.values()
                if metric_name in m
            ]
            if values:
                round_summary[f"mean_{metric_name}"] = float(np.mean(values))
                round_summary[f"std_{metric_name}"] = float(np.std(values))

        round_summary["n_clients"] = len(clients)
        round_summary["per_node"] = {
            nid: {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}
            for nid, m in node_metrics.items()
        }

        return round_summary

    def _save_checkpoint(self, round_num: int, final: bool = False) -> None:
        """Save the current global model state."""
        import torch

        prefix = "final_model" if final else f"global_model_round_{round_num:03d}"
        # Convert numpy state back to torch for saving
        torch_state = {
            name: torch.from_numpy(np.array(arr).copy())
            for name, arr in self.global_state.items()
        }
        ckpt_path = self.output_dir / "checkpoints" / f"{prefix}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch_state, ckpt_path)
        logger.info("Checkpoint saved", path=str(ckpt_path))

    def _save_round_history(self) -> None:
        """Save per-round metric history to JSON."""
        save_json(self.round_history, self.output_dir / "metrics" / "per_round.json")
        logger.info("Round history saved")

    def _compute_final_results(self) -> Dict:
        """Compute final aggregated results from round history."""
        if not self.round_history:
            return {}

        final = self.round_history[-1]
        best_f1_round = max(self.round_history, key=lambda r: r.get("mean_f1", 0.0))

        return {
            "final_round_metrics": final,
            "best_f1": best_f1_round.get("mean_f1", 0.0),
            "best_f1_round": best_f1_round.get("round", 0),
            "convergence_rounds": len(self.round_history),
        }
