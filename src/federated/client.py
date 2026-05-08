"""Federated learning client for local node training in FedAgent-Chain.

Each regional institutional node runs a FederatedClient instance that:
1. Loads local data (consented records only)
2. Trains the global model locally for E epochs
3. Computes differential-privacy-protected model updates
4. Submits the update hash to the blockchain audit layer
5. Reports local metrics and per-group fairness scores back to the aggregator
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.blockchain.chain import PermissionedBlockchain
from src.data.dataset import EmploymentDataset
from src.evaluation.fairness_evaluator import FairnessEvaluator
from src.evaluation.metrics import compute_classification_metrics
from src.federated.fairness import group_performance_from_predictions
from src.federated.privacy import protect_state_dict
from src.models.employment_model import EmploymentMatchingModel
from src.utils.logging_utils import get_logger
from src.utils.seed_utils import set_global_seed


class FederatedClient:
    """Local training client for a single institutional node.

    Parameters
    ----------
    node_id : str
        Unique identifier for this regional node.
    dataset : EmploymentDataset
        Local dataset (consented records only).
    cfg : DictConfig
        Client configuration with fields:
        - local_epochs (int): Number of local training epochs per round.
        - batch_size (int): Training batch size.
        - learning_rate (float): Local optimizer learning rate.
        - privacy.clipping_threshold (float): DP clipping threshold C.
        - privacy.noise_multiplier (float): DP noise multiplier σ.
    blockchain : PermissionedBlockchain
        Shared blockchain instance for audit logging.
    device : str
        PyTorch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        node_id: str,
        dataset: EmploymentDataset,
        cfg: DictConfig,
        blockchain: PermissionedBlockchain,
        device: str = "cpu",
    ) -> None:
        self.node_id = node_id
        self.dataset = dataset
        self.cfg = cfg
        self.blockchain = blockchain
        self.device = torch.device(device)
        self.logger = get_logger(f"FederatedClient[{node_id}]")
        self.fairness_evaluator = FairnessEvaluator()

        fed_cfg     = cfg.get("federated", {})
        privacy_cfg = cfg.get("privacy", {})

        self.local_epochs:   int   = int(fed_cfg.get("local_epochs",           5))
        self.batch_size:     int   = int(fed_cfg.get("batch_size",             64))
        self.learning_rate:  float = float(fed_cfg.get("learning_rate",        0.001))
        self.C:              float = float(privacy_cfg.get("clipping_threshold", 1.0))
        self.sigma:          float = float(privacy_cfg.get("noise_multiplier",   0.1))
        self.consent_ref: str = f"consent_{node_id}_v1"
        self.policy_ref: str = f"policy_{node_id}_gdpr"

    def train_round(
        self,
        global_state: Dict[str, np.ndarray],
        round_number: int,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], int, Dict[str, float]]:
        """Execute one federated learning round on local data.

        This method:
        1. Loads the global model weights
        2. Trains locally for E epochs
        3. Computes the update Δw_k = w_k - w_global
        4. Applies DP protection: clip + Gaussian noise
        5. Submits h_k^t = H(Δw̃_k || ID_k || t) to the blockchain
        6. Returns the protected update and local evaluation metrics

        Parameters
        ----------
        global_state : dict
            Global model parameter dictionary from the aggregator.
        round_number : int
            Current federated learning round index.
        seed : int, optional
            Local training seed for reproducibility.

        Returns
        -------
        tuple of (protected_update, n_samples, metrics)
            - protected_update: Dict of DP-protected parameter updates
            - n_samples: Number of local training samples
            - metrics: Dict with 'loss', 'accuracy', 'f1', fairness scores
        """
        if seed is not None:
            set_global_seed(seed + round_number)

        # Initialise model from global state
        model = EmploymentMatchingModel.from_config(self.cfg.get("model", {}))
        model.load_state_dict_numpy(global_state)
        model.to(self.device)

        # Save initial global state for computing delta
        initial_state = model.get_state_dict_numpy()

        # Local training
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        model.train()
        epoch_losses = []
        for epoch in range(self.local_epochs):
            batch_losses = []
            for batch in loader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                weights = batch.get("weight")

                optimizer.zero_grad()
                preds = model(features).squeeze(-1)
                loss = criterion(preds, labels)

                if weights is not None:
                    loss = (loss * weights.to(self.device)).mean()

                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))

            epoch_loss = float(np.mean(batch_losses))
            epoch_losses.append(epoch_loss)

        final_state = model.get_state_dict_numpy()

        # Compute update delta: Δw_k = w_k^t - w_global^t
        delta: Dict[str, np.ndarray] = {
            name: final_state[name] - initial_state[name]
            for name in final_state
        }

        # Apply differential privacy protection
        # Use a representative flat vector for hashing purposes
        flat_delta = np.concatenate([v.flatten() for v in delta.values()])
        protected_delta = protect_state_dict(delta, C=self.C, sigma=self.sigma, seed=seed)
        flat_protected = np.concatenate([v.flatten() for v in protected_delta.values()])

        # Submit hash to blockchain (ONLY the hash, never the raw update)
        self.blockchain.submit_model_update_hash(
            protected_update=flat_protected,
            node_id=self.node_id,
            round_number=round_number,
            consent_ref=self.consent_ref,
            policy_ref=self.policy_ref,
        )

        # Evaluate local model
        metrics = self._evaluate_local(model)
        metrics["train_loss"] = float(np.mean(epoch_losses))

        self.logger.info(
            "Local training round complete",
            node_id=self.node_id,
            round_number=round_number,
            local_epochs=self.local_epochs,
            n_samples=len(self.dataset),
            train_loss=round(metrics["train_loss"], 4),
            f1=round(metrics.get("f1", 0.0), 4),
        )

        return protected_delta, len(self.dataset), metrics

    def _evaluate_local(self, model: EmploymentMatchingModel) -> Dict[str, float]:
        """Evaluate the locally trained model on the local dataset.

        Parameters
        ----------
        model : EmploymentMatchingModel
            Trained local model.

        Returns
        -------
        dict
            Evaluation metrics including F1, accuracy, and per-group fairness scores.
        """
        model.eval()
        loader = DataLoader(self.dataset, batch_size=256, shuffle=False)

        all_labels, all_preds = [], []
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                labels = batch["label"].numpy()
                preds = model.predict(features).cpu().numpy()
                all_labels.extend(labels.tolist())
                all_preds.extend(preds.tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        metrics = compute_classification_metrics(y_true, y_pred)

        # Per-disability-group fairness score for fairness-aware aggregation
        try:
            group_labels = self.dataset.get_group_labels("disability_category")
            group_f1s = group_performance_from_predictions(y_true, y_pred, group_labels)
            metrics["min_group_f1"] = float(min(group_f1s.values())) if group_f1s else 0.0
            metrics["fairness_disparity_disability"] = float(
                max(group_f1s.values()) - min(group_f1s.values())
            ) if len(group_f1s) >= 2 else 0.0
        except Exception:
            metrics["min_group_f1"] = 0.0
            metrics["fairness_disparity_disability"] = 0.0

        return metrics
