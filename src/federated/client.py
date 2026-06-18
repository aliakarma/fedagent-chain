"""Federated learning client for local node training in FedAgent-Chain.

Each regional institutional node runs a FederatedClient instance that:
1. Loads local data (consented records only)
2. Trains the global model locally for E epochs
3. Computes differential-privacy-protected model updates
4. Submits the update hash to the blockchain audit layer
5. Reports local metrics and per-group fairness scores back to the aggregator
"""

from __future__ import annotations

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
        train_dataset: EmploymentDataset,
        test_dataset: EmploymentDataset,
        cfg: DictConfig,
        blockchain: PermissionedBlockchain,
        device: str = "cpu",
    ) -> None:
        self.node_id = node_id
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.dataset = train_dataset  # backward-compatible alias
        self.cfg = cfg
        self.blockchain = blockchain
        self.device = torch.device(device)
        self.logger = get_logger(f"FederatedClient[{node_id}]")
        self.fairness_evaluator = FairnessEvaluator()

        fed_cfg = cfg.get("federated", {})
        privacy_cfg = cfg.get("privacy", {})

        self.local_epochs: int = int(fed_cfg.get("local_epochs", 5))
        self.batch_size: int = int(fed_cfg.get("batch_size", 64))
        self.learning_rate: float = float(fed_cfg.get("learning_rate", 0.001))
        self.C: float = float(privacy_cfg.get("clipping_threshold", 1.0))
        self.sigma: float = float(privacy_cfg.get("noise_multiplier", 0.1))
        self.privacy_enabled: bool = bool(privacy_cfg.get("enabled", True))
        self.consent_ref: str = f"consent_{node_id}_v1"
        self.policy_ref: str = f"policy_{node_id}_gdpr"

    def train_round(
        self,
        global_state: dict[str, np.ndarray],
        round_number: int,
        seed: int | None = None,
    ) -> tuple[dict[str, np.ndarray], int, dict[str, float]]:
        """Execute one federated learning round on local data.

        Returns
        -------
        tuple of (protected_update, n_samples, metrics)
            - protected_update: Dict of DP-protected parameter updates
            - n_samples: Number of local training samples
            - metrics: Dict with 'loss', 'accuracy', 'f1', fairness scores,
                       'time_local_training', 'time_blockchain', 'model_size_bytes'
        """
        import time

        start_total = time.time()
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
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        model.train()
        epoch_losses = []
        fairness_enabled = bool(self.cfg.get("fairness", {}).get("enabled", False))
        lambda_fair = float(self.cfg.get("fairness", {}).get("lambda_fairness", 0.1))
        batch_counter = 0

        for _epoch in range(self.local_epochs):
            batch_losses = []
            for batch in loader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                weights = batch.get("weight")

                optimizer.zero_grad()
                preds = model(features).squeeze(-1)

                # Base BCE loss (optionally per-sample weighted for fairness reweighting)
                if weights is not None:
                    loss = (criterion(preds, labels) * weights.to(self.device)).mean()
                else:
                    loss = criterion(preds, labels)

                # Ω_fair: add fairness penalty every 5 batches
                if fairness_enabled and batch_counter % 5 == 0:
                    try:
                        batch_group_labels = self._get_batch_group_labels(batch)
                        if batch_group_labels is not None:
                            # Verify alignment: same number of labels as batch samples
                            assert len(batch_group_labels) == len(batch["label"]), (
                                f"Group label count ({len(batch_group_labels)}) != "
                                f"batch size ({len(batch['label'])})"
                            )

                            y_true_np = labels.detach().cpu().numpy().astype(int)
                            y_pred_np = (preds.detach().cpu().numpy() >= 0.5).astype(int)
                            group_f1s = group_performance_from_predictions(
                                y_true_np, y_pred_np, batch_group_labels
                            )
                            if len(group_f1s) >= 2:
                                # Differentiable surrogate: difference in mean predicted probabilities
                                group_means = {}
                                for g in group_f1s.keys():
                                    mask = batch_group_labels == g
                                    if mask.any():
                                        group_means[g] = preds[mask].mean()

                                if len(group_means) >= 2:
                                    vals = torch.stack(list(group_means.values()))
                                    diff = vals.max() - vals.min()
                                    fairness_penalty_value = float(diff.item())
                                    loss = loss + lambda_fair * diff

                                    # Log fairness penalty magnitude every 50 batches
                                    if batch_counter % 50 == 0:
                                        self.logger.debug(
                                            "Fairness penalty applied",
                                            node_id=self.node_id,
                                            batch=batch_counter,
                                            n_groups_in_batch=len(group_means),
                                            fairness_penalty=round(fairness_penalty_value, 6),
                                            lambda_fair=lambda_fair,
                                        )
                    except (Exception, AssertionError) as e:
                        # Never let fairness penalty crash training
                        self.logger.debug("Fairness penalty skipped", error=str(e))

                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))
                batch_counter += 1

            epoch_loss = float(np.mean(batch_losses))
            epoch_losses.append(epoch_loss)

        time_training = time.time() - start_total

        final_state = model.get_state_dict_numpy()

        # Compute update delta: Δw_k = w_k^t - w_global^t
        delta: dict[str, np.ndarray] = {
            name: final_state[name] - initial_state[name] for name in final_state
        }

        # Apply differential privacy protection to the DELTA (clipping + noise)
        if self.privacy_enabled:
            protected_delta = protect_state_dict(delta, C=self.C, sigma=self.sigma, seed=seed)
        else:
            self.logger.debug("Privacy disabled, skipping clipping and noise")
            protected_delta = delta

        # Reconstruct DP-protected absolute weights for communication to server
        # w_k_protected = w_global + Δw̃_k
        protected_absolute_weights: dict[str, np.ndarray] = {
            name: initial_state[name] + protected_delta[name] for name in initial_state
        }

        # Submit hash to blockchain (ONLY the hash of the PROTECTED DELTA)
        flat_protected_delta = np.concatenate([v.flatten() for v in protected_delta.values()])

        start_bc = time.time()
        self.blockchain.submit_model_update_hash(
            protected_update=flat_protected_delta,
            node_id=self.node_id,
            round_number=round_number,
            consent_ref=self.consent_ref,
            policy_ref=self.policy_ref,
        )
        time_blockchain = time.time() - start_bc

        # Evaluate local model
        metrics = self._evaluate_local(model)
        metrics["train_loss"] = float(np.mean(epoch_losses))

        # Systems overhead metrics
        metrics["time_local_training"] = round(time_training, 4)
        metrics["time_blockchain"] = round(time_blockchain, 4)

        # Model size calculation (approximate bytes of float32 weights)
        model_size_bytes = sum(v.nbytes for v in protected_absolute_weights.values())
        metrics["model_size_bytes"] = int(model_size_bytes)

        self.logger.info(
            "Local training round complete",
            node_id=self.node_id,
            round_number=round_number,
            local_epochs=self.local_epochs,
            n_samples=len(self.train_dataset),
            train_loss=round(metrics["train_loss"], 4),
            f1=round(metrics.get("f1", 0.0), 4),
            model_size_kb=round(model_size_bytes / 1024, 2),
        )

        return protected_absolute_weights, len(self.train_dataset), metrics

    def _evaluate_local(self, model: EmploymentMatchingModel) -> dict[str, float]:
        """Evaluate the locally trained model on the held-out test dataset.

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
        loader = DataLoader(self.test_dataset, batch_size=256, shuffle=False)

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
            group_labels = self.test_dataset.get_group_labels("disability_category")
            group_f1s = group_performance_from_predictions(y_true, y_pred, group_labels)
            metrics["min_group_f1"] = float(min(group_f1s.values())) if group_f1s else 0.0
            metrics["fairness_disparity_disability"] = (
                float(max(group_f1s.values()) - min(group_f1s.values()))
                if len(group_f1s) >= 2
                else 0.0
            )
        except Exception:
            metrics["min_group_f1"] = 0.0
            metrics["fairness_disparity_disability"] = 0.0

        return metrics

    def _get_batch_group_labels(self, batch: dict) -> np.ndarray | None:
        """Return disability_category group labels for the exact samples in this batch.

        Uses the dataset index stored in batch['idx'] (added in EmploymentDataset.__getitem__)
        to look up the correct group labels for each sample in the shuffled batch.

        Returns None gracefully if group labels are unavailable.
        """
        if not hasattr(self, "_group_label_cache"):
            try:
                self._group_label_cache = self.train_dataset.get_group_labels("disability_category")
            except Exception:
                self.logger.warning(
                    "Failed to build group label cache for fairness penalty",
                    node_id=self.node_id,
                )
                self._group_label_cache = None  # type: ignore[assignment]

        if self._group_label_cache is None:
            return None

        batch_indices = batch.get("idx")
        if batch_indices is None:
            self.logger.warning(
                "Batch does not contain 'idx' key — fairness penalty cannot be applied. "
                "Ensure EmploymentDataset.__getitem__ returns 'idx'.",
                node_id=self.node_id,
            )
            return None

        indices = batch_indices.cpu().numpy().astype(int)

        # Bounds check
        max_idx = len(self._group_label_cache) - 1
        if np.any(indices > max_idx) or np.any(indices < 0):
            self.logger.error(
                "Batch indices out of bounds for group label cache",
                max_cache_idx=max_idx,
                batch_idx_range=(int(indices.min()), int(indices.max())),
                node_id=self.node_id,
            )
            return None

        return self._group_label_cache[indices]
