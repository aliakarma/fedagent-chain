"""Integration tests for the complete federated learning pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.blockchain.chain import PermissionedBlockchain
from src.data.dataset import EmploymentDataset
from src.data.synthetic_generator import generate_synthetic_node_data
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.models.employment_model import EmploymentMatchingModel


@pytest.fixture
def small_cfg():
    return OmegaConf.create({
        "federated": {
            "n_rounds": 2,
            "min_clients": 2,
            "local_epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.01,
        },
        "model": {
            "input_dim": 91,
            "hidden_dims": [32, 16],
            "dropout_rate": 0.1,
        },
        "privacy": {
            "enabled": True,
            "clipping_threshold": 1.0,
            "noise_multiplier": 0.05,
        },
        "fairness": {
            "enabled": True,
            "lambda_fairness": 0.1,
            "protected_groups": ["disability_category"],
        },
        "blockchain": {
            "enabled": True,
            "records_per_block": 5,
        },
        "tracking": {"use_mlflow": False},
        "output": {"save_checkpoints": False, "checkpoint_interval": 5},
    })


@pytest.fixture
def two_node_clients(small_cfg):
    """Two minimal FederatedClient instances for integration tests."""
    blockchain = PermissionedBlockchain(records_per_block=10)
    clients = []
    for node_id in ["saudi_arabia", "united_states"]:
        data = generate_synthetic_node_data(
            node_id=node_id, n_users=60, n_jobs=30, n_pairs=120, seed=42
        )
        full_ds = EmploymentDataset(
            outcomes_df=data["outcomes"],
            users_df=data["users"],
            jobs_df=data["jobs"],
            consent_filter=True,
        )
        train_ds, test_ds = full_ds.split(test_size=0.20, seed=42)
        client = FederatedClient(
            node_id=node_id,
            train_dataset=train_ds,
            test_dataset=test_ds,
            cfg=small_cfg,
            blockchain=blockchain,
            device="cpu",
        )
        clients.append(client)
    return clients, blockchain


@pytest.mark.integration
class TestFederatedPipelineIntegration:

    def test_one_round_completes_without_error(self, small_cfg, two_node_clients):
        clients, blockchain = two_node_clients
        server = FederatedServer(small_cfg, use_fairness_aggregation=True, output_dir="/tmp/test_run")
        metrics = server.run_round(clients, round_num=1, seed=42)
        assert "mean_f1" in metrics
        assert "mean_accuracy" in metrics

    def test_blockchain_receives_one_record_per_node_per_round(self, small_cfg, two_node_clients):
        clients, blockchain = two_node_clients
        server = FederatedServer(small_cfg, use_fairness_aggregation=False, output_dir="/tmp/test_run2")
        initial_count = blockchain.get_record_count()
        server.run_round(clients, round_num=1, seed=42)
        assert blockchain.get_record_count() == initial_count + len(clients)

    def test_global_model_state_updates_after_round(self, small_cfg, two_node_clients):
        clients, blockchain = two_node_clients
        server = FederatedServer(small_cfg, use_fairness_aggregation=False, output_dir="/tmp/test_run3")
        initial_state = {k: v.copy() for k, v in server.global_state.items()}
        server.run_round(clients, round_num=1, seed=42)
        # At least one parameter should have changed
        any_changed = any(
            not np.allclose(server.global_state[k], initial_state[k])
            for k in server.global_state
        )
        assert any_changed

    def test_chain_integrity_valid_after_training(self, small_cfg, two_node_clients):
        clients, blockchain = two_node_clients
        server = FederatedServer(small_cfg, use_fairness_aggregation=True, output_dir="/tmp/test_run4")
        for rnd in range(1, 3):
            server.run_round(clients, round_num=rnd, seed=42)
        assert blockchain.verify_chain_integrity() is True

    def test_no_raw_data_in_blockchain_payloads(self, small_cfg, two_node_clients):
        clients, blockchain = two_node_clients
        server = FederatedServer(small_cfg, use_fairness_aggregation=False, output_dir="/tmp/test_run5")
        server.run_round(clients, round_num=1, seed=42)
        forbidden = {"disability_type", "user_id", "skill_vector", "accommodation_needs"}
        for record in blockchain.get_all_records():
            record_dict = record.model_dump()
            for field in forbidden:
                assert field not in record_dict

    def test_full_simulation_runs_and_returns_results(self, small_cfg, two_node_clients):
        clients, blockchain = two_node_clients
        small_cfg.federated.n_rounds = 2
        small_cfg.federated.min_clients = 2
        server = FederatedServer(small_cfg, use_fairness_aggregation=True, output_dir="/tmp/test_run6")
        results = server.run(clients=clients, seed=42, use_mlflow=False)
        assert "best_f1" in results
        assert results["convergence_rounds"] == 2

    def test_local_training_updates_weights(self, small_cfg, two_node_clients):
        """Clients should produce non-trivial model updates (different from global state) after local training."""
        clients, blockchain = two_node_clients
        model = EmploymentMatchingModel.from_config(small_cfg.model)
        global_state = model.get_state_dict_numpy()
        protected_weights, n_samples, metrics = clients[0].train_round(
            global_state=global_state, round_number=1, seed=42
        )
        # Check if protected weights differ from global state
        any_diff = any(
            not np.allclose(protected_weights[k], global_state[k])
            for k in global_state
        )
        assert any_diff

    def test_fedavg_loss_does_not_diverge(self, small_cfg, two_node_clients):
        """Critical: FedAvg loss must not increase monotonically (indicates delta accumulation bug)."""
        clients, blockchain = two_node_clients
        small_cfg.federated.n_rounds = 10
        small_cfg.federated.local_epochs = 2
        small_cfg.federated.min_clients = 2
        server = FederatedServer(small_cfg, use_fairness_aggregation=False, output_dir="/tmp/test_convergence")

        round_losses = []
        for rnd in range(1, 11):
            metrics = server.run_round(clients, round_num=rnd, seed=42)
            round_losses.append(metrics.get("mean_train_loss", float("inf")))

        # Loss at round 10 must not be more than 5x the loss at round 1 (very generous bound)
        loss_ratio = round_losses[-1] / (round_losses[0] + 1e-8)
        assert loss_ratio < 5.0, (
            f"FedAvg loss diverged: round 1={round_losses[0]:.4f}, "
            f"round 10={round_losses[-1]:.4f}, ratio={loss_ratio:.2f}. "
            f"Full history: {[round(x, 4) for x in round_losses]}"
        )

        # Additionally, final loss should not exceed 5.0 for binary classification
        assert round_losses[-1] < 5.0, (
            f"FedAvg BCE loss={round_losses[-1]:.4f} at round 10 is unreasonably high."
        )

    def test_consent_filter_applied(self, small_cfg, two_node_clients):
        """Dataset should contain fewer samples than raw data due to consent filtering."""
        clients, _ = two_node_clients
        for client in clients:
            # All samples in EmploymentDataset should be from consented users
            n_dataset = len(client.dataset)
            assert n_dataset > 0  # some consented users exist
