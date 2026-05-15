"""Regression tests for Phase 0 fixes."""
import pytest
from omegaconf import OmegaConf
from src.data.dataset import encode_user_job_pair
import pandas as pd
import numpy as np


def test_client_reads_local_epochs_from_federated_subkey():
    from src.blockchain.chain import PermissionedBlockchain
    from src.data.dataset import EmploymentDataset
    from src.federated.client import FederatedClient

    cfg = OmegaConf.create({
        "federated": {
            "local_epochs": 7,      # non-default value
            "batch_size": 32,
            "learning_rate": 0.005,
        },
        "privacy":   {"clipping_threshold": 2.0, "noise_multiplier": 0.2},
        "model":     {"input_dim": 91, "hidden_dims": [32, 16], "dropout_rate": 0.1},
        "fairness":  {"enabled": False, "lambda_fairness": 0.0, "protected_groups": []},
        "blockchain":{"enabled": False, "records_per_block": 5},
    })
    from src.data.synthetic_generator import generate_synthetic_node_data
    data = generate_synthetic_node_data("saudi_arabia", 30, 15, 60, seed=42)
    ds = EmploymentDataset(data["outcomes"], data["users"], data["jobs"])
    bc = PermissionedBlockchain(records_per_block=5)
    client = FederatedClient("saudi_arabia", ds, ds, cfg, bc, device="cpu")

    assert client.local_epochs == 7,   "local_epochs not read from federated sub-key"
    assert client.batch_size   == 32,  "batch_size not read from federated sub-key"
    assert abs(client.learning_rate - 0.005) < 1e-9


def test_education_ohe_is_proper_one_hot():
    user = pd.Series({
        "skill_vector":       str([1, 0] * 25),
        "accommodation_needs": str([1, 0] * 10),
        "disability_category": "mobility",
        "preferred_work_mode": "hybrid",
        "education_level":     3,           # UNDERGRADUATE
        "employment_goal":     "fulltime",
        "language_primary":    "ar",
    })
    job = pd.Series({
        "required_skills":      str([1, 0] * 25),
        "accommodation_provided": str([1, 0] * 10),
        "work_mode":            "hybrid",
        "language_required":    "ar",
    })
    features = encode_user_job_pair(user, job)
    # Education dims occupy indices 81–85 (50+20+8+3 = 81 start)
    edu_slice = features[81:86]
    assert edu_slice[3] == 1.0, "Expected one-hot at position 3 for edu_level=3"
    assert sum(edu_slice) == pytest.approx(1.0), "Education OHE must sum to 1"


def test_centralized_dataset_pooling():
    from src.data.dataset import EmploymentDataset
    from scripts.run_federated_simulation import pool_node_datasets
    
    # Create mock data for node A
    outcomes_a = pd.DataFrame({"user_id": ["u1"], "job_id": ["j1"], "suitability_label": [1]})
    users_a = pd.DataFrame({"user_id": ["u1"], "attr": [0]}).set_index("user_id")
    jobs_a = pd.DataFrame({"job_id": ["j1"], "attr": [0]}).set_index("job_id")
    ds_a = EmploymentDataset(outcomes_a, users_a, jobs_a)
    
    # Create mock data for node B
    outcomes_b = pd.DataFrame({"user_id": ["u2"], "job_id": ["j2"], "suitability_label": [0]})
    users_b = pd.DataFrame({"user_id": ["u2"], "attr": [1]}).set_index("user_id")
    jobs_b = pd.DataFrame({"job_id": ["j2"], "attr": [1]}).set_index("job_id")
    ds_b = EmploymentDataset(outcomes_b, users_b, jobs_b)
    
    # Pool them (using same DS for train/test for simplicity in mock)
    pooled_train, pooled_test = pool_node_datasets([ds_a, ds_b], [ds_a, ds_b])
    
    assert len(pooled_train) == 2, "Pooled training set should have 2 samples"
    assert len(pooled_test) == 2, "Pooled test set should have 2 samples"
    assert set(pooled_train.outcomes["user_id"]) == {"u1", "u2"}
    assert len(pooled_train.users_df) == 2
    assert len(pooled_train.jobs_df) == 2
