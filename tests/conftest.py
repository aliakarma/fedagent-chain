"""Shared pytest fixtures for FedAgent-Chain test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.blockchain.chain import PermissionedBlockchain
from src.data.schema import (
    DisabilityCategory, EducationLevel, EmploymentGoal,
    NodeID, UserProfile, WorkMode,
)
from src.data.synthetic_generator import generate_synthetic_node_data
from src.utils.seed_utils import set_global_seed


@pytest.fixture(autouse=True)
def reset_seed():
    """Reset global seed before each test for reproducibility."""
    set_global_seed(42)
    yield


@pytest.fixture
def minimal_cfg():
    """Minimal OmegaConf config for unit tests."""
    return OmegaConf.create({
        "federated": {
            "n_rounds": 3,
            "min_clients": 2,
            "local_epochs": 2,
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
            "noise_multiplier": 0.1,
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
        "agents": {
            "employment": {
                "alpha": 0.40,
                "beta": 0.25,
                "gamma": 0.20,
                "delta": 0.15,
                "top_k": 5,
            },
            "governance": {
                "review_threshold": 0.70,
            },
        },
    })


@pytest.fixture
def sample_user_profile():
    """A single synthetic user profile for agent tests."""
    return UserProfile(
        user_id="test-user-001",
        node_id=NodeID.SAUDI_ARABIA,
        skill_vector=[1, 0] * 25,
        education_level=EducationLevel.UNDERGRADUATE,
        disability_category=DisabilityCategory.MOBILITY,
        accommodation_needs=[1, 0] * 10,
        language_primary="ar",
        language_secondary="en",
        preferred_work_mode=WorkMode.HYBRID,
        employment_goal=EmploymentGoal.FULLTIME,
        consent_given=True,
    )


@pytest.fixture
def synthetic_node_data():
    """Small synthetic dataset (100 users, 50 jobs) for integration tests."""
    return generate_synthetic_node_data(
        node_id="saudi_arabia",
        n_users=100,
        n_jobs=50,
        n_pairs=200,
        seed=42,
    )


@pytest.fixture
def blockchain():
    """Fresh in-memory blockchain for each test."""
    return PermissionedBlockchain(records_per_block=5)


@pytest.fixture
def sample_state_dict():
    """A minimal numpy model state dict for aggregation tests."""
    rng = np.random.default_rng(42)
    return {
        "layer1.weight": rng.standard_normal((16, 91)).astype(np.float32),
        "layer1.bias": rng.standard_normal(16).astype(np.float32),
        "output.weight": rng.standard_normal((1, 16)).astype(np.float32),
        "output.bias": rng.standard_normal(1).astype(np.float32),
    }
