"""Data loading, processing, and management for FedAgent-Chain."""

from src.data.schema import (
    BlockchainRecord,
    DisabilityCategory,
    EducationLevel,
    EmploymentGoal,
    EmploymentOutcome,
    JobProfile,
    NodeID,
    UserProfile,
    WorkMode,
)
from src.data.synthetic_generator import (
    generate_synthetic_node_data,
    generate_user_profiles,
    generate_job_profiles,
    save_synthetic_dataset,
)

__all__ = [
    "NodeID",
    "DisabilityCategory",
    "WorkMode",
    "EmploymentGoal",
    "EducationLevel",
    "UserProfile",
    "JobProfile",
    "EmploymentOutcome",
    "BlockchainRecord",
    "generate_synthetic_node_data",
    "generate_user_profiles",
    "generate_job_profiles",
    "save_synthetic_dataset",
]
