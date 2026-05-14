"""PyTorch Dataset and feature encoding for FedAgent-Chain employment records."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.schema import (
    DisabilityCategory,
    EducationLevel,
    EmploymentGoal,
    NodeID,
    WorkMode,
)


# One-hot encoding mappings
DISABILITY_CATEGORIES = [c.value for c in DisabilityCategory]
WORK_MODES = [m.value for m in WorkMode]
EMPLOYMENT_GOALS = [g.value for g in EmploymentGoal]
NODE_IDS = [n.value for n in NodeID]

# Language codes seen in the synthetic dataset
KNOWN_LANGUAGES = ["ar", "en", "zh", "fr", "de", "es", "ur", "tl", "yue"]


def encode_categorical(value: str, categories: List[str]) -> List[float]:
    """One-hot encode a categorical value.

    Parameters
    ----------
    value : str
        Categorical value to encode.
    categories : list of str
        Ordered list of all possible categories.

    Returns
    -------
    list of float
        One-hot encoded vector of length len(categories).
    """
    vec = [0.0] * len(categories)
    if value in categories:
        vec[categories.index(value)] = 1.0
    return vec


def encode_user_job_pair(
    user_row: pd.Series,
    job_row: pd.Series,
) -> np.ndarray:
    """Encode a user-job pair into a fixed-dimension feature vector.

    Feature layout (total = 91 dims):
    - skill_overlap      : 50 dims (element-wise product of skill vectors)
    - accom_coverage     : 20 dims (element-wise min of accommodation vectors)
    - disability_ohe     : 8  dims (one-hot disability category)
    - work_mode_ohe      : 3  dims (one-hot work mode match)
    - education_feat     : 5  dims (one-hot education level)
    - employment_goal    : 4  dims (one-hot employment goal)
    - language_match     : 1  dim  (binary language match indicator)

    Total: 50 + 20 + 8 + 3 + 5 + 4 + 1 = 91 dims

    Parameters
    ----------
    user_row : pd.Series
        A row from the users DataFrame.
    job_row : pd.Series
        A row from the jobs DataFrame.

    Returns
    -------
    np.ndarray
        Feature vector of shape (91,).
    """
    # Parse list columns (stored as strings in CSV)
    def parse_list(val: object) -> List[int]:
        if isinstance(val, list):
            return [int(x) for x in val]
        if isinstance(val, str):
            import ast
            return [int(x) for x in ast.literal_eval(val)]
        return []

    u_skills = np.array(parse_list(user_row["skill_vector"]), dtype=float)
    j_skills = np.array(parse_list(job_row["required_skills"]), dtype=float)
    u_accom = np.array(parse_list(user_row["accommodation_needs"]), dtype=float)
    j_accom = np.array(parse_list(job_row["accommodation_provided"]), dtype=float)

    skill_overlap = np.minimum(u_skills, j_skills)
    accom_coverage = np.minimum(u_accom, j_accom)

    disability_ohe = encode_categorical(
        str(user_row.get("disability_category", "")), DISABILITY_CATEGORIES
    )
    work_mode_ohe = encode_categorical(
        str(user_row.get("preferred_work_mode", "")), WORK_MODES
    )
    edu_level = int(user_row.get("education_level", 0))
    education_ohe = [0.0] * 5
    if 0 <= edu_level <= 4:
        education_ohe[edu_level] = 1.0

    goal_ohe = encode_categorical(
        str(user_row.get("employment_goal", "")), EMPLOYMENT_GOALS
    )

    lang_match = [
        1.0 if user_row.get("language_primary") == job_row.get("language_required") else 0.0
    ]

    feature_vector = np.concatenate([
        skill_overlap,      # 50
        accom_coverage,     # 20
        disability_ohe,     # 8
        work_mode_ohe,      # 3
        education_ohe,      # 5
        goal_ohe,           # 4
        lang_match,         # 1
    ])

    return feature_vector.astype(np.float32)


class EmploymentDataset(Dataset):
    """PyTorch Dataset for disability-employment suitability records.

    Loads user-job pairs and their suitability labels for local node training.

    Parameters
    ----------
    outcomes_df : pd.DataFrame
        Suitability pair labels with user_id, job_id, suitability_label columns.
    users_df : pd.DataFrame
        User profile records indexed by user_id.
    jobs_df : pd.DataFrame
        Job profile records indexed by job_id.
    consent_filter : bool
        If True, exclude records from users who have not given consent.
    sample_weights : np.ndarray, optional
        Per-sample importance weights for fairness reweighting.
    """

    def __init__(
        self,
        outcomes_df: pd.DataFrame,
        users_df: pd.DataFrame,
        jobs_df: pd.DataFrame,
        consent_filter: bool = True,
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        # Apply consent filter (privacy-preserving pipeline requirement)
        if consent_filter and "consent_given" in users_df.columns:
            consented_ids = set(users_df[users_df["consent_given"]]["user_id"].tolist())
            outcomes_df = outcomes_df[outcomes_df["user_id"].isin(consented_ids)].copy()

        self.users_df = users_df.set_index("user_id") if "user_id" in users_df.columns else users_df
        self.jobs_df = jobs_df.set_index("job_id") if "job_id" in jobs_df.columns else jobs_df
        self.outcomes = outcomes_df.reset_index(drop=True)
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return len(self.outcomes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.outcomes.iloc[idx]
        user_id = row["user_id"]
        job_id = row["job_id"]
        label = int(row["suitability_label"])

        user_row = self.users_df.loc[user_id]
        job_row = self.jobs_df.loc[job_id]

        features = encode_user_job_pair(user_row, job_row)
        item: Dict[str, torch.Tensor] = {
            "features": torch.from_numpy(features),
            "label": torch.tensor(label, dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.long),
        }

        if self.sample_weights is not None:
            item["weight"] = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32)

        return item

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension of a single encoded sample."""
        if len(self) == 0:
            return 91
        sample = self[0]
        return int(sample["features"].shape[0])

    def get_labels(self) -> np.ndarray:
        """Return all suitability labels as a numpy array."""
        return self.outcomes["suitability_label"].values.astype(int)

    def get_group_labels(self, attribute: str) -> np.ndarray:
        """Return group membership labels for a protected attribute.

        Parameters
        ----------
        attribute : str
            Protected attribute column name (e.g., 'disability_category').

        Returns
        -------
        np.ndarray
            Group label for each sample in the dataset.
        """
        user_ids = self.outcomes["user_id"].values
        return np.array([
            self.users_df.loc[uid, attribute] if uid in self.users_df.index else "unknown"
            for uid in user_ids
        ])
    def split(
        self,
        test_size: float = 0.20,
        seed: int = 42,
    ) -> tuple["EmploymentDataset", "EmploymentDataset"]:
        """Return (train_dataset, test_dataset) with stratified split on label.

        Parameters
        ----------
        test_size : float
            Fraction of pairs to hold out for testing. Default 0.20 (20 %).
        seed : int
            Random seed for reproducible splits.

        Returns
        -------
        tuple of (EmploymentDataset, EmploymentDataset)
            train_dataset, test_dataset
        """
        from sklearn.model_selection import train_test_split

        labels = self.outcomes["suitability_label"].values
        idx = np.arange(len(self.outcomes))

        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )

        train_outcomes = self.outcomes.iloc[train_idx].reset_index(drop=True)
        test_outcomes = self.outcomes.iloc[test_idx].reset_index(drop=True)

        # Restore DataFrame columns from index so downstream code works
        users_df_reset = self.users_df.reset_index()
        jobs_df_reset = self.jobs_df.reset_index()

        train_weights = (
            self.sample_weights[train_idx]
            if self.sample_weights is not None else None
        )

        train_ds = EmploymentDataset(
            outcomes_df=train_outcomes,
            users_df=users_df_reset,
            jobs_df=jobs_df_reset,
            consent_filter=False,   # Already filtered upstream
            sample_weights=train_weights,
        )
        test_ds = EmploymentDataset(
            outcomes_df=test_outcomes,
            users_df=users_df_reset,
            jobs_df=jobs_df_reset,
            consent_filter=False,
            sample_weights=None,    # Never weight the test set
        )
        return train_ds, test_ds
