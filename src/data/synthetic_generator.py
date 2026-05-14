"""Synthetic disability-employment data generator for FedAgent-Chain.

This module generates realistic synthetic data for prototype evaluation.
All generated records are fictitious and contain no real personal information.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.schema import (
    DisabilityCategory,
    EducationLevel,
    EmploymentGoal,
    EmploymentOutcome,
    JobProfile,
    NodeID,
    UserProfile,
    WorkMode,
)
from src.utils.logging_utils import get_logger
from src.utils.seed_utils import get_rng

logger = get_logger("SyntheticGenerator")

# Cross-country language distribution (primary languages per node)
NODE_LANGUAGE_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "saudi_arabia": {"ar": 0.75, "en": 0.15, "ur": 0.07, "tl": 0.03},
    "united_states": {"en": 0.80, "es": 0.12, "zh": 0.04, "fr": 0.04},
    "china": {"zh": 0.85, "en": 0.10, "yue": 0.05},
    "europe": {"en": 0.35, "de": 0.25, "fr": 0.20, "es": 0.20},
}

# Disability category distribution (calibrated against WHO WRD statistics)
NODE_DISABILITY_DISTRIBUTION: Dict[str, Dict[str, float]] = {
    "saudi_arabia": {
        "mobility": 0.22, "vision": 0.15, "hearing": 0.14,
        "cognitive": 0.12, "communication": 0.08,
        "mental_health": 0.10, "chronic_health": 0.12, "multiple": 0.07,
    },
    "united_states": {
        "mobility": 0.18, "vision": 0.12, "hearing": 0.13,
        "cognitive": 0.15, "communication": 0.10,
        "mental_health": 0.15, "chronic_health": 0.12, "multiple": 0.05,
    },
    "china": {
        "mobility": 0.25, "vision": 0.17, "hearing": 0.18,
        "cognitive": 0.10, "communication": 0.07,
        "mental_health": 0.08, "chronic_health": 0.10, "multiple": 0.05,
    },
    "europe": {
        "mobility": 0.20, "vision": 0.13, "hearing": 0.13,
        "cognitive": 0.14, "communication": 0.09,
        "mental_health": 0.14, "chronic_health": 0.12, "multiple": 0.05,
    },
}

# Sectors available per node
SECTORS = [
    "technology", "healthcare", "education", "finance", "retail",
    "manufacturing", "government", "nonprofit", "logistics", "creative",
]


def _sample_categorical(
    rng: np.random.Generator,
    categories: List[str],
    probabilities: List[float],
    n: int,
) -> List[str]:
    """Sample n items from categories with given probabilities."""
    return rng.choice(categories, size=n, p=probabilities).tolist()


def generate_user_profiles(
    node_id: str,
    n_users: int,
    rng: np.random.Generator,
) -> List[UserProfile]:
    """Generate synthetic user profiles for a single regional node.

    Parameters
    ----------
    node_id : str
        Regional node identifier.
    n_users : int
        Number of user profiles to generate.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    list of UserProfile
        List of validated user profile records.
    """
    lang_dist = NODE_LANGUAGE_DISTRIBUTION[node_id]
    dis_dist = NODE_DISABILITY_DISTRIBUTION[node_id]

    lang_keys = list(lang_dist.keys())
    lang_probs = list(lang_dist.values())

    dis_keys = list(dis_dist.keys())
    dis_probs = list(dis_dist.values())

    profiles = []
    for _ in range(n_users):
        # Generate skill vector: correlated with disability and education level
        skill_vector = rng.integers(0, 2, size=50).tolist()

        # Generate accommodation needs (correlated with disability category)
        accommodation_needs = rng.integers(0, 2, size=20).tolist()

        primary_lang = _sample_categorical(rng, lang_keys, lang_probs, 1)[0]
        has_secondary = rng.random() < 0.25
        secondary_lang = None
        if has_secondary:
            other_langs = [l for l in lang_keys if l != primary_lang]
            if other_langs:
                secondary_lang = rng.choice(other_langs)

        profile = UserProfile(
            user_id=str(uuid.UUID(bytes=rng.bytes(16), version=4)),
            node_id=NodeID(node_id),
            skill_vector=skill_vector,
            education_level=EducationLevel(int(rng.integers(0, 5))),
            disability_category=DisabilityCategory(
                _sample_categorical(rng, dis_keys, dis_probs, 1)[0]
            ),
            accommodation_needs=accommodation_needs,
            language_primary=primary_lang,
            language_secondary=secondary_lang,
            preferred_work_mode=WorkMode(rng.choice(["onsite", "hybrid", "remote"])),
            employment_goal=EmploymentGoal(
                rng.choice(["fulltime", "parttime", "freelance", "internship"])
            ),
            consent_given=bool(rng.random() < 0.92),  # 92% consent rate
        )
        profiles.append(profile)

    logger.info("Generated user profiles", node_id=node_id, n_users=n_users)
    return profiles


def generate_job_profiles(
    node_id: str,
    n_jobs: int,
    rng: np.random.Generator,
) -> List[JobProfile]:
    """Generate synthetic job profiles for a regional node.

    Parameters
    ----------
    node_id : str
        Regional node identifier.
    n_jobs : int
        Number of job profiles to generate.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    list of JobProfile
        List of validated job profile records.
    """
    lang_dist = NODE_LANGUAGE_DISTRIBUTION[node_id]
    lang_keys = list(lang_dist.keys())
    lang_probs = list(lang_dist.values())

    jobs = []
    for _ in range(n_jobs):
        jobs.append(
            JobProfile(
                job_id=str(uuid.UUID(bytes=rng.bytes(16), version=4)),
                node_id=NodeID(node_id),
                required_skills=rng.integers(0, 2, size=50).tolist(),
                accessibility_score=float(rng.beta(3, 2)),  # Skewed towards accessible
                work_mode=WorkMode(rng.choice(["onsite", "hybrid", "remote"])),
                language_required=_sample_categorical(rng, lang_keys, lang_probs, 1)[0],
                education_minimum=EducationLevel(int(rng.integers(0, 5))),
                accommodation_provided=rng.integers(0, 2, size=20).tolist(),
                sector=str(rng.choice(SECTORS)),
            )
        )

    logger.info("Generated job profiles", node_id=node_id, n_jobs=n_jobs)
    return jobs


def compute_suitability_label(
    user: UserProfile,
    job: JobProfile,
    threshold: float = 0.50,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, float]:
    """Compute a deterministic suitability label for a user-job pair.

    The suitability score is computed using a rule-based function that considers
    skill overlap, accommodation compatibility, and language match.

    Parameters
    ----------
    user : UserProfile
        User profile record.
    job : JobProfile
        Job profile record.
    threshold : float
        Score threshold above which the pair is labeled positive (suitable).
    rng : np.random.Generator, optional
        If provided, adds small Gaussian noise for label diversity.

    Returns
    -------
    tuple of (int, float)
        Binary label (0 or 1) and continuous suitability score.
    """
    user_skills = np.array(user.skill_vector, dtype=float)
    job_skills = np.array(job.required_skills, dtype=float)

    # Skill overlap component (weighted Jaccard similarity)
    intersection = np.sum(np.minimum(user_skills, job_skills))
    union = np.sum(np.maximum(user_skills, job_skills))
    skill_overlap = float(intersection / (union + 1e-8))

    # Accommodation compatibility
    user_needs = np.array(user.accommodation_needs, dtype=float)
    job_provides = np.array(job.accommodation_provided, dtype=float)
    accom_coverage = float(
        np.sum(np.minimum(user_needs, job_provides)) / (np.sum(user_needs) + 1e-8)
    )

    # Language match
    lang_match = 1.0 if user.language_primary == job.language_required else 0.3

    # Work mode preference match
    mode_match = 1.0 if user.preferred_work_mode == job.work_mode else 0.5

    # Label-generating oracle uses deliberately different coefficients
    # from the model's scoring function (α=0.40 / β=0.25 / γ=0.20 / δ=0.15)
    # to prevent artificial separability.
    score = (
        0.35 * skill_overlap
        + 0.30 * accom_coverage
        + 0.20 * lang_match
        + 0.15 * mode_match
    )

    # Add noise for label diversity
    if rng is not None:
        score += float(rng.normal(0, 0.08))  # wider noise → more realistic difficulty
        score = float(np.clip(score, 0.0, 1.0))

    label = 1 if score >= threshold else 0
    return label, score


def generate_synthetic_node_data(
    node_id: str,
    n_users: int = 2500,
    n_jobs: int = 1250,
    n_pairs: int = 12500,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generate a complete synthetic dataset for a single node.

    Parameters
    ----------
    node_id : str
        Regional node identifier.
    n_users : int
        Number of user profiles.
    n_jobs : int
        Number of job profiles.
    n_pairs : int
        Number of user-job suitability pairs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict of str -> pd.DataFrame
        Dictionary with keys 'users', 'jobs', 'outcomes'.
    """
    rng = get_rng(seed)

    users = generate_user_profiles(node_id, n_users, rng)
    jobs = generate_job_profiles(node_id, n_jobs, rng)

    # Filter to consented users only
    consented_users = [u for u in users if u.consent_given]

    # Generate suitability pairs
    outcomes = []
    user_ids = [u.user_id for u in consented_users]
    job_ids = [j.job_id for j in jobs]

    sampled_user_ids = rng.choice(user_ids, size=n_pairs, replace=True).tolist()
    sampled_job_ids = rng.choice(job_ids, size=n_pairs, replace=True).tolist()

    user_map = {u.user_id: u for u in consented_users}
    job_map = {j.job_id: j for j in jobs}

    for uid, jid in zip(sampled_user_ids, sampled_job_ids):
        label, score = compute_suitability_label(user_map[uid], job_map[jid], rng=rng)
        outcomes.append(
            EmploymentOutcome(
                user_id=uid,
                job_id=jid,
                node_id=NodeID(node_id),
                suitability_label=label,
                suitability_score=score,
            )
        )

    users_df = pd.DataFrame([u.model_dump(mode="json") for u in users])
    jobs_df = pd.DataFrame([j.model_dump(mode="json") for j in jobs])
    outcomes_df = pd.DataFrame([o.model_dump(mode="json") for o in outcomes])

    logger.info(
        "Node data generated",
        node_id=node_id,
        n_users=len(users),
        n_jobs=len(jobs),
        n_pairs=len(outcomes),
        positive_rate=float(outcomes_df["suitability_label"].mean()),
    )

    return {"users": users_df, "jobs": jobs_df, "outcomes": outcomes_df}


def save_synthetic_dataset(
    output_dir: str | Path,
    n_users_per_node: int = 2500,
    n_jobs_per_node: int = 1250,
    n_pairs_per_node: int = 12500,
    seed: int = 42,
) -> None:
    """Generate and save the complete synthetic dataset for all four nodes.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save the generated CSV files.
    n_users_per_node : int
        Number of user profiles per regional node.
    n_jobs_per_node : int
        Number of job profiles per regional node.
    n_pairs_per_node : int
        Number of suitability pairs per regional node.
    seed : int
        Base random seed. Each node uses seed + node_index for independence.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes = ["saudi_arabia", "united_states", "china", "europe"]
    all_users, all_jobs, all_outcomes = [], [], []

    for i, node_id in enumerate(nodes):
        node_seed = seed + i * 1000  # Independent seeds per node
        data = generate_synthetic_node_data(
            node_id=node_id,
            n_users=n_users_per_node,
            n_jobs=n_jobs_per_node,
            n_pairs=n_pairs_per_node,
            seed=node_seed,
        )

        # Save per-node files
        node_dir = output_dir / node_id
        node_dir.mkdir(exist_ok=True)
        data["users"].to_csv(node_dir / "users.csv", index=False)
        data["jobs"].to_csv(node_dir / "jobs.csv", index=False)
        data["outcomes"].to_csv(node_dir / "outcomes.csv", index=False)

        all_users.append(data["users"])
        all_jobs.append(data["jobs"])
        all_outcomes.append(data["outcomes"])

    # Save combined files
    pd.concat(all_users).to_csv(output_dir / "all_users.csv", index=False)
    pd.concat(all_jobs).to_csv(output_dir / "all_jobs.csv", index=False)
    pd.concat(all_outcomes).to_csv(output_dir / "all_outcomes.csv", index=False)

    logger.info(
        "Synthetic dataset saved",
        output_dir=str(output_dir),
        total_users=n_users_per_node * len(nodes),
        total_jobs=n_jobs_per_node * len(nodes),
        total_pairs=n_pairs_per_node * len(nodes),
    )
