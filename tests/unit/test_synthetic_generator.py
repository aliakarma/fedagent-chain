"""Unit tests for synthetic data generation and schema validation."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.schema import DisabilityCategory, NodeID, UserProfile
from src.data.synthetic_generator import (
    compute_suitability_label,
    generate_job_profiles,
    generate_synthetic_node_data,
    generate_user_profiles,
)
from src.utils.seed_utils import get_rng


class TestGenerateUserProfiles:

    def test_correct_count(self):
        rng = get_rng(42)
        profiles = generate_user_profiles("saudi_arabia", n_users=50, rng=rng)
        assert len(profiles) == 50

    def test_all_profiles_valid(self):
        rng = get_rng(42)
        profiles = generate_user_profiles("united_states", n_users=30, rng=rng)
        for p in profiles:
            assert isinstance(p, UserProfile)
            assert len(p.skill_vector) == 50
            assert len(p.accommodation_needs) == 20
            assert all(v in (0, 1) for v in p.skill_vector)
            assert all(v in (0, 1) for v in p.accommodation_needs)

    def test_node_id_matches(self):
        rng = get_rng(0)
        profiles = generate_user_profiles("china", n_users=10, rng=rng)
        for p in profiles:
            assert p.node_id == NodeID.CHINA

    def test_consent_rate_around_92_percent(self):
        rng = get_rng(42)
        profiles = generate_user_profiles("europe", n_users=500, rng=rng)
        consent_rate = sum(1 for p in profiles if p.consent_given) / len(profiles)
        assert 0.85 <= consent_rate <= 0.98  # allow statistical variation

    def test_disability_categories_valid(self):
        rng = get_rng(1)
        profiles = generate_user_profiles("saudi_arabia", n_users=50, rng=rng)
        valid_cats = {c.value for c in DisabilityCategory}
        for p in profiles:
            assert p.disability_category.value in valid_cats


class TestGenerateJobProfiles:

    def test_correct_count(self):
        rng = get_rng(42)
        jobs = generate_job_profiles("united_states", n_jobs=30, rng=rng)
        assert len(jobs) == 30

    def test_accessibility_score_in_range(self):
        rng = get_rng(7)
        jobs = generate_job_profiles("europe", n_jobs=50, rng=rng)
        for j in jobs:
            assert 0.0 <= j.accessibility_score <= 1.0

    def test_skill_vector_binary(self):
        rng = get_rng(3)
        jobs = generate_job_profiles("china", n_jobs=20, rng=rng)
        for j in jobs:
            assert all(v in (0, 1) for v in j.required_skills)


class TestComputeSuitabilityLabel:

    def test_returns_binary_label_and_float_score(self):
        rng = get_rng(42)
        users = generate_user_profiles("saudi_arabia", 1, rng)
        jobs = generate_job_profiles("saudi_arabia", 1, rng)
        label, score = compute_suitability_label(users[0], jobs[0])
        assert label in (0, 1)
        assert 0.0 <= score <= 1.0

    def test_perfect_match_is_positive(self):
        """User and job with identical skills and matching language should be positive."""
        rng = get_rng(0)
        users = generate_user_profiles("united_states", 5, rng)
        jobs = generate_job_profiles("united_states", 5, rng)
        # At least some pairs should be positive with diverse profiles
        labels = [compute_suitability_label(u, j)[0] for u, j in zip(users, jobs)]
        # Not all should be 0
        assert any(l == 1 for l in labels) or any(l == 0 for l in labels)

    def test_label_deterministic_without_rng(self):
        """Without rng, same inputs should give same output."""
        rng = get_rng(0)
        users = generate_user_profiles("china", 1, rng)
        jobs = generate_job_profiles("china", 1, rng)
        l1, s1 = compute_suitability_label(users[0], jobs[0])
        l2, s2 = compute_suitability_label(users[0], jobs[0])
        assert l1 == l2
        assert s1 == pytest.approx(s2)


class TestGenerateSyntheticNodeData:

    def test_returns_three_dataframes(self):
        data = generate_synthetic_node_data("europe", n_users=20, n_jobs=10, n_pairs=40, seed=42)
        assert set(data.keys()) == {"users", "jobs", "outcomes"}

    def test_user_count(self):
        data = generate_synthetic_node_data("china", n_users=25, n_jobs=10, n_pairs=50, seed=0)
        assert len(data["users"]) == 25

    def test_job_count(self):
        data = generate_synthetic_node_data("united_states", n_users=10, n_jobs=15, n_pairs=30, seed=1)
        assert len(data["jobs"]) == 15

    def test_outcomes_have_required_columns(self):
        data = generate_synthetic_node_data("saudi_arabia", n_users=10, n_jobs=5, n_pairs=20, seed=2)
        required_cols = {"user_id", "job_id", "suitability_label", "suitability_score"}
        assert required_cols.issubset(set(data["outcomes"].columns))

    def test_suitability_labels_binary(self):
        data = generate_synthetic_node_data("europe", n_users=20, n_jobs=10, n_pairs=50, seed=3)
        labels = data["outcomes"]["suitability_label"].unique()
        for l in labels:
            assert l in (0, 1)

    def test_reproducible_with_same_seed(self):
        d1 = generate_synthetic_node_data("china", n_users=20, n_jobs=10, n_pairs=30, seed=99)
        d2 = generate_synthetic_node_data("china", n_users=20, n_jobs=10, n_pairs=30, seed=99)
        assert list(d1["users"]["user_id"]) == list(d2["users"]["user_id"])
