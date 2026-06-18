"""Unit tests for fairness disparity computation."""

from __future__ import annotations

import numpy as np
import pytest

from src.federated.fairness import (
    compute_all_disparities,
    compute_fairness_disparity,
    compute_fairness_penalty,
    fairness_reweight_samples,
)


class TestComputeFairnessDisparity:
    """Tests for the compute_fairness_disparity() function."""

    def test_identical_groups_zero_disparity(self):
        """D_fair should be 0 when all groups have equal performance."""
        metrics = {"mobility": 0.85, "vision": 0.85, "hearing": 0.85}
        assert compute_fairness_disparity(metrics) == pytest.approx(0.0)

    def test_known_disparity_value(self):
        """D_fair = max - min = 0.91 - 0.72 = 0.19 (tolerance 1e-6)."""
        metrics = {"mobility": 0.85, "vision": 0.72, "hearing": 0.91}
        assert compute_fairness_disparity(metrics) == pytest.approx(0.19, abs=1e-6)

    def test_single_group_zero_disparity(self):
        """D_fair with a single group should be 0 (no comparison possible)."""
        metrics = {"mobility": 0.83}
        assert compute_fairness_disparity(metrics) == pytest.approx(0.0)

    def test_two_groups_max_disparity(self):
        """Two groups with scores 0 and 1 should give D_fair = 1.0."""
        metrics = {"group_a": 0.0, "group_b": 1.0}
        assert compute_fairness_disparity(metrics) == pytest.approx(1.0)

    def test_empty_groups_raises_error(self):
        """Empty group_metrics should raise ValueError."""
        with pytest.raises(ValueError):
            compute_fairness_disparity({})

    def test_disparity_is_non_negative(self):
        """D_fair must always be ≥ 0."""
        import random

        for _ in range(20):
            n = random.randint(2, 8)
            metrics = {f"g{i}": random.random() for i in range(n)}
            assert compute_fairness_disparity(metrics) >= 0.0

    def test_disparity_at_most_one(self):
        """D_fair should be ≤ 1.0 when metrics are in [0, 1]."""
        metrics = {f"g{i}": np.random.uniform(0, 1) for i in range(5)}
        assert compute_fairness_disparity(metrics) <= 1.0 + 1e-9


class TestComputeFairnessPenalty:
    """Tests for the compute_fairness_penalty() function."""

    def test_penalty_is_lambda_times_disparity(self):
        """Penalty = λ × D_fair."""
        metrics = {"a": 0.8, "b": 0.6}
        lambda_f = 0.1
        expected = lambda_f * 0.2
        assert compute_fairness_penalty(metrics, lambda_f) == pytest.approx(expected)

    def test_zero_lambda_gives_zero_penalty(self):
        """λ=0 should always give Ω_fair = 0."""
        metrics = {"a": 0.9, "b": 0.5}
        assert compute_fairness_penalty(metrics, lambda_fairness=0.0) == pytest.approx(0.0)


class TestComputeAllDisparities:
    """Tests for multi-attribute disparity computation."""

    def test_returns_disparity_per_attribute(self):
        """Should return one D_fair per attribute."""
        per_group = {
            "disability": {"mobility": 0.82, "vision": 0.74},
            "language": {"en": 0.85, "ar": 0.77},
        }
        result = compute_all_disparities(per_group)
        assert set(result.keys()) == {"disability", "language"}
        assert result["disability"] == pytest.approx(0.82 - 0.74, abs=1e-6)
        assert result["language"] == pytest.approx(0.85 - 0.77, abs=1e-6)


class TestFairnessReweightSamples:
    """Tests for the fairness_reweight_samples() function."""

    def test_uniform_distribution_reweight(self):
        """Uniform target should upweight underrepresented groups."""
        groups = np.array(["a"] * 80 + ["b"] * 20)
        weights = fairness_reweight_samples(groups)
        assert weights.shape == (100,)
        # Group 'b' should have higher weights (underrepresented)
        assert weights[80] > weights[0]

    def test_weights_sum_to_n_samples(self):
        """Normalized weights should sum to n_samples."""
        groups = np.array(["a"] * 60 + ["b"] * 40)
        weights = fairness_reweight_samples(groups)
        assert abs(weights.sum() - len(groups)) < 1e-4

    def test_balanced_groups_equal_weights(self):
        """Perfectly balanced groups should receive equal weights."""
        groups = np.array(["a"] * 50 + ["b"] * 50)
        weights = fairness_reweight_samples(groups)
        np.testing.assert_array_almost_equal(weights[:50], weights[50:], decimal=4)
