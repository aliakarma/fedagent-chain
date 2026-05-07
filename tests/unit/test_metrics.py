"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_full_metrics,
    precision_at_k,
    recall_at_k,
)


class TestClassificationMetrics:

    def test_perfect_predictions(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])
        m = compute_classification_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 1, 1])
        m = compute_classification_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_metric_range(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        y_pred = rng.integers(0, 2, size=100)
        m = compute_classification_metrics(y_true, y_pred)
        for v in m.values():
            assert 0.0 <= v <= 1.0

    def test_returns_all_required_keys(self):
        y_true = np.ones(10)
        y_pred = np.ones(10)
        m = compute_classification_metrics(y_true, y_pred)
        assert set(m.keys()) == {"accuracy", "precision", "recall", "f1"}


class TestPrecisionAtK:

    def test_perfect_ranking(self):
        y_true = np.array([1, 1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        assert precision_at_k(y_true, y_scores, k=3) == pytest.approx(1.0)

    def test_worst_ranking(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        assert precision_at_k(y_true, y_scores, k=3) == pytest.approx(0.0)

    def test_k_larger_than_n_capped(self):
        y_true = np.array([1, 0, 1])
        y_scores = np.array([0.9, 0.5, 0.8])
        # k=10 but only 3 items; should not raise
        p = precision_at_k(y_true, y_scores, k=10)
        assert 0.0 <= p <= 1.0

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            precision_at_k(np.ones(5), np.ones(5), k=0)


class TestRecallAtK:

    def test_all_relevant_found_in_top_k(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.4, 0.3, 0.2])
        assert recall_at_k(y_true, y_scores, k=2) == pytest.approx(1.0)

    def test_no_relevant_found(self):
        y_true = np.array([0, 0, 1, 1, 0])
        y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.7])
        assert recall_at_k(y_true, y_scores, k=2) == pytest.approx(0.0)

    def test_no_relevant_items_returns_zero(self):
        y_true = np.array([0, 0, 0])
        y_scores = np.array([0.9, 0.5, 0.1])
        assert recall_at_k(y_true, y_scores, k=2) == pytest.approx(0.0)


class TestComputeFullMetrics:

    def test_includes_classification_and_ranking(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=50)
        y_pred = rng.integers(0, 2, size=50)
        y_scores = rng.random(50)
        m = compute_full_metrics(y_true, y_pred, y_scores, k_values=[5, 10])
        assert "f1" in m
        assert "precision_at_5" in m
        assert "recall_at_10" in m

    def test_without_scores_no_ranking_metrics(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        m = compute_full_metrics(y_true, y_pred, y_scores=None)
        assert "precision_at_5" not in m
        assert "f1" in m
