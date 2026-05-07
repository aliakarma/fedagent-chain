"""Unit tests for federated averaging aggregators."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.federated.aggregator import FairnessAwareFedAvgAggregator, FedAvgAggregator


def make_updates(n_nodes: int = 3, param_size: int = 10, seed: int = 42):
    """Helper: create fake (node_id, state_dict, n_samples) tuples."""
    rng = np.random.default_rng(seed)
    updates = []
    for i in range(n_nodes):
        state = {
            "layer.weight": rng.standard_normal((param_size,)).astype(np.float32),
            "layer.bias": rng.standard_normal((4,)).astype(np.float32),
        }
        updates.append((f"node_{i}", state, (i + 1) * 100))
    return updates


class TestFedAvgAggregator:

    def test_aggregate_returns_correct_keys(self):
        agg = FedAvgAggregator()
        updates = make_updates(3)
        result = agg.aggregate(updates)
        assert set(result.keys()) == {"layer.weight", "layer.bias"}

    def test_weights_sum_to_one(self):
        agg = FedAvgAggregator()
        updates = make_updates(4)
        weights = agg.compute_weights(updates)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weights_proportional_to_n_samples(self):
        agg = FedAvgAggregator()
        # node_0 has 100, node_1 has 900 → weights 0.1 and 0.9
        updates = [
            ("node_0", {"w": np.ones(5, dtype=np.float32)}, 100),
            ("node_1", {"w": np.ones(5, dtype=np.float32)}, 900),
        ]
        weights = agg.compute_weights(updates)
        assert weights["node_0"] == pytest.approx(0.1)
        assert weights["node_1"] == pytest.approx(0.9)

    def test_aggregate_equal_weights_gives_mean(self):
        agg = FedAvgAggregator()
        updates = [
            ("node_0", {"w": np.array([2.0, 4.0], dtype=np.float32)}, 100),
            ("node_1", {"w": np.array([4.0, 8.0], dtype=np.float32)}, 100),
        ]
        result = agg.aggregate(updates)
        np.testing.assert_array_almost_equal(result["w"], [3.0, 6.0])

    def test_aggregate_empty_raises(self):
        agg = FedAvgAggregator()
        with pytest.raises(ValueError, match="empty"):
            agg.aggregate([])

    def test_aggregate_preserves_dtype(self):
        agg = FedAvgAggregator()
        updates = make_updates(2)
        result = agg.aggregate(updates)
        for name, arr in result.items():
            assert arr.dtype == np.float32

    def test_aggregate_single_node(self):
        """Single-node aggregation should return the same parameters."""
        agg = FedAvgAggregator()
        state = {"w": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
        result = agg.aggregate([("only_node", state, 500)])
        np.testing.assert_array_almost_equal(result["w"], state["w"])


class TestFairnessAwareFedAvgAggregator:

    def setup_method(self):
        self.cfg = OmegaConf.create({
            "lambda_fairness": 0.1,
            "protected_groups": ["disability_category"],
        })
        self.agg = FairnessAwareFedAvgAggregator(self.cfg)

    def test_fairness_weights_sum_to_one(self):
        updates = make_updates(4)
        fairness_scores = {f"node_{i}": 0.5 + i * 0.1 for i in range(4)}
        weights = self.agg.compute_weights(updates, fairness_scores)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_low_fairness_node_gets_higher_weight(self):
        """Node with lower min-group score (worse fairness) should get upweighted."""
        updates = [
            ("node_fair", {"w": np.ones(5, dtype=np.float32)}, 500),
            ("node_unfair", {"w": np.ones(5, dtype=np.float32)}, 500),
        ]
        fairness_scores = {"node_fair": 0.9, "node_unfair": 0.3}
        weights = self.agg.compute_weights(updates, fairness_scores)
        assert weights["node_unfair"] > weights["node_fair"]

    def test_no_fairness_scores_falls_back_to_fedavg(self):
        """When fairness_scores is None, should use standard FedAvg weights."""
        updates = make_updates(3)
        fa_weights = self.agg.compute_weights(updates, fairness_scores=None)
        base_weights = FedAvgAggregator().compute_weights(updates)
        for node_id in base_weights:
            assert fa_weights[node_id] == pytest.approx(base_weights[node_id], abs=1e-6)

    def test_fairness_adjustment_factor_increases_for_low_score(self):
        """ρ_k should be > 1 for any node with min_group_score < 1."""
        rho = self.agg.compute_fairness_adjustment("node_x", {"node_x": 0.5})
        assert rho > 1.0

    def test_fairness_adjustment_factor_for_perfect_fairness(self):
        """ρ_k = 1.0 when node has perfect fairness (score=1.0)."""
        rho = self.agg.compute_fairness_adjustment("node_x", {"node_x": 1.0})
        assert rho == pytest.approx(1.0)
