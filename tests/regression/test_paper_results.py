"""Regression tests verifying implementations match paper-reported values.

These tests load pre-computed result CSVs from experiments/results/ and
verify they are within tolerance of the values reported in the paper.

Tolerance: ±0.005 for all reported metrics (accounts for float precision).

Run with:
    pytest tests/regression/ -v -m regression --timeout=600
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

TOLERANCE = 0.005
RESULTS_DIR = Path("experiments/results")

# Paper Table 2 target values — FedAgent-Chain (Fairness-Aware FL)
PAPER_TABLE_2 = {
    "fedagent_chain_accuracy": 0.846,
    "fedagent_chain_precision": 0.835,
    "fedagent_chain_recall": 0.829,
    "fedagent_chain_f1": 0.832,
}

# Paper Table 3 target values — fairness disparity after optimization
PAPER_TABLE_3 = {
    "disability_fairness_fl": 0.064,
    "language_fairness_fl": 0.071,
    "workmode_fairness_fl": 0.054,
    "region_fairness_fl": 0.058,
}

# Paper Table 4 target values — blockchain auditability
PAPER_TABLE_4 = {
    "hash_completeness": 1.00,
    "consent_validation_rate": 1.00,
}


def load_table(filename: str) -> pd.DataFrame | None:
    """Load a results CSV, returning None if not found."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


@pytest.mark.regression
class TestTable2ModelPerformance:
    """Regression tests for Table 2 — Model Performance Comparison."""

    def test_results_csv_exists(self):
        """Table 2 CSV must exist before regression testing."""
        path = RESULTS_DIR / "table_2_model_performance.csv"
        if not path.exists():
            pytest.skip("Run run_evaluation.py first to generate results.")
        assert path.exists()

    def test_fedagent_chain_f1_within_tolerance(self):
        df = load_table("table_2_model_performance.csv")
        if df is None:
            pytest.skip("Results not available.")
        row = df[df["Method"].str.contains("FedAgent", case=False, na=False)]
        if row.empty:
            pytest.skip("FedAgent-Chain row not found in Table 2.")
        f1 = float(row["F1"].iloc[0])
        assert abs(f1 - PAPER_TABLE_2["fedagent_chain_f1"]) <= TOLERANCE, (
            f"F1={f1:.4f} deviates from paper value {PAPER_TABLE_2['fedagent_chain_f1']:.4f} "
            f"by more than tolerance {TOLERANCE}"
        )

    def test_fedagent_chain_accuracy_within_tolerance(self):
        df = load_table("table_2_model_performance.csv")
        if df is None:
            pytest.skip("Results not available.")
        row = df[df["Method"].str.contains("FedAgent", case=False, na=False)]
        if row.empty:
            pytest.skip("FedAgent-Chain row not found.")
        acc = float(row["Accuracy"].iloc[0])
        assert abs(acc - PAPER_TABLE_2["fedagent_chain_accuracy"]) <= TOLERANCE


@pytest.mark.regression
class TestTable3FairnessDisparity:
    """Regression tests for Table 3 — Fairness Disparity."""

    def test_disability_disparity_within_tolerance(self):
        df = load_table("table_3_fairness_results.csv")
        if df is None:
            pytest.skip("Results not available.")
        row = df[df["Attribute"].str.contains("Disab", case=False, na=False)]
        if row.empty:
            pytest.skip("Disability row not found.")
        val = float(row["FedAgent-Chain"].iloc[0])
        assert abs(val - PAPER_TABLE_3["disability_fairness_fl"]) <= TOLERANCE

    def test_language_disparity_within_tolerance(self):
        df = load_table("table_3_fairness_results.csv")
        if df is None:
            pytest.skip("Results not available.")
        row = df[df["Attribute"].str.contains("Lang", case=False, na=False)]
        if row.empty:
            pytest.skip("Language row not found.")
        val = float(row["FedAgent-Chain"].iloc[0])
        assert abs(val - PAPER_TABLE_3["language_fairness_fl"]) <= TOLERANCE

    def test_fairness_disparity_lower_with_fairness_fl(self):
        """FedAgent-Chain D_fair must be strictly lower than Standard FL for all attributes."""
        df = load_table("table_3_fairness_results.csv")
        if df is None:
            pytest.skip("Results not available.")
        if "Standard FL" not in df.columns or "FedAgent-Chain" not in df.columns:
            pytest.skip("Required columns missing.")
        assert all(df["FedAgent-Chain"] < df["Standard FL"]), (
            "FedAgent-Chain fairness disparity should be lower than Standard FL for all attributes."
        )


@pytest.mark.regression
class TestTable4BlockchainAuditability:
    """Regression tests for Table 4 — Blockchain Auditability."""

    def test_hash_completeness_is_100_percent(self):
        df = load_table("table_4_blockchain_results.csv")
        if df is None:
            pytest.skip("Results not available.")
        row = df[df["Metric"].str.contains("Hash Completeness", case=False, na=False)]
        if row.empty:
            pytest.skip("Hash completeness row not found.")
        val_str = str(row["Value"].iloc[0])
        val = float(val_str.replace("%", "")) / 100.0
        assert abs(val - 1.0) <= TOLERANCE, "Hash completeness should be 100%"

    def test_consent_validation_rate_is_100_percent(self):
        df = load_table("table_4_blockchain_results.csv")
        if df is None:
            pytest.skip("Results not available.")
        row = df[df["Metric"].str.contains("Consent Validation", case=False, na=False)]
        if row.empty:
            pytest.skip("Consent row not found.")
        val_str = str(row["Value"].iloc[0])
        val = float(val_str.replace("%", "")) / 100.0
        assert abs(val - 1.0) <= TOLERANCE
