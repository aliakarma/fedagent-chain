from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

RESULTS_DIR = Path("experiments/results")
STATS_DIR = RESULTS_DIR / "statistics"
TOL = 0.010   # tolerance against paper-reported values


# ── Paper-reported reference values (disable.tex) ─────────────────────────────
REFERENCE = {
    "fedagent_chain_f1_mean":   0.7207,
    "standard_fedavg_f1_mean":  0.7116,
    "local_baseline_f1_mean":   0.5380,
    "centralized_lr_f1_mean":   0.7115,
    "centralized_nn_f1_mean":   0.7383,
    "disability_disparity_fedagent":    0.0517,
    "disability_disparity_standard_fl": 0.0428,
}


def load_table(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _method_row(df: pd.DataFrame, needle: str) -> pd.DataFrame:
    return df[df["Method"].str.contains(needle, case=False, na=False, regex=False)]


@pytest.mark.regression
class TestTable2ModelPerformance:

    def test_summary_has_five_methods(self):
        df = load_table(STATS_DIR / "table_2_multi_seed_summary.csv")
        if df is None:
            pytest.skip("Run aggregate_multi_seed_results.py first.")
        for needle in ["FedAgent", "FedAvg", "Local", "Centralized (LR)", "Centralized (NN)"]:
            assert not _method_row(df, needle).empty, f"missing method: {needle}"

    @pytest.mark.parametrize("needle,key", [
        ("FedAgent",        "fedagent_chain_f1_mean"),
        ("FedAvg",          "standard_fedavg_f1_mean"),
        ("Local",           "local_baseline_f1_mean"),
        ("Centralized (LR)", "centralized_lr_f1_mean"),
        ("Centralized (NN)", "centralized_nn_f1_mean"),
    ])
    def test_f1_matches_paper(self, needle, key):
        df = load_table(STATS_DIR / "table_2_multi_seed_summary.csv")
        if df is None:
            pytest.skip("Run aggregate_multi_seed_results.py first.")
        row = _method_row(df, needle)
        if row.empty:
            pytest.skip(f"{needle} row not present")
        actual = float(row["F1_mean"].iloc[0])
        assert abs(actual - REFERENCE[key]) <= TOL, (
            f"{needle} F1_mean={actual} deviates from paper {REFERENCE[key]}"
        )


@pytest.mark.regression
class TestStatisticalTests:

    def test_includes_centralized_nn_comparison(self):
        df = load_table(STATS_DIR / "statistical_tests.csv")
        if df is None:
            pytest.skip("Run aggregate_multi_seed_results.py first.")
        nn = df[df["comparison"].str.contains("Centralized (NN)", case=False, na=False, regex=False)]
        assert not nn.empty, "missing FedAgent-Chain vs Centralized (NN) comparison"
        assert abs(float(nn["cohens_d"].iloc[0]) - (-0.31)) <= 0.05

    def test_no_comparison_significant_at_n5(self):
        df = load_table(STATS_DIR / "statistical_tests.csv")
        if df is None:
            pytest.skip()
        # With n=5 seeds the paper reports no significant comparison.
        assert (df["p_value"] >= 0).all() and (df["p_value"] <= 1).all()
        assert not df["significant"].any()


@pytest.mark.regression
class TestTable3FairnessDisparity:

    def test_disability_disparity_matches_paper(self):
        df = load_table(STATS_DIR / "table_3_multi_seed_summary.csv")
        if df is None:
            pytest.skip()
        row = df[df["Attribute"].str.contains("Disab", case=False, na=False)]
        if row.empty:
            pytest.skip()
        fac = float(row["FedAgent-Chain_mean"].iloc[0])
        std = float(row["Standard FedAvg_mean"].iloc[0])
        assert abs(fac - REFERENCE["disability_disparity_fedagent"]) <= TOL
        assert abs(std - REFERENCE["disability_disparity_standard_fl"]) <= TOL
        # Paper's honest finding: penalty does NOT beat FedAvg on disability disparity.
        assert fac > std


@pytest.mark.regression
class TestTable4BlockchainAuditability:

    def test_six_indicators_present(self):
        df = load_table(RESULTS_DIR / "table_4_blockchain_results.csv")
        if df is None:
            pytest.skip("Run run_blockchain_audit.py first.")
        col = df.columns[0]
        text = " ".join(df[col].astype(str)).lower()
        for needle in ["hash logging", "consent validation", "aggregation event",
                       "unauthorized update", "hash computation", "raw disability"]:
            assert needle in text, f"missing blockchain indicator: {needle}"

    def test_unauthorized_rejection_rate(self):
        df = load_table(RESULTS_DIR / "table_4_blockchain_results.csv")
        if df is None:
            pytest.skip()
        row = df[df.iloc[:, 0].str.contains("Unauthorized", case=False, na=False)]
        assert not row.empty
        assert "96.7" in str(row.iloc[0, 1]) and "29/30" in str(row.iloc[0, 1])


@pytest.mark.regression
class TestComponentAblation:

    def test_six_configurations(self):
        df = load_table(RESULTS_DIR / "table_ablation.csv")
        if df is None:
            pytest.skip("Run generate_ablation_table.py first.")
        assert len(df) == 6
        rows = {r["Configuration"]: r for _, r in df.iterrows()}
        # Removing blockchain zeroes audit completeness; removing governance
        # zeroes detection; both leave F1 unchanged.
        assert float(rows["w/o blockchain layer"]["Audit completeness"]) == 0.0
        assert float(rows["w/o governance agent"]["Gov detection"]) == 0.0
        assert abs(float(rows["Full FedAgent-Chain"]["F1 Mean"]) - 0.7207) <= TOL


@pytest.mark.regression
class TestAccessibilityInclusion:

    def test_six_indicators_fac_beats_baseline(self):
        df = load_table(RESULTS_DIR / "table_accessibility_inclusion.csv")
        if df is None:
            pytest.skip("Run generate_accessibility_table.py first.")
        assert len(df) == 6
        assert (df["FedAgent-Chain (%)"] > df["Baseline platform (%)"]).all()


@pytest.mark.regression
class TestEducationDemos:

    def test_four_pathway_demos_exist(self):
        demos = list((RESULTS_DIR / "demos").glob("education_pathway_*.md"))
        if not demos:
            pytest.skip("Run generate_education_demonstrations.py first.")
        assert len(demos) == 4
