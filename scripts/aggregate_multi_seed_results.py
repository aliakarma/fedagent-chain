#!/usr/bin/env python3
"""Aggregate results across multiple seeds and compute statistical tests.

Reads table_2_model_performance.csv produced per-seed, aggregates mean/std,
and runs paired t-tests between FedAgent-Chain and each baseline.

Usage:
    python scripts/aggregate_multi_seed_results.py \
        --seeds 42 123 2024 \
        --results-dir experiments/results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t as t_dist

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("aggregate_multi_seed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds",       nargs="+", type=int, default=[42, 123, 2024])
    p.add_argument("--results-dir", type=str,  default="experiments/results/")
    return p.parse_args()


def run_statistical_tests(
    method_a_scores: list[float],
    method_b_scores: list[float],
    method_a_name: str,
    method_b_name: str,
) -> dict | None:
    """Run paired t-test and compute Cohen's d between two methods.
    
    Returns None (not a result row) if the test cannot be validly performed.
    This prevents numerically degenerate results (t=-inf, p=0) from being
    written to statistical_tests.csv.
    """
    a = np.array(method_a_scores)
    b = np.array(method_b_scores)

    # Guard 1: Minimum sample size
    if len(a) < 2:
        logger.warning(
            "Insufficient samples for statistical test",
            method_a=method_a_name,
            method_b=method_b_name,
            n_samples=len(a),
        )
        return None
    
    # Guard 2: Degenerate variance (all values identical)
    diff = a - b
    diff_std = float(np.std(diff, ddof=1))
    if diff_std < 1e-8:
        logger.warning(
            "Statistical test degenerate: std(diff) ~ 0. "
            "This indicates all seeds produced identical results, "
            "which suggests the per-seed results are copies of a single run "
            "rather than genuinely independent experiments. "
            "Test results WILL NOT be written.",
            method_a=method_a_name,
            method_b=method_b_name,
            diff_std=diff_std,
            a_values=list(a),
            b_values=list(b),
        )
        return None
    
    # Guard 3: Values must be in valid range [0, 1] for F1
    for arr, name in [(a, method_a_name), (b, method_b_name)]:
        if np.any(arr < 0) or np.any(arr > 1):
            logger.error(
                "F1 values out of [0,1] range — data integrity issue",
                method=name,
                values=list(arr),
            )
            return None
    
    t_stat, p_value = stats.ttest_rel(a, b)
    
    # Guard 4: Check for infinite or NaN results
    if not np.isfinite(t_stat) or not np.isfinite(p_value):
        logger.error(
            "Non-finite t-statistic or p-value despite std > 0 — numerical issue",
            t_stat=float(t_stat),
            p_value=float(p_value),
            diff_std=diff_std,
        )
        return None
    
    cohens_d = float(np.mean(diff)) / (diff_std + 1e-12)

    return {
        "comparison": f"{method_a_name} vs {method_b_name}",
        "mean_diff": round(float(np.mean(diff)), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "cohens_d": round(float(cohens_d), 4),
        "significant": bool(p_value < 0.05),
        "n_seeds": len(a),
        "warning": "With n=3 seeds, statistical power is limited. p < 0.05 requires very large effect sizes." if len(a) < 5 else None,
    }


def main() -> None:
    args = parse_args()
    setup_logging(format="console")
    results_dir = ensure_dir(Path(args.results_dir))
    seeds = args.seeds

    # ── Load Table 2 per seed ────────────────────────────────────────────────
    per_seed: dict[int, pd.DataFrame] = {}
    for seed in seeds:
        seed_dir = results_dir / f"seed_{seed}"
        t2_path  = seed_dir / "table_2_model_performance.csv"
        if not t2_path.exists():
            # Fallback for missing seed-subdir (checking if it's in the main dir)
            t2_path = results_dir / "table_2_model_performance.csv"
        if not t2_path.exists():
            logger.warning("Table 2 not found for seed", seed=seed, path=str(t2_path))
            continue
        per_seed[seed] = pd.read_csv(t2_path)

    if len(per_seed) < 2:
        logger.error(
            "Need results for at least 2 seeds. "
            "Run run_evaluation.py for each seed first, "
            "saving to experiments/results/seed_{seed}/ subdirectories."
        )
        return

    # ── Collect F1 scores per method across seeds ────────────────────────────
    # Use methods from the first available seed
    first_seed = next(iter(per_seed))
    methods = per_seed[first_seed]["Method"].tolist()
    f1_by_method: dict[str, list[float]] = {m: [] for m in methods}

    for seed, df in per_seed.items():
        for method in methods:
            row = df[df["Method"] == method]
            if not row.empty:
                f1_by_method[method].append(float(row["F1"].iloc[0]))

    # ── Summary table with CIs ───────────────────────────────────────────────
    summary_rows = []
    for method, f1_list in f1_by_method.items():
        if not f1_list:
            continue
        arr = np.array(f1_list)
        n   = len(arr)
        mean = np.mean(arr)
        std  = np.std(arr, ddof=1) if n > 1 else 0.0
        
        row = {
            "Method":  method,
            "F1_mean": round(float(mean), 4),
            "F1_std":  round(float(std), 4),
            "F1_min":  round(float(np.min(arr)), 4),
            "F1_max":  round(float(np.max(arr)), 4),
            "n_seeds": n,
        }
        
        # 95% Confidence Intervals
        if n > 1:
            se   = std / np.sqrt(n)
            df_t = n - 1
            t_cv = t_dist.ppf(0.975, df=df_t)
            row["CI_95_lower"] = round(float(mean - t_cv * se), 4)
            row["CI_95_upper"] = round(float(mean + t_cv * se), 4)
        else:
            row["CI_95_lower"] = round(float(mean), 4)
            row["CI_95_upper"] = round(float(mean), 4)
            
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "table_2_multi_seed_summary.csv", index=False)

    print("\n=== Multi-Seed F1 Summary (with 95% CI) ===")
    print(summary_df.to_string(index=False))

    # ── Statistical tests: FedAgent-Chain vs each baseline ───────────────────
    fedagent_key = next((k for k in f1_by_method if "FedAgent" in k), None)
    if fedagent_key is None:
        logger.warning("FedAgent-Chain results not found in summary table")
        return

    test_rows = []
    for method in methods:
        if method == fedagent_key:
            continue
        if len(f1_by_method[method]) < 2:
            continue
        # Ensure we have the same seeds for both methods
        if len(f1_by_method[fedagent_key]) != len(f1_by_method[method]):
            logger.warning(
                "Skipping comparison due to seed count mismatch",
                method=method,
                fedagent_n=len(f1_by_method[fedagent_key]),
                method_n=len(f1_by_method[method])
            )
            continue
            
        result = run_statistical_tests(
            f1_by_method[fedagent_key],
            f1_by_method[method],
            fedagent_key, method,
        )
        if result is not None:
            test_rows.append(result)

    if test_rows:
        tests_df = pd.DataFrame(test_rows)
        tests_df.to_csv(results_dir / "statistical_tests.csv", index=False)

        print("\n=== Statistical Tests (paired t-test) ===")
        print(tests_df.to_string(index=False))
        print(
            "\n  Significance threshold: p < 0.05  |  "
            "Strong effect: |d| > 0.80  |  "
            "Medium: |d| > 0.50"
        )
    else:
        print("\n[!] No statistical tests performed (insufficient multi-seed data)")

    print(f"\n[OK] Statistical analysis saved to: {results_dir}")


if __name__ == "__main__":
    main()
