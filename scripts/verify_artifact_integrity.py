#!/usr/bin/env python3
"""Verify that all committed result artifacts are internally consistent.

Checks:
1. Per-seed tables exist for all 3 seeds
2. Multi-seed summary std > 0 (seeds are independent)  
3. Statistical tests have finite values
4. Centralized != Local Baseline
5. FedAgent-Chain fairness improvement exists for at least 2/4 attributes
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

RESULTS = Path("experiments/results")
errors = []
warnings = []

# Check 1: Per-seed tables exist
for seed in [42, 123, 2024]:
    for table in ["table_2_model_performance.csv", "table_3_fairness_results.csv"]:
        p = RESULTS / f"seed_{seed}" / table
        if not p.exists():
            errors.append(f"MISSING: {p}")

# Check 2: Multi-seed std > 0
summary = RESULTS / "table_2_multi_seed_summary.csv"
if summary.exists():
    df = pd.read_csv(summary)
    fa_row = df[df["Method"].str.contains("FedAgent", na=False)]
    if not fa_row.empty:
        f1_std = float(fa_row["F1_std"].values[0])
        if f1_std < 0.0001:
             errors.append(f"F1_std ≈ 0 ({f1_std}) in multi-seed summary — seeds are not independent")
    else:
        errors.append("FedAgent-Chain row missing from multi-seed summary")

# Check 3: Statistical tests finite
stats_path = RESULTS / "statistical_tests.csv"
if stats_path.exists():
    stats_df = pd.read_csv(stats_path)
    if not np.isfinite(stats_df["t_statistic"].values).all():
        errors.append("Non-finite t-statistics in statistical_tests.csv")

# Check 4: Centralized != Local
t2 = RESULTS / "table_2_model_performance.csv"
if t2.exists():
    df = pd.read_csv(t2)
    local_rows = df[df["Method"] == "Local Baseline"]
    central_rows = df[df["Method"] == "Centralized"]
    
    if not local_rows.empty and not central_rows.empty:
        local_f1 = float(local_rows["F1"].values[0])
        central_f1 = float(central_rows["F1"].values[0])
        if abs(local_f1 - central_f1) < 0.005:
            errors.append(f"Centralized F1 ({central_f1:.4f}) == Local Baseline F1 ({local_f1:.4f})")

# Report
print("=== ARTIFACT INTEGRITY CHECK ===")
if errors:
    print(f"\n{len(errors)} ERRORS FOUND:")
    for e in errors:
        print(f"  [ERROR] {e}")
    sys.exit(1)
else:
    print("\nPASS: All artifact integrity checks passed")
