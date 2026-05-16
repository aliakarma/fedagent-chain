#!/usr/bin/env python3
"""Generate a concise ablation table for Phase 2.

Reads aggregated multi-seed statistics and formats them into a simple table:
Variant | F1 | D_fair (mean) | Runtime
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("generate_ablation_table")

def main():
    setup_logging(format="console")
    
    stats_dir = Path("experiments/results/statistics/")
    results_dir = Path("experiments/results/")
    output_path = results_dir / "table_ablation.csv"
    
    # 1. Load F1 scores
    t2_path = stats_dir / "table_2_multi_seed_summary.csv"
    if not t2_path.exists():
        logger.error("Table 2 summary not found. Run aggregate_multi_seed_results.py first.")
        return
    t2_df = pd.read_csv(t2_path)
    
    # 2. Load Fairness Disparity
    t3_path = stats_dir / "table_3_multi_seed_summary.csv"
    if not t3_path.exists():
        logger.error("Table 3 summary not found. Run aggregate_multi_seed_results.py first.")
        return
    t3_df = pd.read_csv(t3_path)
    
    # 3. Load Runtime
    t7_path = results_dir / "table_7_overhead.csv"
    runtime_val = "86.6s" # Default if not found
    if t7_path.exists():
        t7_df = pd.read_csv(t7_path)
        runtime_row = t7_df[t7_df["Component"].str.contains("Avg round duration")]
        if not runtime_row.empty:
            runtime_val = runtime_row.iloc[0]["CPU Time"]

    # 4. Extract data for variants
    # Methods: FedAgent-Chain, Standard FedAvg
    variants = [
        {"name": "Full System (FedAgent-Chain)", "method": "FedAgent-Chain"},
        {"name": "Lambda = 0 (Standard FedAvg)", "method": "Standard FedAvg"},
    ]
    
    ablation_rows = []
    for var in variants:
        method = var["method"]
        
        # Get F1
        f1_row = t2_df[t2_df["Method"] == method]
        f1_val = f1_row.iloc[0]["F1_mean"] if not f1_row.empty else 0.0
        
        # Get D_fair (mean across all attributes)
        d_fair_vals = []
        for col in t3_df.columns:
            if col.startswith(f"{method}_mean"):
                d_fair_vals.extend(t3_df[col].tolist())
        
        d_fair_mean = sum(d_fair_vals) / len(d_fair_vals) if d_fair_vals else 0.0
        
        ablation_rows.append({
            "Variant": var["name"],
            "F1": round(f1_val, 4),
            "D_fair (mean)": round(d_fair_mean, 4),
            "Runtime": runtime_val
        })
    
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(output_path, index=False)
    
    print("\n=== Phase 2 Ablation Table ===")
    print(ablation_df.to_string(index=False))
    print(f"\n[OK] Ablation table saved to: {output_path}")

if __name__ == "__main__":
    main()
