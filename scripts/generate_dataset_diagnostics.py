#!/usr/bin/env python3
"""Generate dataset diagnostics for Phase 4.

Analyzes the suitability label balance (0 vs 1) across all regional nodes
and saves the results to experiments/results/class_distribution.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main():
    data_dir = Path("data/synthetic/")
    results_dir = Path("experiments/results/")
    results_dir.mkdir(parents=True, exist_ok=True)

    nodes = ["saudi_arabia", "united_states", "china", "europe"]
    rows = []

    for node in nodes:
        outcomes_path = data_dir / node / "outcomes.csv"
        if not outcomes_path.exists():
            continue

        df = pd.read_csv(outcomes_path)
        total = len(df)
        pos = int(df["suitability_label"].sum())
        neg = total - pos
        pos_pct = (pos / total) * 100 if total > 0 else 0

        rows.append(
            {
                "Node": node,
                "Total Samples": total,
                "Suitable (1)": pos,
                "Unsuitable (0)": neg,
                "Balance (%)": f"{pos_pct:.1f}%",
            }
        )

    dist_df = pd.DataFrame(rows)
    output_path = results_dir / "class_distribution.csv"
    dist_df.to_csv(output_path, index=False)

    print("\n=== Phase 4: Class Distribution Statistics ===")
    print(dist_df.to_string(index=False))
    print(f"\n[OK] Statistics saved to: {output_path}")


if __name__ == "__main__":
    main()
