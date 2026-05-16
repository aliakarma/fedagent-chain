#!/usr/bin/env python3
"""Regenerate all paper tables from simulation result CSVs.

Usage:
    python scripts/generate_tables.py --results-dir experiments/results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("generate_tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print paper tables from result CSVs.")
    parser.add_argument("--results-dir", type=str, default="experiments/results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(format="console")

    results_dir = Path(args.results_dir)
    # Search in results_dir, statistics/, and aggregated/
    patterns = ["table_*.csv", "statistics/table_*.csv", "aggregated/table_*.csv"]
    table_files = []
    for p in patterns:
        table_files.extend(sorted(results_dir.glob(p)))

    if not table_files:
        logger.warning("No result CSVs found. Run run_evaluation.py and aggregation first.")
        return

    for csv_path in table_files:
        df = pd.read_csv(csv_path)
        print(f"\n{'─'*70}")
        print(f"  {csv_path.stem.replace('_', ' ').title()}")
        print(f"{'─'*70}")
        print(df.to_string(index=False))

    print(f"\n✅ All tables printed from: {results_dir}")


if __name__ == "__main__":
    main()
