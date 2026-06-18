#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("generate_accessibility_table")

# (indicator, baseline platform %, FedAgent-Chain %) — paper Table:
# accessibility_inclusion. Baseline = conventional centralized job platform.
ACCESSIBILITY_INDICATORS = [
    ("Accessible job recommendation coverage", 61.4, 84.7),
    ("Accommodation-aware recommendations",    48.9, 86.2),
    ("Multilingual support availability",      55.3, 88.5),
    ("Human-reviewed high-risk decisions",     22.1, 91.3),
    ("User-facing explanation availability",   39.8, 87.6),
    ("Training pathway personalization",       52.7, 82.4),
]


def generate_accessibility_table(results_dir: Path) -> pd.DataFrame:
    rows = [
        {
            "Indicator": name,
            "Baseline platform (%)": baseline,
            "FedAgent-Chain (%)": fac,
        }
        for name, baseline, fac in ACCESSIBILITY_INDICATORS
    ]
    df = pd.DataFrame(rows)
    out = results_dir / "table_accessibility_inclusion.csv"
    df.to_csv(out, index=False)
    logger.info("Accessibility & inclusion table saved", path=str(out))
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=str, default="experiments/results/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(format="console")
    df = generate_accessibility_table(ensure_dir(Path(args.results_dir)))
    print("\nTable — Accessibility & Inclusion Outcomes")
    print(df.to_string(index=False))
    print(
        "\nNote: architectural design properties (paper), not multi-seed "
        "empirical measurements."
    )


if __name__ == "__main__":
    main()
