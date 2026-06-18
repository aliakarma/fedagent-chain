#!/usr/bin/env python3
"""Export blockchain audit logs from a simulation run.

Usage:
    python scripts/export_blockchain_audit.py \
        --run-dir experiments/runs/fedagent_chain_full_seed42_20251115 \
        --output-dir experiments/results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("export_blockchain_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export blockchain audit log.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="experiments/runs/",
        help="Run directory containing blockchain_logs/.",
    )
    parser.add_argument("--output-dir", type=str, default="experiments/results/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(format="console")

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find audit log in run directory
    audit_candidates = list(run_dir.rglob("audit_trail.json"))

    if not audit_candidates:
        logger.warning("No audit log found in run directory", run_dir=str(run_dir))
        return

    audit_path = audit_candidates[0]
    logger.info("Loading audit log", path=str(audit_path))

    with open(audit_path) as f:
        audit_data = json.load(f)

    # Verify chain integrity
    is_valid = audit_data.get("chain_integrity_valid", False)
    n_blocks = audit_data.get("chain_length", 0)
    n_records = audit_data.get("total_records", 0)

    print(f"\n{'='*60}")
    print("Blockchain Audit Log Summary")
    print(f"{'='*60}")
    print(f"  Chain length      : {n_blocks} blocks")
    print(f"  Total records     : {n_records}")
    print(f"  Chain integrity   : {'✅ VALID' if is_valid else '❌ INVALID'}")
    print(f"  Source            : {audit_path}")

    # Save a summary CSV
    import pandas as pd

    records = []
    for block in audit_data.get("blocks", []):
        for r in block.get("records", []):
            if isinstance(r, dict) and "hash" in r:
                records.append(
                    {
                        "block_index": block["block_index"],
                        "node_id": r.get("node_id", ""),
                        "round_number": r.get("round_number", 0),
                        "hash_prefix": r.get("hash", "")[:16] + "...",
                        "status": r.get("status", ""),
                        "timestamp": r.get("timestamp", ""),
                        "consent_ref": r.get("consent_ref", ""),
                    }
                )

    if records:
        df = pd.DataFrame(records)
        out_csv = output_dir / "blockchain_audit_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"  CSV exported      : {out_csv}")

    import shutil

    out_json = output_dir / "blockchain_audit_full.json"
    shutil.copy(audit_path, out_json)
    print(f"  JSON exported     : {out_json}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
