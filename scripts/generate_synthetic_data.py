#!/usr/bin/env python3
"""Generate synthetic disability-employment dataset for FedAgent-Chain.

This script creates the primary evaluation dataset used in the paper.
All generated records are completely fictitious (no real personal data).

Usage:
    python scripts/generate_synthetic_data.py \
        --config configs/experiment/fedagent_chain_full.yaml \
        --seed 42 \
        --output-dir data/synthetic/

The generated dataset follows the schema in data/synthetic/README.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic_generator import save_synthetic_dataset
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.seed_utils import set_global_seed

logger = get_logger("generate_synthetic_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic disability-employment dataset for FedAgent-Chain.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/fedagent_chain_full.yaml",
        help="Path to experiment configuration YAML file.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/synthetic/",
        help="Directory to write generated CSV files.",
    )
    parser.add_argument(
        "--n-users-per-node", type=int, default=None,
        help="Override: number of user profiles per node.",
    )
    parser.add_argument(
        "--n-jobs-per-node", type=int, default=None,
        help="Override: number of job profiles per node.",
    )
    parser.add_argument(
        "--n-pairs-per-node", type=int, default=None,
        help="Override: number of suitability pairs per node.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, format="console")
    set_global_seed(args.seed)

    cfg = load_config(args.config)

    n_users = args.n_users_per_node or int(cfg.get("data", {}).get("n_users_per_node", 2500))
    n_jobs = args.n_jobs_per_node or int(cfg.get("data", {}).get("n_jobs_per_node", 1250))
    n_pairs = args.n_pairs_per_node or int(cfg.get("data", {}).get("n_pairs_per_node", 12500))

    logger.info(
        "Starting synthetic data generation",
        seed=args.seed,
        output_dir=args.output_dir,
        n_users_per_node=n_users,
        n_jobs_per_node=n_jobs,
        n_pairs_per_node=n_pairs,
    )

    save_synthetic_dataset(
        output_dir=args.output_dir,
        n_users_per_node=n_users,
        n_jobs_per_node=n_jobs,
        n_pairs_per_node=n_pairs,
        seed=args.seed,
    )

    logger.info(
        "Synthetic data generation complete",
        output_dir=args.output_dir,
        total_users=n_users * 4,
        total_jobs=n_jobs * 4,
        total_pairs=n_pairs * 4,
    )
    print(f"\n[OK] Synthetic dataset saved to: {args.output_dir}")
    print(f"   Total users  : {n_users * 4:,}")
    print(f"   Total jobs   : {n_jobs * 4:,}")
    print(f"   Total pairs  : {n_pairs * 4:,}")
    print(f"\n   Schema documented in: data/synthetic/README.md")


if __name__ == "__main__":
    main()
