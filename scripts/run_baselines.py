#!/usr/bin/env python3
"""Run local and centralized baseline experiments for FedAgent-Chain.

Usage:
    python scripts/run_baselines.py \
        --config configs/experiment/baseline_local.yaml --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed_utils import set_global_seed

logger = get_logger("run_baselines")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedAgent-Chain baseline experiments.")
    parser.add_argument("--config", type=str, required=True, help="Baseline config YAML.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(format="console")
    set_global_seed(args.seed)

    cfg = load_config(args.config)
    exp_name = cfg.get("experiment", {}).get("name", "baseline")
    logger.info("Running baseline", experiment=exp_name, seed=args.seed)

    # For baselines, we import and reuse the federated simulation infrastructure
    # but with a single round and no aggregation.
    from scripts.run_federated_simulation import main as run_main
    import sys
    sys.argv = [
        "run_federated_simulation.py",
        "--config", args.config,
        "--seed", str(args.seed),
        "--no-fairness",
        "--no-mlflow",
    ]
    if args.output_dir:
        sys.argv += ["--output-dir", args.output_dir]

    run_main()


if __name__ == "__main__":
    main()
