#!/usr/bin/env python3
"""Run all ablation studies for FedAgent-Chain.

Iterates over ablation config files and runs each experiment,
producing a unified comparison table.

Usage:
    python scripts/run_ablation_study.py \
        --ablation-configs configs/experiment/ablation/ \
        --seed 42 \
        --output-dir experiments/results/ablations/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_utils import get_logger, setup_logging
from src.utils.seed_utils import set_global_seed

logger = get_logger("run_ablation_study")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FedAgent-Chain ablation studies.")
    parser.add_argument(
        "--ablation-configs", type=str, default="configs/experiment/ablation/",
        help="Directory containing ablation YAML files.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="experiments/results/ablations/")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(format="console")
    set_global_seed(args.seed)

    ablation_dir = Path(args.ablation_configs)
    ablation_configs = sorted(ablation_dir.glob("*.yaml"))

    if not ablation_configs:
        logger.warning("No ablation configs found", directory=str(ablation_dir))
        return

    logger.info("Running ablation studies", n_configs=len(ablation_configs))

    for config_path in ablation_configs:
        ablation_name = config_path.stem
        output_dir = Path(args.output_dir) / ablation_name
        logger.info("Starting ablation", ablation=ablation_name)

        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_federated_simulation.py",
                "--config", str(config_path),
                "--seed", str(args.seed),
                "--output-dir", str(output_dir),
                "--no-mlflow",
            ],
            capture_output=False,
        )

        if result.returncode == 0:
            logger.info("Ablation complete", ablation=ablation_name)
        else:
            logger.error("Ablation failed", ablation=ablation_name)

    print(f"\n✅ All ablation studies complete. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
