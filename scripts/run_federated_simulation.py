#!/usr/bin/env python3
"""Main federated simulation entry point for FedAgent-Chain.

Runs the complete federated learning simulation across all four regional nodes
with configurable privacy, fairness, and blockchain settings.

Usage:
    python scripts/run_federated_simulation.py \
        --config configs/experiment/fedagent_chain_full.yaml \
        --seed 42

    # With hyperparameter overrides:
    python scripts/run_federated_simulation.py \
        --config configs/experiment/fedagent_chain_full.yaml \
        --seed 42 \
        --override federated.n_rounds=30 privacy.noise_multiplier=0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow
import pandas as pd
from omegaconf import OmegaConf

from src.blockchain.chain import PermissionedBlockchain
from src.data.dataset import EmploymentDataset
from src.data.synthetic_generator import generate_synthetic_node_data
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.utils.config import config_to_dict, get_git_commit_hash, load_config
from src.utils.io_utils import ensure_dir, save_json
from src.utils.logging_utils import get_logger, log_hardware_info, setup_logging
from src.utils.seed_utils import set_global_seed

logger = get_logger("run_federated_simulation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FedAgent-Chain federated learning simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/experiment/fedagent_chain_full.yaml",
        help="Experiment configuration YAML.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--data-dir", type=str, default="data/synthetic/",
        help="Directory containing synthetic data CSVs.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results. Auto-named if not provided.",
    )
    parser.add_argument(
        "--no-fairness", action="store_true",
        help="Disable fairness-aware aggregation (use standard FedAvg).",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow experiment tracking.",
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Override config values as key=value pairs.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_node_dataset(
    node_id: str,
    data_dir: Path,
    cfg: object,
    seed: int,
) -> tuple:
    """Load or generate dataset for a single node and return train/test split."""
    node_dir = data_dir / node_id
    users_csv = node_dir / "users.csv"
    jobs_csv = node_dir / "jobs.csv"
    outcomes_csv = node_dir / "outcomes.csv"

    if users_csv.exists() and jobs_csv.exists() and outcomes_csv.exists():
        logger.info("Loading existing dataset", node_id=node_id)
        users_df = pd.read_csv(users_csv)
        jobs_df = pd.read_csv(jobs_csv)
        outcomes_df = pd.read_csv(outcomes_csv)
    else:
        logger.info("Generating synthetic dataset", node_id=node_id)
        data = generate_synthetic_node_data(
            node_id=node_id, n_users=2500, n_jobs=1250, n_pairs=12500, seed=seed
        )
        users_df, jobs_df, outcomes_df = (
            data["users"], data["jobs"], data["outcomes"]
        )

    full_dataset = EmploymentDataset(
        outcomes_df=outcomes_df,
        users_df=users_df,
        jobs_df=jobs_df,
        consent_filter=True,
    )
    train_ds, test_ds = full_dataset.split(test_size=0.20, seed=seed)
    return train_ds, test_ds


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level, format="console")

    # Set seed before any stochastic operation
    set_global_seed(args.seed)

    # Load and optionally override config
    cfg = load_config(args.config)
    for override in args.override:
        key, _, value = override.partition("=")
        OmegaConf.update(cfg, key, value, merge=True)

    # Create run output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("experiment", {}).get("name", "fedagent_chain")
    run_name = f"{exp_name}_seed{args.seed}_{timestamp}"
    output_dir = Path(args.output_dir or f"experiments/runs/{run_name}")
    ensure_dir(output_dir)

    # Save resolved config and hardware info
    save_json(config_to_dict(cfg), output_dir / "config.json")
    hw_info = log_hardware_info()
    save_json(hw_info, output_dir / "hardware.json")
    save_json({"git_commit": get_git_commit_hash(), "seed": args.seed}, output_dir / "provenance.json")

    logger.info("Simulation started", run_name=run_name, output_dir=str(output_dir))

    # Initialise shared blockchain
    blockchain_enabled = cfg.get("blockchain", {}).get("enabled", True)
    blockchain = PermissionedBlockchain(
        records_per_block=cfg.get("blockchain", {}).get("records_per_block", 10),
        storage_path=output_dir / "blockchain_logs" if blockchain_enabled else None,
    )

    # Load datasets for all nodes
    data_dir = Path(args.data_dir)
    nodes = cfg.get("data", {}).get("nodes", ["saudi_arabia", "united_states", "china", "europe"])

    clients = []
    for i, node_id in enumerate(nodes):
        train_ds, test_ds = load_node_dataset(
            node_id, data_dir, cfg, seed=args.seed + i * 1000
        )
        client = FederatedClient(
            node_id=node_id,
            train_dataset=train_ds,
            test_dataset=test_ds,
            cfg=cfg,
            blockchain=blockchain,
            device="cpu",
        )
        clients.append(client)
        logger.info(
            "Client created",
            node_id=node_id,
            n_train=len(train_ds),
            n_test=len(test_ds),
        )

    # Create server and run simulation
    use_mlflow = not args.no_mlflow and cfg.get("tracking", {}).get("use_mlflow", True)
    use_fairness = not args.no_fairness and cfg.get("fairness", {}).get("enabled", True)

    if use_mlflow:
        mlflow_uri = cfg.get("tracking", {}).get("mlflow_tracking_uri", "http://localhost:5000")
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(exp_name)
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({"seed": args.seed, "git_commit": get_git_commit_hash()})
        except Exception as e:
            logger.warning("MLflow connection failed, continuing without tracking", error=str(e))
            use_mlflow = False

    try:
        server = FederatedServer(
            cfg=cfg,
            use_fairness_aggregation=use_fairness,
            output_dir=output_dir,
        )

        start_time = time.time()
        results = server.run(clients=clients, seed=args.seed, use_mlflow=use_mlflow)
        total_duration = time.time() - start_time

        results["total_duration_seconds"] = round(total_duration, 2)
        save_json(results, output_dir / "metrics" / "final.json")

        # Export blockchain audit log
        if blockchain_enabled:
            blockchain.export_audit_log(output_dir / "blockchain_logs" / "audit_trail.json")
            hash_completeness = blockchain.get_hash_completeness()
            results["blockchain_hash_completeness"] = hash_completeness
            logger.info(
                "Blockchain audit",
                n_records=blockchain.get_record_count(),
                hash_completeness=hash_completeness,
                chain_valid=blockchain.verify_chain_integrity(),
            )

        logger.info(
            "Simulation complete",
            total_duration_s=round(total_duration, 2),
            best_f1=round(results.get("best_f1", 0.0), 4),
            output_dir=str(output_dir),
        )

        print(f"\n{'='*60}")
        print(f" [OK] Simulation complete: {run_name}")
        print(f"{'='*60}")
        print(f"  Best F1        : {results.get('best_f1', 0.0):.4f} (round {results.get('best_f1_round', 0)})")
        print(f"  Total rounds   : {results.get('convergence_rounds', 0)}")
        print(f"  Duration       : {total_duration:.1f}s")
        print(f"  Results saved  : {output_dir}")
        if blockchain_enabled:
            print(f"  Blockchain records: {blockchain.get_record_count()}")
            print(f"  Hash completeness : {blockchain.get_hash_completeness():.1%}")
        print(f"{'='*60}\n")

    finally:
        if use_mlflow:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()
