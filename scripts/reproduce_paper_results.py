#!/usr/bin/env python3
"""One-click reproduction script for FedAgent-Chain paper results."""
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, description):
    print(f"\n>>> {description}...")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {description} failed.")
        sys.exit(1)

def main():
    root = Path(__file__).parent.parent
    
    # 1. Data Generation
    run_cmd([
        sys.executable, "scripts/generate_synthetic_data.py",
        "--config", "configs/experiment/fedagent_chain_full.yaml",
        "--seed", "42"
    ], "Step 1/5: Generating Synthetic Dataset")

    # 2. Main Simulation (FedAgent-Chain)
    run_cmd([
        sys.executable, "scripts/run_federated_simulation.py",
        "--config", "configs/experiment/fedagent_chain_full.yaml",
        "--seed", "42", "--no-mlflow"
    ], "Step 2/5: Running FedAgent-Chain Simulation")

    # 3. Ablation (Standard FedAvg)
    run_cmd([
        sys.executable, "scripts/run_federated_simulation.py",
        "--config", "configs/experiment/ablation/no_fairness.yaml",
        "--seed", "42", "--no-mlflow"
    ], "Step 3/5: Running Standard FedAvg Ablation")

    # 4. Baselines (Local & Centralized)
    run_cmd([
        sys.executable, "scripts/run_federated_simulation.py",
        "--config", "configs/experiment/baseline_local.yaml",
        "--seed", "42", "--no-mlflow"
    ], "Step 4.1/5: Running Local Baseline")
    
    run_cmd([
        sys.executable, "scripts/run_federated_simulation.py",
        "--config", "configs/experiment/baseline_centralized.yaml",
        "--seed", "42", "--no-mlflow"
    ], "Step 4.2/5: Running Centralized Baseline")

    # 5. Evaluation & Aggregation
    run_cmd([
        sys.executable, "scripts/run_evaluation.py",
        "--seed", "42"
    ], "Step 5/5: Evaluating Results & Generating Tables")

    print("\n" + "="*60)
    print("SUCCESS: Paper results reproduced for Seed 42.")
    print("Check 'experiments/results/' for CSV tables and 'experiments/figures/' for plots.")
    print("="*60)

if __name__ == "__main__":
    main()
