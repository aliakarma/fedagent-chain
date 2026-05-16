#!/usr/bin/env python3
"""Generate systems overhead plots and CSVs for FedAgent-Chain.

Reads per_round.json from the latest simulation run and produces:
- experiments/results/plots/runtime_breakdown.pdf
- experiments/results/plots/communication_costs.pdf
- experiments/results/system_overhead.csv
- experiments/results/communication_costs.csv
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=str, default="experiments/runs/")
    parser.add_argument("--results-dir", type=str, default="experiments/results/")
    return parser.parse_args()

def find_latest_run(runs_dir: Path) -> Path | None:
    runs = sorted(runs_dir.glob("fedagent_chain_full_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None

def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    latest_run = find_latest_run(runs_dir)
    if not latest_run:
        print("No simulation runs found.")
        return

    metrics_path = latest_run / "metrics" / "per_round.json"
    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}")
        return

    with open(metrics_path, encoding="utf-8") as f:
        history = json.load(f)

    rounds = [r["round"] for r in history]
    local_times = [r["mean_time_local_training"] for r in history]
    agg_times = [r["time_aggregation"] for r in history]
    
    # Calculate blockchain time (mean across nodes)
    bc_times = []
    for r in history:
        node_bc = [n.get("time_blockchain", 0.0) for n in r["per_node"].values()]
        bc_times.append(np.mean(node_bc))

    # 1. Runtime Breakdown Plot
    plt.figure(figsize=(10, 6))
    plt.bar(rounds, local_times, label="Local Training")
    plt.bar(rounds, agg_times, bottom=local_times, label="Aggregation")
    plt.bar(rounds, bc_times, bottom=np.array(local_times) + np.array(agg_times), label="Blockchain Logging")
    
    plt.xlabel("Round")
    plt.ylabel("Duration (seconds)")
    plt.title("System Runtime Breakdown per Round")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(plots_dir / "runtime_breakdown.pdf", bbox_inches="tight")
    plt.close()

    # 2. Communication Costs Plot
    bytes_transmitted = [r["bytes_transmitted"] for r in history]
    cumulative_mb = np.cumsum(bytes_transmitted) / (1024 * 1024)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, cumulative_mb, marker='o', linestyle='-', color='teal', label="Cumulative MB")
    plt.xlabel("Round")
    plt.ylabel("Total Data Transmitted (MB)")
    plt.title("Federated Communication Volume")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(plots_dir / "communication_costs.pdf", bbox_inches="tight")
    plt.close()

    # 3. Export CSVs
    overhead_df = pd.DataFrame({
        "Metric": ["Avg Local Training Time", "Avg Aggregation Time", "Avg Blockchain Logging Time", "Model Size (KB)"],
        "Value": [
            f"{np.mean(local_times):.2f}s",
            f"{np.mean(agg_times):.4f}s",
            f"{np.mean(bc_times):.4f}s",
            f"{history[0]['model_size_kb']:.1f} KB"
        ]
    })
    overhead_df.to_csv(results_dir / "system_overhead.csv", index=False)

    comm_df = pd.DataFrame({
        "Round": rounds,
        "Bytes_Transmitted": bytes_transmitted,
        "Cumulative_MB": cumulative_mb
    })
    comm_df.to_csv(results_dir / "communication_costs.csv", index=False)

    print(f"\n[OK] Profiling results exported to {results_dir}")
    print(f"  - Plots saved in {plots_dir}")

if __name__ == "__main__":
    main()
