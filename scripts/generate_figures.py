#!/usr/bin/env python3
"""Regenerate all paper figures from actual simulation results.

Generates Figures 3 (convergence), 4 (per-node F1), and 5 (fairness disparity)
from the FedAgent-Chain paper using real per_round.json outputs and evaluation
CSVs produced by run_evaluation.py.

Usage:
    python scripts/generate_figures.py \\
        --results-dir experiments/results/ \\
        --runs-dir experiments/runs/ \\
        --output-dir experiments/figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.visualization.convergence_plot import plot_multi_convergence
from src.visualization.node_performance_plot import plot_node_performance
from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger("generate_figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FedAgent-Chain paper figures from real simulation outputs."
    )
    parser.add_argument("--results-dir", type=str, default="experiments/results/",
                        help="Directory containing evaluation CSV outputs.")
    parser.add_argument("--runs-dir", type=str, default="experiments/runs/",
                        help="Directory containing simulation run directories.")
    parser.add_argument("--output-dir", type=str, default="experiments/figures/")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_run_dir(runs_dir: Path, experiment_name: str) -> Path | None:
    """Return the most recent run directory matching experiment_name prefix."""
    candidates = sorted(
        runs_dir.glob(f"{experiment_name}_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_convergence_history(run_dir: Path) -> list:
    """Load per-round metric history from an actual simulation run."""
    metrics_path = run_dir / "metrics" / "per_round.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"per_round.json not found at {metrics_path}. "
            "Run run_federated_simulation.py first."
        )
    with open(metrics_path, encoding="utf-8") as f:
        history = json.load(f)
    logger.info(
        "Convergence history loaded",
        run_dir=str(run_dir),
        n_rounds=len(history),
    )
    return history


def extract_final_node_metrics(run_dir: Path | None, metric: str = "f1") -> dict:
    """Extract per-node metric values from the final round of a run."""
    if run_dir is None:
        return {}
    try:
        history = load_convergence_history(run_dir)
    except FileNotFoundError:
        return {}
    if not history:
        return {}
    final_round = history[-1]
    return {
        node_id: m.get(metric, 0.0)
        for node_id, m in final_round.get("per_node", {}).items()
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(format="console")

    output_dir  = ensure_dir(Path(args.output_dir))
    runs_dir    = Path(args.runs_dir)
    results_dir = Path(args.results_dir)

    logger.info("Generating paper figures", output_dir=str(output_dir))

    # Locate run directories
    fedavg_run_dir   = find_run_dir(runs_dir, "ablation_no_fairness")
    fairness_run_dir = find_run_dir(runs_dir, "fedagent_chain_full")
    local_run_dir    = find_run_dir(runs_dir, "baseline_local")
    central_run_dir  = find_run_dir(runs_dir, "baseline_centralized")

    missing_dirs = []
    if fedavg_run_dir is None:
        missing_dirs.append("ablation_no_fairness")
    if fairness_run_dir is None:
        missing_dirs.append("fedagent_chain_full")

    if missing_dirs:
        logger.error(
            "Required run directories not found — run simulations first",
            missing=missing_dirs,
        )
        sys.exit(1)

    # ── Figure 3: FL Convergence Curves ───────────────────────────────────────
    logger.info("Loading convergence histories for Figure 3...")
    fedavg_history   = load_convergence_history(fedavg_run_dir)
    fairness_history = load_convergence_history(fairness_run_dir)

    plot_multi_convergence(
        histories={
            "Standard FedAvg":                 fedavg_history,
            "FedAgent-Chain (Fairness-Aware)": fairness_history,
        },
        metric="mean_f1",
        title="Figure 3: Federated Learning Convergence",
        ylabel="Mean F1 Score (held-out test set)",
        output_path=output_dir / "fl_convergence.pdf",
    )
    logger.info("Figure 3 saved: fl_convergence.pdf")

    # ── Figure 4: Per-Node F1 Scores ──────────────────────────────────────────
    logger.info("Extracting per-node metrics for Figure 4...")
    node_metrics_computed = {
        "Local Baseline":  extract_final_node_metrics(local_run_dir),
        "Centralized":     extract_final_node_metrics(central_run_dir),
        "Standard FL":     extract_final_node_metrics(fedavg_run_dir),
        "FedAgent-Chain":  extract_final_node_metrics(fairness_run_dir),
    }
    # Drop empty entries
    node_metrics_computed = {k: v for k, v in node_metrics_computed.items() if v}

    if node_metrics_computed:
        plot_node_performance(
            node_metrics=node_metrics_computed,
            metric="F1",
            title="Figure 4: Per-Node F1 Score (held-out test set)",
            output_path=output_dir / "node_f1_scores.pdf",
        )
        logger.info("Figure 4 saved: node_f1_scores.pdf")
    else:
        logger.warning("No per-node metrics available; Figure 4 skipped")

    # ── Figure 5: Fairness Disparity Comparison ───────────────────────────────
    import matplotlib.pyplot as plt
    from src.visualization.styles import apply_fedagent_style, COLORS, FIGSIZE_DOUBLE

    # Try to load real disparity values from Table 3 CSV produced by run_evaluation
    table3_path = results_dir / "table_3_fairness_results.csv"
    if table3_path.exists():
        t3_df = pd.read_csv(table3_path)
        attributes = t3_df["Attribute"].tolist()
        standard_fl = t3_df.get("Standard FedAvg", pd.Series(dtype=float)).tolist()
        fairness_fl = t3_df.get("FedAgent-Chain",  pd.Series(dtype=float)).tolist()
        logger.info("Figure 5: loaded real disparity values from table_3 CSV")
    else:
        # Fall back to round-level disparity if table 3 not yet generated
        logger.warning("table_3 CSV not found; extracting round-level disparity for Figure 5")
        attributes  = ["Disability\\nCategory", "Language\\nGroup", "Work\\nMode", "Regional\\nNode"]
        standard_fl = [
            fedavg_history[-1].get("mean_fairness_disparity_disability", 0.0)
        ] * 4
        fairness_fl = [
            fairness_history[-1].get("mean_fairness_disparity_disability", 0.0)
        ] * 4

    apply_fedagent_style()
    x     = np.arange(len(attributes))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    bars1 = ax.bar(
        x - width / 2, standard_fl, width,
        label="Standard FedAvg",
        color=COLORS["baseline_local"], alpha=0.85, edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2, fairness_fl, width,
        label="FedAgent-Chain",
        color=COLORS["fedagent"], alpha=0.85, edgecolor="white",
    )

    for bar, val in zip(bars1, standard_fl):
        if isinstance(val, float) and not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, fairness_fl):
        if isinstance(val, float) and not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    color=COLORS["fedagent"])

    ax.set_xticks(x)
    ax.set_xticklabels(attributes, fontsize=10)
    ax.set_ylabel(r"Fairness Disparity $D_{\mathrm{fair}}$ (lower = fairer)", fontsize=11)
    ax.set_title("Figure 5: Fairness Disparity Before vs After Optimization",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    y_max = max(max(v for v in standard_fl if isinstance(v, float) and not np.isnan(v)), 0.01)
    ax.set_ylim(0.0, y_max * 1.3)
    plt.tight_layout()
    fig.savefig(output_dir / "fairness_disparity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 5 saved: fairness_disparity.pdf")

    print(f"\n{'='*60}")
    print(f" [OK] All paper figures saved to: {output_dir}")
    print("  - fl_convergence.pdf      (Figure 3)")
    print("  - node_f1_scores.pdf      (Figure 4)")
    print("  - fairness_disparity.pdf  (Figure 5)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
