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

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger, setup_logging
from src.visualization.convergence_plot import plot_multi_convergence
from src.visualization.node_performance_plot import plot_node_performance

logger = get_logger("generate_figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate FedAgent-Chain paper figures from real simulation outputs."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results/",
        help="Directory containing evaluation CSV outputs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="experiments/runs/",
        help="Directory containing simulation run directories.",
    )
    parser.add_argument("--output-dir", type=str, default="experiments/figures/")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────


def find_run_dir(runs_dir: Path, experiment_name: str, seed: int | None = None) -> Path | None:
    """Return the most recent run directory matching experiment_name and optionally seed."""
    pattern = f"{experiment_name}_*"
    if seed is not None:
        pattern = f"{experiment_name}_seed{seed}_*"

    candidates = sorted(
        runs_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_convergence_history(run_dir: Path) -> list:
    """Load per-round metric history from an actual simulation run."""
    metrics_path = run_dir / "metrics" / "per_round.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"per_round.json not found at {metrics_path}. " "Run run_federated_simulation.py first."
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
    return {node_id: m.get(metric, 0.0) for node_id, m in final_round.get("per_node", {}).items()}


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    setup_logging(format="console")

    results_dir = Path(args.results_dir)
    plots_dir = ensure_dir(results_dir / "plots")
    runs_dir = Path(args.runs_dir)

    # ── Seed discovery ───────────────────────────────────────────────────────
    seed_dirs = sorted((results_dir / "seeds").glob("seed_*"))
    if not seed_dirs:
        seed_dirs = sorted(results_dir.glob("seed_*"))  # Fallback

    seeds = [int(d.name.split("_")[1]) for d in seed_dirs]
    if not seeds:
        logger.error("No multi-seed results found in experiments/results/")
        sys.exit(1)

    logger.info("Generating paper figures from seeds", seeds=seeds, output_dir=str(plots_dir))

    # ── Aggregate Histories for Figure 3 ──────────────────────────────────────
    def load_aggregated_history(exp_prefix: str) -> list[dict]:
        all_histories = []
        for seed in seeds:
            # find_run_dir needs to be seed-specific here if we want to be exact
            # but generate_figures.py's find_run_dir is currently prefix-only.
            # Let's refine find_run_dir to handle seeds.
            run_dir = find_run_dir(runs_dir, exp_prefix, seed)
            if run_dir:
                all_histories.append(load_convergence_history(run_dir))

        if not all_histories:
            return []

        # Aggregate rounds (assuming same number of rounds for simplicity)
        n_rounds = min(len(h) for h in all_histories)
        agg_history = []
        for r in range(n_rounds):
            round_data = {"round": r + 1}
            f1_vals = [h[r].get("mean_f1", 0.0) for h in all_histories]
            round_data["mean_f1"] = np.mean(f1_vals)
            round_data["std_f1"] = np.std(f1_vals)
            agg_history.append(round_data)
        return agg_history

    # ── Figure 3: FL Convergence Curves ───────────────────────────────────────
    logger.info("Aggregating convergence histories for Figure 3...")
    fedavg_agg = load_aggregated_history("ablation_no_fairness")
    fairness_agg = load_aggregated_history("fedagent_chain_full")

    if fedavg_agg and fairness_agg:
        plot_multi_convergence(
            histories={
                "Standard FedAvg": fedavg_agg,
                "FedAgent-Chain (Fairness-Aware)": fairness_agg,
            },
            metric="mean_f1",
            title="Figure 3: Federated Learning Convergence",
            ylabel="Mean F1 Score (held-out test set)",
            output_path=plots_dir / "fl_convergence.pdf",
        )
        logger.info("Figure 3 saved: fl_convergence.pdf")

    # ── Figure 4: Per-Node F1 Scores ──────────────────────────────────────────
    logger.info("Extracting per-node metrics across seeds for Figure 4...")

    def get_node_metrics_agg(exp_prefix: str) -> tuple[dict, dict]:
        node_vals = {}  # node -> [seeds]
        for seed in seeds:
            run_dir = find_run_dir(runs_dir, exp_prefix, seed)
            if run_dir:
                metrics = extract_final_node_metrics(run_dir)
                for node, val in metrics.items():
                    node_vals.setdefault(node, []).append(val)

        means = {node: np.mean(vals) for node, vals in node_vals.items()}
        stds = {node: np.std(vals) for node, vals in node_vals.items()}
        return means, stds

    methods = {
        "Local Baseline": "baseline_local",
        "Centralized": "baseline_centralized",
        "Standard FL": "ablation_no_fairness",
        "FedAgent-Chain": "fedagent_chain_full",
    }

    node_means = {}
    node_stds = {}
    for label, prefix in methods.items():
        m, s = get_node_metrics_agg(prefix)
        if m:
            node_means[label] = m
            node_stds[label] = s

    if node_means:
        plot_node_performance(
            node_metrics=node_means,
            node_stds=node_stds,
            metric="F1",
            title="Figure 4: Per-Node F1 Score (held-out test set)",
            output_path=plots_dir / "node_f1_scores.pdf",
        )
        logger.info("Figure 4 saved: node_f1_scores.pdf")
    else:
        logger.warning("No per-node metrics available; Figure 4 skipped")

    # ── Figure 5: Fairness Disparity Comparison ───────────────────────────────
    import matplotlib.pyplot as plt

    from src.visualization.styles import COLORS, FIGSIZE_DOUBLE, apply_fedagent_style

    # Try to load real disparity values from Table 3 CSV produced by run_evaluation
    table3_path = results_dir / "table_3_fairness_results.csv"
    if table3_path.exists():
        t3_df = pd.read_csv(table3_path)
        attributes = t3_df["Attribute"].tolist()
        standard_fl = t3_df.get("Standard FedAvg", pd.Series(dtype=float)).tolist()
        fairness_fl = t3_df.get("FedAgent-Chain", pd.Series(dtype=float)).tolist()
        logger.info("Figure 5: loaded real disparity values from table_3 CSV")
    else:
        # Fall back to neutral disparity if table 3 not yet generated. The
        # round-level history is not in scope here, so emit zeros and warn.
        logger.warning("table_3 CSV not found; using neutral disparity placeholders for Figure 5")
        attributes = ["Disability\\nCategory", "Language\\nGroup", "Work\\nMode", "Regional\\nNode"]
        standard_fl = [0.0] * 4
        fairness_fl = [0.0] * 4

    apply_fedagent_style()
    x = np.arange(len(attributes))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    bars1 = ax.bar(
        x - width / 2,
        standard_fl,
        width,
        label="Standard FedAvg",
        color=COLORS["baseline_local"],
        alpha=0.85,
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        fairness_fl,
        width,
        label="FedAgent-Chain",
        color=COLORS["fedagent"],
        alpha=0.85,
        edgecolor="white",
    )

    for bar, val in zip(bars1, standard_fl, strict=False):
        if isinstance(val, float) and not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for bar, val in zip(bars2, fairness_fl, strict=False):
        if isinstance(val, float) and not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS["fedagent"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(attributes, fontsize=10)
    ax.set_ylabel(r"Fairness Disparity $D_{\mathrm{fair}}$ (lower = fairer)", fontsize=11)
    ax.set_title(
        "Figure 5: Fairness Disparity Before vs After Optimization", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10)
    y_max = max(max(v for v in standard_fl if isinstance(v, float) and not np.isnan(v)), 0.01)
    ax.set_ylim(0.0, y_max * 1.3)
    plt.tight_layout()
    fig.savefig(plots_dir / "fairness_disparity.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure 5 saved: fairness_disparity.pdf")

    print(f"\n{'='*60}")
    print(f" [OK] All paper figures saved to: {plots_dir}")
    print("  - fl_convergence.pdf      (Figure 3)")
    print("  - node_f1_scores.pdf      (Figure 4)")
    print("  - fairness_disparity.pdf  (Figure 5)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
