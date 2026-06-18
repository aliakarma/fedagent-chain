"""Aggregate lambda sweep results across seeds and generate F1 vs D_fair tradeoff plot with CIs."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualization.styles import FIGSIZE_SINGLE, apply_fedagent_style


def main():
    apply_fedagent_style()
    results_dir = Path("experiments/results")
    plots_dir = Path("experiments/results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover Seeds
    seed_dirs = sorted((results_dir / "seeds").glob("seed_*"))
    if not seed_dirs:
        seed_dirs = sorted(results_dir.glob("seed_*"))
    seeds = [int(d.name.split("_")[1]) for d in seed_dirs] or [42]

    # 2. Collect metrics for each lambda across seeds
    all_results = []
    for lam_config in sorted(Path("configs/experiment/lambda_sweep").glob("*.yaml")):
        try:
            lam_str = lam_config.stem.split("_")[-1]
            lam = float(lam_str)
        except (IndexError, ValueError):
            continue

        f1_vals = []
        d_vals = []

        for seed in seeds:
            pattern = f"lambda_sweep_{lam:.2f}_seed{seed}_*"
            runs = sorted(Path("experiments/runs").glob(pattern))
            if not runs:
                continue
            latest = runs[-1]

            final_json = latest / "metrics" / "final.json"
            if final_json.exists():
                with open(final_json) as f:
                    final = json.load(f)
                f1_vals.append(final.get("best_f1", float("nan")))
                d_vals.append(
                    final.get("final_round_metrics", {}).get(
                        "mean_fairness_disparity_disability", float("nan")
                    )
                )
            else:
                per_round_json = latest / "metrics" / "per_round.json"
                if per_round_json.exists():
                    with open(per_round_json) as f:
                        history = json.load(f)
                    if history:
                        f1_vals.append(max(r.get("mean_f1", 0) for r in history))
                        d_vals.append(
                            history[-1].get("mean_fairness_disparity_disability", float("nan"))
                        )

        if f1_vals:
            all_results.append(
                {
                    "lambda": lam,
                    "f1_mean": np.nanmean(f1_vals),
                    "f1_std": np.nanstd(f1_vals),
                    "d_mean": np.nanmean(d_vals),
                    "d_std": np.nanstd(d_vals),
                    "n_seeds": len(f1_vals),
                }
            )

    if not all_results:
        print("Error: No lambda sweep results collected.")
        return

    df = pd.DataFrame(all_results).sort_values("lambda")
    df.to_csv(results_dir / "statistics" / "lambda_tradeoff_multi_seed.csv", index=False)
    print(df.to_string(index=False))

    # 3. Plot Tradeoff with CI Ellipses/Bars
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Scatter points for the means
    scatter = ax.scatter(
        df["d_mean"],
        df["f1_mean"],
        c=df["lambda"],
        cmap="viridis",
        s=100,
        zorder=5,
        edgecolor="white",
        alpha=0.9,
        linewidth=1.5,
    )

    # Error bars for both dimensions
    ax.errorbar(
        df["d_mean"],
        df["f1_mean"],
        xerr=df["d_std"],
        yerr=df["f1_std"],
        fmt="none",
        ecolor="gray",
        alpha=0.4,
        capsize=0,
        zorder=2,
    )

    # Pareto curve
    ax.plot(df["d_mean"], df["f1_mean"], "k--", alpha=0.3, zorder=1, linewidth=1)

    for _, row in df.iterrows():
        ax.annotate(
            f"λ={row['lambda']:.2f}",
            (row["d_mean"], row["f1_mean"]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    plt.colorbar(scatter, label="Fairness Penalty (λ)")
    ax.set_xlabel(r"Fairness Disparity ($D_{\mathrm{fair}}$) — Lower is Better", fontsize=11)
    ax.set_ylabel("Maximum F1 Score — Higher is Better", fontsize=11)
    ax.set_title("Figure 6: Fairness-Accuracy Tradeoff", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.savefig(plots_dir / "lambda_tradeoff_ci.pdf", dpi=300, bbox_inches="tight")
    print(f"Figure saved: {plots_dir / 'lambda_tradeoff_ci.pdf'}")


if __name__ == "__main__":
    main()
