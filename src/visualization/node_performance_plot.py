"""Per-node F1 bar chart — Figure 4 from the FedAgent-Chain paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.styles import FIGSIZE_DOUBLE, apply_fedagent_style


def plot_node_performance(
    node_metrics: dict[str, dict[str, float]],
    node_stds: dict[str, dict[str, float]] | None = None,
    metric: str = "f1",
    title: str = "Per-Node F1 Score by Experimental Setting",
    output_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot grouped bar chart of per-node performance across experiment settings.

    Generates Figure 4 from the paper showing per-node F1 scores for
    Local Baseline, Centralized, Standard FL, and Fairness-Aware FL.

    Parameters
    ----------
    node_metrics : dict
        Mapping: experiment_name → {node_id → metric_value}.
        Example: {"Standard FL": {"saudi_arabia": 0.82, "united_states": 0.85, ...}}
    node_stds : dict, optional
        Mapping: experiment_name → {node_id → std_value}.
        If provided, error bars are drawn.
    metric : str
        Metric name for the y-axis label.
    title : str
        Figure title.
    output_path : str or Path, optional
        Save the figure to this path if provided.
    show : bool
        Display the figure interactively.

    Returns
    -------
    plt.Figure
        Generated matplotlib figure.
    """
    apply_fedagent_style()

    nodes = list(next(iter(node_metrics.values())).keys()) if node_metrics else []
    experiment_names = list(node_metrics.keys())
    n_experiments = len(experiment_names)
    n_nodes = len(nodes)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    bar_width = 0.18
    x = np.arange(n_nodes)

    for i, exp_name in enumerate(experiment_names):
        values = [node_metrics[exp_name].get(node, 0.0) for node in nodes]
        stds = (
            [node_stds[exp_name].get(node, 0.0) for node in nodes]
            if node_stds and exp_name in node_stds
            else None
        )

        offset = (i - n_experiments / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=exp_name,
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": "black"},
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    node_display = [n.replace("_", "\n").title() for n in nodes]
    ax.set_xticks(x)
    ax.set_xticklabels(node_display, fontsize=10)
    ax.set_xlabel("Regional Node", fontsize=12)
    ax.set_ylabel(f"{metric.upper()} Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig
