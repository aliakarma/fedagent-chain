"""Federated learning convergence curve visualizations for FedAgent-Chain.

Generates Figure 3 from the paper: FL convergence curves comparing
Standard FedAvg vs Fairness-Aware FedAvg across training rounds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.styles import COLORS, FIGSIZE_SINGLE, apply_fedagent_style


def plot_convergence_curve(
    round_history: list[dict],
    metric: str = "mean_f1",
    label: str = "FedAgent-Chain",
    ax: plt.Axes | None = None,
    color: str | None = None,
) -> plt.Axes:
    """Plot a single convergence curve from round-level metric history.

    Parameters
    ----------
    round_history : list of dict
        Per-round evaluation results (output of FederatedServer.run()).
    metric : str
        Metric key to plot (e.g., 'mean_f1', 'mean_accuracy').
    label : str
        Legend label for this curve.
    ax : plt.Axes, optional
        Axes to plot on. Creates new figure if None.
    color : str, optional
        Line color. Defaults to palette color.

    Returns
    -------
    plt.Axes
        The axes containing the plot.
    """
    apply_fedagent_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    rounds = [r.get("round", i + 1) for i, r in enumerate(round_history)]
    values = [r.get(metric, float("nan")) for r in round_history]
    std_key = metric.replace("mean_", "std_")
    stds = [r.get(std_key, 0.0) for r in round_history]

    rounds_arr = np.array(rounds)
    values_arr = np.array(values)
    stds_arr = np.array(stds)

    c = color or COLORS["fedagent"]
    ax.plot(rounds_arr, values_arr, label=label, color=c, linewidth=2.0, marker="o", markersize=3)
    ax.fill_between(
        rounds_arr,
        values_arr - stds_arr,
        values_arr + stds_arr,
        alpha=0.15,
        color=c,
    )
    return ax


def plot_multi_convergence(
    histories: dict[str, list[dict]],
    metric: str = "mean_f1",
    title: str = "Federated Learning Convergence",
    ylabel: str = "F1 Score",
    output_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot convergence curves for multiple FL variants on one figure.

    Generates Figure 3 from the paper showing Standard FL vs Fairness-Aware FL.

    Parameters
    ----------
    histories : dict
        Mapping from experiment label to round history list.
    metric : str
        Metric key to compare across variants.
    title : str
        Figure title.
    ylabel : str
        Y-axis label.
    output_path : str or Path, optional
        If provided, save the figure to this path.
    show : bool
        If True, display the figure interactively.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    apply_fedagent_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    color_cycle = list(COLORS.values())
    for i, (label, history) in enumerate(histories.items()):
        plot_convergence_curve(
            history, metric=metric, label=label, ax=ax, color=color_cycle[i % len(color_cycle)]
        )

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0.0)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_fairness_disparity_over_rounds(
    histories: dict[str, list[dict]],
    attribute: str = "disability",
    output_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot fairness disparity D_fair over training rounds.

    Parameters
    ----------
    histories : dict
        Mapping from experiment label to round history.
    attribute : str
        Protected attribute name (suffix of metric key).
    output_path : str or Path, optional
        Save path for the figure.
    show : bool
        Whether to display the figure.

    Returns
    -------
    plt.Figure
        The generated figure.
    """
    metric_key = f"mean_fairness_disparity_{attribute}"
    return plot_multi_convergence(
        histories=histories,
        metric=metric_key,
        title=f"Fairness Disparity Over Rounds ({attribute.replace('_', ' ').title()})",
        ylabel=r"$D_{\mathrm{fair}}$ (lower = fairer)",
        output_path=output_path,
        show=show,
    )
