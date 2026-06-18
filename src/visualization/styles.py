"""Consistent plot styles and color palette for FedAgent-Chain figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# FedAgent-Chain brand color palette
COLORS = {
    "fedagent": "#2563EB",  # Primary blue — FedAgent-Chain full model
    "baseline_local": "#DC2626",  # Red — local-only baseline
    "baseline_central": "#16A34A",  # Green — centralized baseline
    "fairness": "#7C3AED",  # Purple — fairness-aware variant
    "no_blockchain": "#EA580C",  # Orange — ablation (no blockchain)
    "no_governance": "#0891B2",  # Cyan — ablation (no governance)
    "accent": "#F59E0B",  # Amber — accent highlights
}

# Consistent figure sizes (width, height) in inches
FIGSIZE_SINGLE = (7, 4.5)
FIGSIZE_DOUBLE = (12, 4.5)
FIGSIZE_SQUARE = (6, 6)

# Node colors for per-node bar charts
NODE_COLORS = {
    "saudi_arabia": "#2563EB",
    "united_states": "#DC2626",
    "china": "#16A34A",
    "europe": "#7C3AED",
}


def apply_fedagent_style() -> None:
    """Apply the FedAgent-Chain publication-grade matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )
    sns.set_palette(list(COLORS.values()))
