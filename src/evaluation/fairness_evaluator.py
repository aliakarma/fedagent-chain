"""Fairness disparity computation across protected groups.

Implements the fairness evaluation methodology from Section 5.3 of the paper.
Computes D_fair across disability categories, language groups, work modes,
and regional nodes.
"""

from __future__ import annotations

import pandas as pd

from src.federated.fairness import compute_fairness_disparity, group_performance_from_predictions
from src.utils.logging_utils import get_logger

logger = get_logger("FairnessEvaluator")

# Protected attributes evaluated in the paper (Table 3)
PROTECTED_ATTRIBUTES = ["disability_category", "language_primary", "preferred_work_mode", "node_id"]


class FairnessEvaluator:
    """Evaluate fairness disparity across all protected groups.

    Parameters
    ----------
    protected_attributes : list of str
        Column names in the DataFrame to evaluate fairness across.
    metric : str
        Performance metric to use for disparity computation ('f1', 'accuracy', etc.).
    """

    def __init__(
        self,
        protected_attributes: list[str] = PROTECTED_ATTRIBUTES,
        metric: str = "f1",
    ) -> None:
        self.protected_attributes = protected_attributes
        self.metric = metric

    def evaluate(
        self,
        df: pd.DataFrame,
        y_true_col: str = "suitability_label",
        y_pred_col: str = "predicted_label",
    ) -> dict[str, float]:
        """Compute fairness disparity across all protected attribute groups.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing predictions, ground truth, and group labels.
        y_true_col : str
            Column name for ground truth labels.
        y_pred_col : str
            Column name for predicted labels.

        Returns
        -------
        dict
            Mapping from attribute name to fairness disparity D_fair.
        """
        y_true = df[y_true_col].values
        y_pred = df[y_pred_col].values

        per_attribute_disparities = {}

        for attr in self.protected_attributes:
            if attr not in df.columns:
                logger.warning("Protected attribute not found in DataFrame", attribute=attr)
                continue

            group_labels = df[attr].values
            group_f1s = group_performance_from_predictions(y_true, y_pred, group_labels)

            if len(group_f1s) < 2:
                logger.debug("Skipping disparity (fewer than 2 groups)", attribute=attr)
                continue

            disparity = compute_fairness_disparity(group_f1s, metric_name=attr)
            per_attribute_disparities[attr] = disparity

            logger.info(
                "Fairness disparity computed",
                attribute=attr,
                disparity=round(disparity, 4),
                n_groups=len(group_f1s),
                group_scores={k: round(v, 4) for k, v in group_f1s.items()},
            )

        return per_attribute_disparities

    def evaluate_across_rounds(
        self,
        round_results: list[dict],
    ) -> pd.DataFrame:
        """Track fairness disparity across federated learning rounds.

        Parameters
        ----------
        round_results : list of dict
            List of per-round evaluation results, each containing
            per-attribute disparity values.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: round, attribute, disparity.
        """
        rows = []
        for round_data in round_results:
            round_num = round_data.get("round", 0)
            for attr, disparity in round_data.get("disparities", {}).items():
                rows.append({"round": round_num, "attribute": attr, "disparity": disparity})
        return pd.DataFrame(rows)

    def generate_fairness_report(
        self,
        df: pd.DataFrame,
        y_true_col: str = "suitability_label",
        y_pred_col: str = "predicted_label",
    ) -> dict:
        """Generate a comprehensive fairness evaluation report.

        Parameters
        ----------
        df : pd.DataFrame
            Predictions and group labels DataFrame.
        y_true_col : str
            Ground truth label column.
        y_pred_col : str
            Predicted label column.

        Returns
        -------
        dict
            Report with per-attribute disparity values and group-level metrics.
        """
        y_true = df[y_true_col].values
        y_pred = df[y_pred_col].values

        report: dict = {
            "overall_disparity": {},
            "group_metrics": {},
            "intersectional_analysis": {},
        }

        for attr in self.protected_attributes:
            if attr not in df.columns:
                continue

            group_labels = df[attr].values
            group_f1s = group_performance_from_predictions(y_true, y_pred, group_labels)

            if len(group_f1s) >= 2:
                report["overall_disparity"][attr] = compute_fairness_disparity(group_f1s)
                report["group_metrics"][attr] = group_f1s

        return report
