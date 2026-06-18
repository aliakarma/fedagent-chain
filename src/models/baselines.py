from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import EmploymentDataset, encode_user_job_pair


def build_feature_matrix(dataset: EmploymentDataset) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)`` for a dataset using the shared 91-dim encoding.

    Iterates over every consented user-job pair and stacks the encoded feature
    vectors (identical to those fed to the neural model) into a dense matrix.

    Parameters
    ----------
    dataset : EmploymentDataset
        Dataset whose ``outcomes`` rows are encoded.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Feature matrix ``X`` of shape ``(n, 91)`` and integer label vector ``y``.
    """
    feats: list[np.ndarray] = []
    labels: list[int] = []
    for _, row in dataset.outcomes.iterrows():
        user_row = dataset.users_df.loc[row["user_id"]]
        job_row = dataset.jobs_df.loc[row["job_id"]]
        feats.append(encode_user_job_pair(user_row, job_row))
        labels.append(int(row["suitability_label"]))
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    return X, y


class SklearnProbWrapper:
    """Adapt a fitted scikit-learn classifier to the torch ``model(x)`` API.

    Calling the wrapper with a feature tensor returns a ``(batch, 1)`` tensor of
    positive-class probabilities, matching :class:`EmploymentMatchingModel` so the
    same ``evaluate_model_on_node`` routine can be reused.
    """

    def __init__(self, clf) -> None:
        self.clf = clf

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        x = features.detach().cpu().numpy()
        probs = self.clf.predict_proba(x)[:, 1]
        return torch.from_numpy(probs.astype(np.float32)).unsqueeze(-1)

    def eval(self) -> SklearnProbWrapper:  # torch-API compatibility (no-op)
        return self

    def train(self, mode: bool = True) -> SklearnProbWrapper:  # no-op
        return self


def train_centralized_lr(train_dataset: EmploymentDataset, seed: int = 42):
    """Fit a logistic-regression baseline on the pooled training set.

    Parameters
    ----------
    train_dataset : EmploymentDataset
        Pooled (all-node) training dataset.
    seed : int
        Random seed for the solver.

    Returns
    -------
    SklearnProbWrapper
        Wrapped, fitted logistic-regression classifier.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = build_feature_matrix(train_dataset)
    clf = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(max_iter=1000, random_state=seed, n_jobs=None),
    )
    clf.fit(X, y)
    return SklearnProbWrapper(clf)
