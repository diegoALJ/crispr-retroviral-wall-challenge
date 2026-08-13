from __future__ import annotations

import numpy as np

from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score


def safe_pr_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if np.unique(y_true).size < 2:
        return np.nan

    return float(
        average_precision_score(y_true, y_prob)
    )


def safe_spearman(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) < 2:
        return np.nan

    corr = spearmanr(y_true, y_pred).correlation

    return float(corr) if corr is not None else np.nan


def to_rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)

    if len(values) == 0:
        return values

    return rankdata(
        values,
        method="average",
    ) / len(values)


def blend_rank_scores(
    p_active: np.ndarray,
    regression_pred: np.ndarray,
    classification_weight: float = 0.35,
    regression_weight: float = 0.65,
) -> np.ndarray:
    total_weight = (
        classification_weight
        + regression_weight
    )

    if not np.isclose(total_weight, 1.0):
        raise ValueError(
            "Classification and regression weights must sum to 1."
        )

    cls_rank = to_rank01(p_active)
    reg_rank = to_rank01(regression_pred)

    return (
        classification_weight * cls_rank
        + regression_weight * reg_rank
    )
