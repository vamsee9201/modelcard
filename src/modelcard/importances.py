"""Extract per-feature importances from a fitted scikit-learn estimator.

Two families are supported:

* **Tree-based** estimators expose ``feature_importances_`` (Gini / impurity
  reduction). These are already non-negative and comparable across features.
* **Linear** estimators expose ``coef_``. We collapse the (possibly
  multi-class) coefficient matrix into a single magnitude per feature by taking
  the mean absolute coefficient across output rows.

Estimators exposing neither return ``None`` — the card then simply omits the
feature-importance section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FeatureImportance:
    """A single feature paired with its importance score."""

    name: str
    score: float


def _tree_importances(estimator: Any) -> np.ndarray | None:
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return None
    return np.asarray(values, dtype=float).ravel()


def _linear_importances(estimator: Any) -> np.ndarray | None:
    coef = getattr(estimator, "coef_", None)
    if coef is None:
        return None
    arr = np.asarray(coef, dtype=float)
    # Binary logistic regression yields shape (1, n_features); multiclass yields
    # (n_classes, n_features); a plain linear model yields (n_features,).
    if arr.ndim == 1:
        return np.abs(arr)
    return np.abs(arr).mean(axis=0)


def extract_importances(
    estimator: Any,
    feature_names: list[str],
    *,
    top_k: int | None = None,
) -> list[FeatureImportance] | None:
    """Return feature importances sorted from most to least important.

    Args:
        estimator: A fitted scikit-learn estimator (or any object exposing
            ``feature_importances_`` or ``coef_``).
        feature_names: Names for each input feature, in column order.
        top_k: If given, keep only the ``top_k`` highest-scoring features.

    Returns:
        A list of :class:`FeatureImportance`, or ``None`` if the estimator
        exposes no supported importance attribute.

    Raises:
        ValueError: If the estimator's importance vector length does not match
            ``feature_names``, or if ``top_k`` is not positive.
    """
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    scores = _tree_importances(estimator)
    if scores is None:
        scores = _linear_importances(estimator)
    if scores is None:
        return None

    if scores.shape[0] != len(feature_names):
        raise ValueError(
            f"estimator reports {scores.shape[0]} importances but "
            f"{len(feature_names)} feature names were provided"
        )

    pairs = [
        FeatureImportance(name, float(s))
        for name, s in zip(feature_names, scores, strict=True)
    ]
    # Deterministic ordering: descending score, ties broken by feature name.
    pairs.sort(key=lambda fi: (-fi.score, fi.name))
    if top_k is not None:
        pairs = pairs[:top_k]
    return pairs
