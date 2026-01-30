"""Tests for feature-importance extraction across estimator families."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from modelcard.importances import extract_importances


def _toy_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(0)
    x = rng.rand(60, 3)
    # Make feature 0 strongly predictive.
    y = (x[:, 0] > 0.5).astype(int)
    return x, y


def test_tree_importances_sorted_desc() -> None:
    x, y = _toy_data()
    tree = DecisionTreeClassifier(random_state=0).fit(x, y)
    result = extract_importances(tree, ["f0", "f1", "f2"])
    assert result is not None
    scores = [fi.score for fi in result]
    assert scores == sorted(scores, reverse=True)
    # Feature 0 drives the label, so it should rank first.
    assert result[0].name == "f0"


def test_linear_importances_multiclass_mean_abs() -> None:
    rng = np.random.RandomState(1)
    x = rng.rand(90, 3)
    y = rng.randint(0, 3, size=90)
    clf = LogisticRegression(max_iter=500).fit(x, y)
    result = extract_importances(clf, ["a", "b", "c"])
    assert result is not None
    assert len(result) == 3
    assert all(fi.score >= 0 for fi in result)


def test_unsupported_estimator_returns_none() -> None:
    x, y = _toy_data()
    dummy = DummyClassifier(strategy="most_frequent").fit(x, y)
    assert extract_importances(dummy, ["f0", "f1", "f2"]) is None


def test_top_k_truncates() -> None:
    x, y = _toy_data()
    tree = DecisionTreeClassifier(random_state=0).fit(x, y)
    result = extract_importances(tree, ["f0", "f1", "f2"], top_k=1)
    assert result is not None
    assert len(result) == 1


def test_top_k_must_be_positive() -> None:
    x, y = _toy_data()
    tree = DecisionTreeClassifier(random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="top_k must be a positive"):
        extract_importances(tree, ["f0", "f1", "f2"], top_k=0)


def test_length_mismatch_raises() -> None:
    x, y = _toy_data()
    tree = DecisionTreeClassifier(random_state=0).fit(x, y)
    with pytest.raises(ValueError, match="reports 3 importances"):
        extract_importances(tree, ["only_one"])


def test_ties_broken_by_name() -> None:
    class _Fake:
        feature_importances_ = np.array([0.5, 0.5, 0.0])

    result = extract_importances(_Fake(), ["zeta", "alpha", "gamma"])
    assert result is not None
    # Both 0.5 scores tie; alpha sorts before zeta.
    assert [fi.name for fi in result[:2]] == ["alpha", "zeta"]
