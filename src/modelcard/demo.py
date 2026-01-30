"""A tiny, fully offline demo: train a model and produce its card.

Uses scikit-learn's bundled Iris dataset (shipped inside the ``scikit-learn``
wheel — no network access required) and a fixed random seed so the resulting
card is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .card import render_model_card
from .metadata import DatasetInfo, ModelInfo

RANDOM_STATE = 42


@dataclass
class DemoResult:
    """Container for the trained demo artifacts."""

    estimator: RandomForestClassifier
    metrics: dict[str, Any]
    dataset: DatasetInfo
    model: ModelInfo


def train_demo() -> DemoResult:
    """Train a small RandomForest on Iris and compute held-out metrics.

    Returns:
        A :class:`DemoResult` with the fitted estimator, metrics, and metadata.
    """
    data = load_iris()
    feature_names = [str(name) for name in data.feature_names]
    target_names = [str(name) for name in data.target_names]

    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )

    estimator = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=RANDOM_STATE,
    )
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "n_test_samples": len(y_test),
    }

    dataset = DatasetInfo(
        name="Iris (scikit-learn bundled copy)",
        description=(
            "Fisher's classic Iris dataset: sepal and petal measurements for "
            "three iris species. Bundled with scikit-learn and used here purely "
            "as an offline demonstration fixture."
        ),
        n_samples=int(data.data.shape[0]),
        n_features=int(data.data.shape[1]),
        feature_names=feature_names,
        target_names=target_names,
        source="sklearn.datasets.load_iris",
    )

    model = ModelInfo(
        name="Iris Species Classifier",
        version="0.1.0",
        intended_use=(
            "Educational demonstration of the modelcard library. Predicts iris "
            "species from four flower measurements."
        ),
        out_of_scope=[
            "Any production or real-world botanical decision-making.",
            "Species outside the three the model was trained on.",
        ],
        limitations=[
            "Trained on only 150 samples across three balanced classes.",
            "No validation on data collected outside the original 1936 study.",
        ],
        ethical_considerations=(
            "Low risk: the dataset contains no personal or sensitive data. "
            "Included only to exercise the card generator end-to-end."
        ),
        authors=["Data Science Team"],
        license="MIT",
    )

    return DemoResult(estimator=estimator, metrics=metrics, dataset=dataset, model=model)


def render_demo_card(top_k_features: int = 10) -> str:
    """Train the demo model and return its Markdown card."""
    result = train_demo()
    return render_model_card(
        result.estimator,
        result.metrics,
        result.dataset,
        result.model,
        top_k_features=top_k_features,
    )
