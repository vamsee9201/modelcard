"""Tests for the Markdown card renderer."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

from modelcard.card import render_model_card
from modelcard.metadata import DatasetInfo, ModelInfo


def _dataset(n_features: int = 3) -> DatasetInfo:
    names = [f"f{i}" for i in range(n_features)]
    return DatasetInfo(
        name="toy",
        description="A toy dataset.",
        n_samples=60,
        n_features=n_features,
        feature_names=names,
        target_names=["neg", "pos"],
        source="unit-test",
    )


def _model() -> ModelInfo:
    return ModelInfo(
        name="Toy Model",
        version="9.9",
        intended_use="Testing.",
        out_of_scope=["Real decisions."],
        limitations=["Tiny data."],
        ethical_considerations="None.",
        authors=["Data Science Team"],
        license="MIT",
    )


def _fitted_tree() -> DecisionTreeClassifier:
    rng = np.random.RandomState(0)
    x = rng.rand(60, 3)
    y = (x[:, 0] > 0.5).astype(int)
    return DecisionTreeClassifier(random_state=0).fit(x, y)


def test_card_contains_core_sections() -> None:
    card = render_model_card(
        _fitted_tree(), {"accuracy": 0.95}, _dataset(), _model()
    )
    for heading in (
        "# Model Card: Toy Model",
        "## Intended Use",
        "## Training Data",
        "## Evaluation Metrics",
        "## Top Feature Importances",
        "## Limitations",
        "## Ethical Considerations",
    ):
        assert heading in card
    assert "`DecisionTreeClassifier`" in card


def test_metrics_rendered_sorted_and_rounded() -> None:
    card = render_model_card(
        _fitted_tree(),
        {"zeta": 0.123456, "alpha": 1},
        _dataset(),
        _model(),
    )
    lines = card.splitlines()
    alpha_idx = next(i for i, ln in enumerate(lines) if ln.startswith("| alpha "))
    zeta_idx = next(i for i, ln in enumerate(lines) if ln.startswith("| zeta "))
    assert alpha_idx < zeta_idx  # sorted by metric name
    assert "| alpha | 1 |" in card  # int stays int
    assert "| zeta | 0.1235 |" in card  # float rounded to 4dp


def test_card_is_deterministic() -> None:
    args = (_fitted_tree(), {"accuracy": 0.95}, _dataset(), _model())
    assert render_model_card(*args) == render_model_card(*args)


def test_card_ends_with_newline() -> None:
    card = render_model_card(_fitted_tree(), {"accuracy": 0.95}, _dataset(), _model())
    assert card.endswith("\n")


def test_no_importance_section_for_unsupported_model() -> None:
    rng = np.random.RandomState(0)
    x = rng.rand(60, 3)
    y = (x[:, 0] > 0.5).astype(int)
    dummy = DummyClassifier(strategy="most_frequent").fit(x, y)
    card = render_model_card(dummy, {"accuracy": 0.5}, _dataset(), _model())
    assert "Feature Importances" not in card


def test_empty_metrics_raises() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        render_model_card(_fitted_tree(), {}, _dataset(), _model())


def test_optional_sections_omitted_when_absent() -> None:
    minimal = ModelInfo(name="Bare", version="0", intended_use="x")
    card = render_model_card(_fitted_tree(), {"accuracy": 0.9}, _dataset(), minimal)
    assert "## Ethical Considerations" not in card
    assert "**Authors:**" not in card
    assert "**License:**" not in card
    # Empty bullet lists fall back to a placeholder.
    assert "- None documented." in card


def test_top_k_limits_importance_rows() -> None:
    card = render_model_card(
        _fitted_tree(), {"accuracy": 0.95}, _dataset(), _model(), top_k_features=1
    )
    assert "## Top Feature Importances (up to 1)" in card
    # Exactly one data row under the importance header.
    section = card.split("## Top Feature Importances")[1].split("## Limitations")[0]
    data_rows = [ln for ln in section.splitlines() if ln.startswith("| f")]
    assert len(data_rows) == 1
