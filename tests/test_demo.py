"""Tests for the offline demo: reproducibility and expected content."""

from __future__ import annotations

from modelcard.demo import render_demo_card, train_demo


def test_demo_metrics_are_reproducible() -> None:
    first = train_demo()
    second = train_demo()
    assert first.metrics == second.metrics


def test_demo_accuracy_matches_documented_value() -> None:
    # This number is quoted in the README; keep them in lockstep.
    result = train_demo()
    assert round(result.metrics["accuracy"], 4) == 0.9211
    assert result.metrics["n_test_samples"] == 38


def test_demo_card_is_byte_reproducible() -> None:
    assert render_demo_card() == render_demo_card()


def test_demo_card_lists_petal_length_as_top_feature() -> None:
    card = render_demo_card()
    section = card.split("## Top Feature Importances")[1]
    first_feature_row = next(
        ln for ln in section.splitlines() if ln.startswith("| petal") or ln.startswith("| sepal")
    )
    assert "petal length (cm)" in first_feature_row
