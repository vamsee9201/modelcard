"""Tests for the command-line interface."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from modelcard.cli import main


def test_demo_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["demo"])
    assert code == 0
    out = capsys.readouterr().out
    assert "# Model Card: Iris Species Classifier" in out


def test_demo_to_file(tmp_path: Path) -> None:
    out = tmp_path / "card.md"
    code = main(["demo", "--out", str(out)])
    assert code == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Model Card:")


def test_render_from_pickle_and_spec(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    x = rng.rand(60, 2)
    y = (x[:, 0] > 0.5).astype(int)
    tree = DecisionTreeClassifier(random_state=0).fit(x, y)

    est_path = tmp_path / "est.pkl"
    with est_path.open("wb") as fh:
        pickle.dump(tree, fh)

    spec = {
        "metrics": {"accuracy": 0.88},
        "dataset": {
            "name": "toy",
            "description": "toy data",
            "n_samples": 60,
            "n_features": 2,
            "feature_names": ["a", "b"],
            "target_names": ["neg", "pos"],
        },
        "model": {
            "name": "CLI Model",
            "version": "1.2.3",
            "intended_use": "cli test",
            "limitations": ["small"],
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    out = tmp_path / "card.md"
    code = main(
        ["render", "--estimator", str(est_path), "--spec", str(spec_path), "--out", str(out)]
    )
    assert code == 0
    card = out.read_text(encoding="utf-8")
    assert "# Model Card: CLI Model" in card
    assert "| accuracy | 0.8800 |" in card
    assert "Feature Importances" in card


def test_render_missing_spec_key_errors(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    tree = DecisionTreeClassifier(random_state=0).fit(rng.rand(20, 2), [0, 1] * 10)
    est_path = tmp_path / "est.pkl"
    with est_path.open("wb") as fh:
        pickle.dump(tree, fh)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({"metrics": {"a": 1}}), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        main(["render", "--estimator", str(est_path), "--spec", str(spec_path)])
    assert exc.value.code == 2  # argparse error exit code


def test_no_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        main([])
