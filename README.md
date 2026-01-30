# modelcard

Turn a trained scikit-learn estimator into a **deterministic Markdown model card** - the kind of documentation MLOps and model-governance reviews ask for, generated straight from the artifact instead of hand-written.

## Problem

Model cards (intended use, training data, metrics, limitations, feature importances) are widely recommended but rarely kept up to date, because writing them by hand is tedious and drifts from the actual model. `modelcard` derives the card from the estimator and an eval-metrics dict, so the documentation is regenerated in one command and is **byte-identical given identical inputs** - safe to commit and diff in CI.

## What it does

- Reads the estimator's class name and, for tree/linear models, its feature importances (`feature_importances_` or mean-absolute `coef_`).
- Renders a metrics table (sorted by name, floats to 4 dp for stable diffs).
- Emits sections for intended use, out-of-scope uses, training data, limitations, and ethical considerations from typed metadata objects.
- Ships a library **and** a CLI (`modelcard demo`, `modelcard render`).
- Fully offline: the demo uses scikit-learn's bundled Iris dataset - no network, no downloads, no API keys.

## Quickstart

From a fresh clone (requires [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run modelcard demo --out card.md   # writes a model card for a