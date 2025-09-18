"""modelcard: generate deterministic Markdown model cards for scikit-learn models."""

from __future__ import annotations

from .card import render_model_card
from .importances import FeatureImportance, extract_importances
from .metadata import DatasetInfo, ModelInfo

__version__ = "0.1.0"

__all__ = [
    "DatasetInfo",
    "FeatureImportance",
    "ModelInfo",
    "extract_importances",
    "render_model_card",
]
