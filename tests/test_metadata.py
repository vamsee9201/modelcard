"""Tests for metadata containers and their validation."""

from __future__ import annotations

import pytest

from modelcard.metadata import DatasetInfo, ModelInfo


def test_dataset_info_valid() -> None:
    ds = DatasetInfo(
        name="toy",
        description="desc",
        n_samples=10,
        n_features=2,
        feature_names=["a", "b"],
    )
    assert ds.n_features == 2
    assert ds.target_names is None


def test_dataset_info_feature_name_mismatch() -> None:
    with pytest.raises(ValueError, match="feature_names has 1"):
        DatasetInfo(
            name="toy",
            description="desc",
            n_samples=10,
            n_features=2,
            feature_names=["a"],
        )


def test_dataset_info_negative_samples() -> None:
    with pytest.raises(ValueError, match="n_samples must be non-negative"):
        DatasetInfo(
            name="toy",
            description="desc",
            n_samples=-1,
            n_features=0,
            feature_names=[],
        )


def test_model_info_defaults() -> None:
    m = ModelInfo(name="M", version="1.0", intended_use="use")
    assert m.limitations == []
    assert m.out_of_scope == []
    assert m.authors == []
    assert m.license is None
