"""Typed metadata containers describing the model and its training data."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata about the dataset a model was trained/evaluated on.

    Attributes:
        name: Human-readable dataset name.
        description: One or two sentences on what the data represents.
        n_samples: Number of training samples.
        n_features: Number of input features.
        feature_names: Names of the input features, in column order.
        target_names: Names of the target classes (classification) or ``None``
            for regression / when not applicable.
        source: Optional provenance string (URL, internal path, citation).
    """

    name: str
    description: str
    n_samples: int
    n_features: int
    feature_names: list[str]
    target_names: list[str] | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if self.n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        if self.n_features < 0:
            raise ValueError("n_features must be non-negative")
        if len(self.feature_names) != self.n_features:
            raise ValueError(
                f"feature_names has {len(self.feature_names)} entries "
                f"but n_features is {self.n_features}"
            )


@dataclass(frozen=True)
class ModelInfo:
    """Human-authored context that cannot be inferred from the estimator.

    Attributes:
        name: Model name, e.g. ``"Iris Species Classifier"``.
        version: Semantic or date-based version string.
        intended_use: What the model is meant to be used for.
        out_of_scope: Uses the model is explicitly *not* suitable for.
        limitations: Known limitations / caveats, one per bullet.
        ethical_considerations: Optional fairness / bias notes.
        authors: Optional list of author names.
        license: Optional license identifier, e.g. ``"MIT"``.
    """

    name: str
    version: str
    intended_use: str
    out_of_scope: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    ethical_considerations: str | None = None
    authors: list[str] = field(default_factory=list)
    license: str | None = None
