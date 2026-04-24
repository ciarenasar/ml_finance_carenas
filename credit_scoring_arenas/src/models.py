"""Model training, evaluation and serialization helpers."""

from pathlib import Path

import pandas as pd

SEED = 42


def train_all_models(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Train the candidate models and return the best estimator per family."""
    raise NotImplementedError("Implement model training with GridSearchCV.")


def evaluate_models(
    models: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Return a comparison table with test metrics for each trained model."""
    raise NotImplementedError("Implement comparative model evaluation.")


def save_model(model: object, path: str, metadata: dict) -> Path:
    """Persist the trained model and its metadata to disk."""
    raise NotImplementedError("Implement model and metadata serialization.")
