"""Metrics and scorecard helpers for the credit scoring project."""

import numpy as np
import pandas as pd


def auc_roc(model: object, X: np.ndarray, y: np.ndarray) -> float:
    """Return the AUC-ROC of a fitted model."""
    raise NotImplementedError("Implement AUC-ROC evaluation.")


def costo_total(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    umbral: float,
    c_fn: float = 500,
    c_fp: float = 100,
) -> float:
    """Return the asymmetric total cost for a probability threshold."""
    raise NotImplementedError("Implement vectorized asymmetric cost.")


def build_scorecard(woe_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an interpretable scorecard dataframe from WoE tables."""
    raise NotImplementedError("Implement scorecard construction.")
