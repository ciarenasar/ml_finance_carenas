"""Feature engineering helpers for WoE and IV transformations."""

import pandas as pd


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """Compute the WoE table and IV value for one feature."""
    raise NotImplementedError("Implement WoE and IV computation.")


def select_features_by_iv(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.1,
) -> list[str]:
    """Return the feature names whose IV is above the threshold."""
    raise NotImplementedError("Implement IV-based feature selection.")


def build_woe_tables(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Build WoE tables for the selected training features."""
    raise NotImplementedError("Implement WoE table generation for train data.")


def transform_woe(
    df: pd.DataFrame,
    woe_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Transform a dataframe using precomputed training WoE tables."""
    raise NotImplementedError("Implement WoE transformation using train tables only.")
