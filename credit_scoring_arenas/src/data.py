"""Utilities for loading and preparing the raw credit dataset."""

import pandas as pd

REQUIRED_COLUMNS = [
    "Age",
    "Employ",
    "Address",
    "Income",
    "Creddebt",
    "OthDebt",
    "MonthlyLoad",
    "Default",
]


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    raise NotImplementedError("Implement CSV loading for bankloan data.")


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ValueError when required columns are missing."""
    raise NotImplementedError("Implement schema validation for the raw dataset.")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataset with derived features added."""
    raise NotImplementedError("Implement feature engineering without mutating the input.")
