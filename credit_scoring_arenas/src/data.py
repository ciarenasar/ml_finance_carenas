"""Utilidades para cargar y preparar el dataset crudo de credit scoring."""

import numpy as np
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
TARGET_COLUMN = "Default"
OPTIONAL_NUMERIC_COLUMNS = ["Leverage"]
OUTLIER_BOUNDS = {
    "Income": 300,
    "Creddebt": 15,
    "OthDebt": 30,
}


def _ensure_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Lanza ValueError cuando una columna esperada no es numerica."""
    invalid = [column for column in columns if not pd.api.types.is_numeric_dtype(df[column])]
    if invalid:
        invalid_str = ", ".join(invalid)
        raise ValueError(f"Expected numeric columns: {invalid_str}")


def _replace_sentinel_values(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna un dataframe con valores centinela reemplazados por NaN."""
    cleaned = df.copy()
    if "Leverage" in cleaned.columns:
        cleaned["Leverage"] = cleaned["Leverage"].replace(999999, np.nan)
    return cleaned


def _impute_numeric_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna un dataframe con missings numericos imputados por la mediana."""
    imputed = df.copy()
    numeric_columns = imputed.select_dtypes(include="number").columns
    imputed[numeric_columns] = imputed[numeric_columns].fillna(imputed[numeric_columns].median())
    return imputed


def _filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna un dataframe filtrado por los umbrales de outliers del homework."""
    filtered = df.copy()
    for column, upper_bound in OUTLIER_BOUNDS.items():
        if column in filtered.columns:
            filtered = filtered.loc[filtered[column] < upper_bound]
    return filtered


def load_raw(path: str) -> pd.DataFrame:
    """Carga el dataset crudo desde un archivo CSV."""
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Lanza ValueError cuando faltan columnas obligatorias o el schema es invalido."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")

    numeric_columns = REQUIRED_COLUMNS[:-1] + [
        column for column in OPTIONAL_NUMERIC_COLUMNS if column in df.columns
    ]
    _ensure_numeric_columns(df, numeric_columns)

    target_values = set(df[TARGET_COLUMN].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError("Default must be a binary column with values in {0, 1}.")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna una copia del dataset con variables derivadas y limpieza base."""
    validate_schema(df)

    features = _replace_sentinel_values(df)
    features = _impute_numeric_medians(features)
    features = _filter_outliers(features)

    income = features["Income"].replace(0, np.nan)
    features["OthDebtRatio"] = features["OthDebt"].div(income)
    features["OthDebtRatio"] = features["OthDebtRatio"].replace([np.inf, -np.inf], np.nan)
    features["OthDebtRatio"] = features["OthDebtRatio"].fillna(0.0)

    return features.reset_index(drop=True)
