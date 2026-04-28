"""Funciones de ingenieria de variables para transformaciones WoE e IV."""

import numpy as np
import pandas as pd

EPSILON = 1e-6


def _make_bins(series: pd.Series, bins: int) -> pd.Series:
    """Retorna bins por cuantiles para una serie numerica con fallback seguro."""
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series("missing", index=series.index, dtype="object")

    unique_values = non_null.nunique()
    effective_bins = max(1, min(bins, unique_values))
    if effective_bins == 1:
        return pd.Series("all", index=series.index, dtype="object")

    binned = pd.qcut(series, q=effective_bins, duplicates="drop")
    return binned.astype("object").where(series.notna(), "missing")


def _prepare_feature_groups(df: pd.DataFrame, feature: str, bins: int) -> pd.Series:
    """Retorna grupos listos para agregar una variable en el calculo de WoE."""
    series = df[feature]
    if pd.api.types.is_numeric_dtype(series):
        return _make_bins(series, bins)

    return series.astype("object").fillna("missing")


def _format_bin_label(value: object) -> str:
    """Retorna una etiqueta de texto estable para un bin o categoria."""
    if pd.isna(value):
        return "missing"
    return str(value)


def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """Calcula la tabla WoE y el IV total de una variable."""
    work_df = df[[feature, target]].copy()
    groups = _prepare_feature_groups(work_df, feature, bins)
    work_df["__bin"] = groups

    summary = (
        work_df.groupby("__bin", dropna=False)[target]
        .agg(total="count", n_events="sum")
        .reset_index()
        .rename(columns={"__bin": "bin"})
    )
    summary["n_non_events"] = summary["total"] - summary["n_events"]

    total_events = float(summary["n_events"].sum())
    total_non_events = float(summary["n_non_events"].sum())

    summary["event_rate"] = (summary["n_events"] + EPSILON) / (
        total_events + EPSILON * len(summary)
    )
    summary["non_event_rate"] = (summary["n_non_events"] + EPSILON) / (
        total_non_events + EPSILON * len(summary)
    )
    summary["woe"] = np.log(summary["non_event_rate"] / summary["event_rate"])
    summary["iv_bin"] = (
        summary["non_event_rate"] - summary["event_rate"]
    ) * summary["woe"]

    summary["bin"] = summary["bin"].map(_format_bin_label)
    summary["feature"] = feature

    ordered_columns = [
        "feature",
        "bin",
        "total",
        "n_events",
        "n_non_events",
        "event_rate",
        "non_event_rate",
        "woe",
        "iv_bin",
    ]
    woe_table = summary[ordered_columns].sort_values("woe").reset_index(drop=True)
    iv = float(woe_table["iv_bin"].sum())
    return woe_table, iv


def select_features_by_iv(
    df: pd.DataFrame,
    target: str,
    threshold: float = 0.1,
) -> list[str]:
    """Retorna los nombres de variables con IV sobre el umbral."""
    selected_features: list[str] = []
    for feature in df.columns:
        if feature == target:
            continue
        _, iv = compute_woe_iv(df, feature, target)
        if iv >= threshold:
            selected_features.append(feature)
    return selected_features


def build_woe_tables(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    bins: int = 10,
) -> dict[str, pd.DataFrame]:
    """Construye las tablas WoE para las variables seleccionadas del train."""
    tables: dict[str, pd.DataFrame] = {}
    for feature in features:
        table, _ = compute_woe_iv(df, feature, target, bins=bins)
        tables[feature] = table
    return tables


def _parse_interval_bounds(label: str) -> tuple[float, float] | None:
    """Retorna limites numericos desde una etiqueta tipo qcut cuando es posible."""
    if not label.startswith(("(", "[")) or "," not in label:
        return None

    cleaned = label.strip("()[]")
    left_text, right_text = [part.strip() for part in cleaned.split(",", maxsplit=1)]
    try:
        return float(left_text), float(right_text)
    except ValueError:
        return None


def _transform_numeric_feature(series: pd.Series, table: pd.DataFrame) -> pd.Series:
    """Retorna una variable numerica transformada con su tabla WoE aprendida."""
    transformed = pd.Series(np.nan, index=series.index, dtype="float64")
    missing_woe = table.loc[table["bin"] == "missing", "woe"]
    default_woe = float(missing_woe.iloc[0]) if not missing_woe.empty else 0.0

    for row in table.itertuples(index=False):
        bounds = _parse_interval_bounds(row.bin)
        if bounds is None:
            continue
        lower, upper = bounds
        mask = series.gt(lower) & series.le(upper)
        transformed.loc[mask] = float(row.woe)

    transformed.loc[series.isna()] = default_woe
    return transformed.fillna(default_woe)


def _transform_categorical_feature(series: pd.Series, table: pd.DataFrame) -> pd.Series:
    """Retorna una variable categorica transformada con su tabla WoE aprendida."""
    mapping = dict(zip(table["bin"], table["woe"]))
    default_woe = float(mapping.get("missing", 0.0))
    clean_series = series.astype("object").fillna("missing").map(_format_bin_label)
    return clean_series.map(mapping).fillna(default_woe).astype("float64")


def transform_woe(
    df: pd.DataFrame,
    woe_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Transforma un dataframe usando tablas WoE precomputadas en train."""
    transformed = pd.DataFrame(index=df.index)

    for feature, table in woe_tables.items():
        if feature not in df.columns:
            raise ValueError(f"Missing feature for WoE transform: {feature}")

        output_column = f"{feature}_woe"
        if pd.api.types.is_numeric_dtype(df[feature]):
            transformed[output_column] = _transform_numeric_feature(df[feature], table)
        else:
            transformed[output_column] = _transform_categorical_feature(df[feature], table)

    return transformed
