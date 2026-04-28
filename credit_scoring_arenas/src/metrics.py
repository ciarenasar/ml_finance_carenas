"""Metricas y utilidades de scorecard para el proyecto de credit scoring."""

import numpy as np
import pandas as pd

FACTOR_PUNTAJE = 20.0


def _obtener_scores_modelo(model: object, X: np.ndarray) -> np.ndarray:
    """Obtiene scores continuos desde un clasificador ajustado."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype="float64")
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype="float64")
    raise ValueError("El modelo debe implementar predict_proba o decision_function.")


def auc_roc(model: object, X: np.ndarray, y: np.ndarray) -> float:
    """Retorna el AUC-ROC de un modelo ajustado."""
    from sklearn.metrics import roc_auc_score

    y_score = _obtener_scores_modelo(model, X)
    return float(roc_auc_score(y, y_score))


def costo_total(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    umbral: float,
    c_fn: float = 500,
    c_fp: float = 100,
) -> float:
    """Retorna el costo total asimetrico para un umbral de decision."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob, dtype="float64")
    y_pred = (y_prob_arr >= umbral).astype(int)

    falsos_negativos = (y_true_arr == 1) & (y_pred == 0)
    falsos_positivos = (y_true_arr == 0) & (y_pred == 1)

    costo = falsos_negativos.sum() * c_fn + falsos_positivos.sum() * c_fp
    return float(costo)


def build_scorecard(woe_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construye un scorecard interpretable a partir de tablas WoE."""
    filas: list[pd.DataFrame] = []

    for feature, table in woe_tables.items():
        scorecard_feature = table.copy()
        scorecard_feature["feature"] = feature
        # Menor WoE implica mayor riesgo relativo, por eso invertimos el signo.
        scorecard_feature["points"] = (-FACTOR_PUNTAJE * scorecard_feature["woe"]).round(2)
        scorecard_feature["riesgo_relativo"] = np.where(
            scorecard_feature["woe"] < 0,
            "alto",
            np.where(scorecard_feature["woe"] > 0, "bajo", "neutro"),
        )
        filas.append(
            scorecard_feature[
                [
                    "feature",
                    "bin",
                    "total",
                    "n_events",
                    "n_non_events",
                    "woe",
                    "iv_bin",
                    "points",
                    "riesgo_relativo",
                ]
            ]
        )

    if not filas:
        return pd.DataFrame(
            columns=[
                "feature",
                "bin",
                "total",
                "n_events",
                "n_non_events",
                "woe",
                "iv_bin",
                "points",
                "riesgo_relativo",
            ]
        )

    return pd.concat(filas, ignore_index=True)
