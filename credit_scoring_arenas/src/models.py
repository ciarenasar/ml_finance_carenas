"""Funciones para entrenar, evaluar y serializar modelos."""

import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

SEED = 42
CV_FOLDS = 5
N_JOBS = 1

MODELOS_CONFIG = {
    "Logistic Regression": (
        LogisticRegression(
            solver="liblinear",
            random_state=SEED,
            max_iter=1000,
        ),
        {
            "C": np.logspace(-3, 3, 7),
            "penalty": ["l1", "l2"],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 8, None],
            "min_samples_leaf": [1, 5, 10],
        },
    ),
    "XGBoost": (
        XGBClassifier(
            random_state=SEED,
            eval_metric="logloss",
        ),
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        },
    ),
}


def _to_serializable(value: object) -> object:
    """Convierte objetos con tipos de NumPy a valores serializables en JSON."""
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _predict_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    """Retorna scores tipo probabilidad desde un clasificador ajustado."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.asarray(scores, dtype="float64")
    raise ValueError("El modelo debe implementar predict_proba o decision_function.")


def train_all_models(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    """Entrena los modelos candidatos y retorna el mejor estimador por familia."""
    trained_models: dict[str, object] = {}

    for model_name, (estimator, param_grid) in MODELOS_CONFIG.items():
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=CV_FOLDS,
            n_jobs=N_JOBS,
            refit=True,
        )
        grid.fit(X, y)
        trained_models[model_name] = grid.best_estimator_

    return trained_models


def evaluate_models(
    models: dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Retorna una tabla comparativa con metricas de test por modelo."""
    rows: list[dict[str, float | str]] = []

    for model_name, model in models.items():
        y_score = _predict_scores(model, X)
        rows.append(
            {
                "Modelo": model_name,
                "AUC_ROC": float(roc_auc_score(y, y_score)),
            }
        )

    return pd.DataFrame(rows).sort_values("AUC_ROC", ascending=False).reset_index(drop=True)


def save_model(model: object, path: str, metadata: dict) -> Path:
    """Guarda en disco el modelo entrenado junto con su metadata."""
    if not isinstance(metadata, dict):
        raise TypeError("metadata debe ser un diccionario.")

    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = metadata.get("model_name", model.__class__.__name__)
    model_slug = str(model_name).lower().replace(" ", "_")
    model_path = output_dir / f"{model_slug}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata_to_save = dict(metadata)
    metadata_to_save["saved_at"] = date.today().isoformat()
    if hasattr(model, "get_params"):
        metadata_to_save.setdefault("hyperparameters", model.get_params())
    metadata_to_save = _to_serializable(metadata_to_save)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_to_save, f, indent=2, ensure_ascii=False)

    return model_path
