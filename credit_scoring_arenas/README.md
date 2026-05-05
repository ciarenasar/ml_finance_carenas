# credit_scoring_arenas

Repositorio del homework de credit scoring del curso Machine Learning Aplicado a las Finanzas.

## Descripcion

Este proyecto implementa un pipeline de credit scoring con carga y validacion de datos, feature engineering, transformacion WoE, entrenamiento de modelos y serializacion del mejor baseline.

## Reproduccion

```bash
git clone <url-del-repositorio> && cd credit_scoring_arenas
conda env create -f environment.yml && conda activate credit_scoring_arenas
jupyter lab
```

## Estructura

```text
credit_scoring_arenas/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── baseline_v1/
├── notebooks/
│   └── 01_credit_scoring.ipynb
├── reports/
│   └── figures/
└── src/
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── metrics.py
    └── models.py
```

## Flujo del proyecto

1. Copiar `bankloan.csv` en `data/raw/`.
2. Ejecutar `notebooks/01_credit_scoring.ipynb` de arriba hacia abajo.
3. Revisar los artefactos generados en `models/baseline_v1/`.

## Entregables esperados

- Notebook orquestador en `notebooks/01_credit_scoring.ipynb`.
- Modulos en `src/` con la logica de negocio.
- Modelo serializado y `metadata.json` generados automaticamente al guardar el mejor baseline.
