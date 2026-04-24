# credit_scoring_arenas

Repositorio base para el homework de credit scoring del curso Machine Learning Aplicado a las Finanzas.

## Descripcion

Este proyecto implementa un pipeline de credit scoring con feature engineering, transformacion WoE, entrenamiento de modelos y serializacion del mejor baseline.

## Reproduccion

```bash
git clone <url-del-repositorio>
cd credit_scoring_arenas
conda env create -f environment.yml
conda activate credit_scoring_arenas
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

