"""Preprocesamiento y variables derivadas (idénticos a las Fases 2 y 3).

Las clases/funciones viven aquí (y no en un cuaderno) para que los modelos guardados con joblib
puedan cargarse desde el dashboard sin depender de ``__main__``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
CLASSES = ["Dropout", "Enrolled", "Graduate"]
TARGET = "Target"
TARGET_REG = "Curricular units 2nd sem (grade)"

CANON = [
    "Marital Status", "Application mode", "Application order", "Course", "Daytime/evening attendance",
    "Previous qualification", "Previous qualification (grade)", "Nacionality", "Mother's qualification",
    "Father's qualification", "Mother's occupation", "Father's occupation", "Admission grade", "Displaced",
    "Educational special needs", "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
    "Age at enrollment", "International", "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)", "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)", "Unemployment rate", "Inflation rate", "GDP", "Target",
]

CONT_COLS = [
    "Application order", "Age at enrollment", "Previous qualification (grade)", "Admission grade",
    "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP",
]
DERIV = ["tasa_aprobacion_1s", "tasa_exito_eval_1s", "prop_convalidadas_1s", "riesgo_financiero", "dif_nota_ingreso_previa"]
INDIC = ["sin_evaluaciones_1s", "sin_carga_1s"]
NOMINAL = ["Marital Status", "Application mode", "Course", "Previous qualification", "Mother's qualification",
           "Father's qualification", "Mother's occupation", "Father's occupation"]
BINARY = ["Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
          "Tuition fees up to date", "Gender", "Scholarship holder", "International"] + INDIC
# Variables con log1p (resultado de la regla de la Fase 2: asimetría > 1 tras winsorizar y reducción de |asimetría|)
NUM_LOG = ["Application order", "Age at enrollment", "Curricular units 1st sem (credited)",
           "Curricular units 1st sem (without evaluations)"]
NUM_BASE = CONT_COLS + DERIV
NUM_LIN = [c for c in NUM_BASE if c not in NUM_LOG]
USED = NUM_BASE + NOMINAL + BINARY  # 36 predictores, ninguno del 2.º semestre

CLUS_NUM = ["Age at enrollment", "Admission grade", "Previous qualification (grade)", "Application order",
            "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
            "Curricular units 1st sem (credited)", "tasa_aprobacion_1s"]
CLUS_BIN = ["Debtor", "Tuition fees up to date", "Scholarship holder", "Displaced", "Gender", "Daytime/evening attendance"]
CL_LOG = [c for c in CLUS_NUM if c in NUM_LOG]
CL_LIN = [c for c in CLUS_NUM if c not in CL_LOG]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta cada columna a los percentiles [lower, upper] calculados SOLO en ajuste (entrenamiento)."""

    def __init__(self, lower=1, upper=99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_ = np.nanpercentile(X, self.lower, axis=0)
        self.hi_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas e indicadores de la Fase 2 (solo información hasta el 1.er semestre)."""
    d = d.copy()
    en = d["Curricular units 1st sem (enrolled)"].astype(float)
    ev = d["Curricular units 1st sem (evaluations)"].astype(float)
    ap = d["Curricular units 1st sem (approved)"].astype(float)
    cr = d["Curricular units 1st sem (credited)"].astype(float)
    d["tasa_aprobacion_1s"] = np.where(en > 0, ap / en.where(en > 0, 1), 0.0)
    d["tasa_exito_eval_1s"] = np.where(ev > 0, ap / ev.where(ev > 0, 1), 0.0)
    d["prop_convalidadas_1s"] = np.where((cr + en) > 0, cr / (cr + en).where((cr + en) > 0, 1), 0.0)
    d["riesgo_financiero"] = d["Debtor"].astype(int) + (1 - d["Tuition fees up to date"].astype(int))
    d["dif_nota_ingreso_previa"] = d["Admission grade"] - d["Previous qualification (grade)"]
    d["sin_evaluaciones_1s"] = (ev == 0).astype(int)
    d["sin_carga_1s"] = (en == 0).astype(int)
    return d


def build_preprocessor() -> ColumnTransformer:
    lin = Pipeline([("win", Winsorizer(1, 99)), ("sc", StandardScaler())])
    log = Pipeline([("win", Winsorizer(1, 99)),
                    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("sc", StandardScaler())])
    ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)
    return ColumnTransformer([("num_lin", lin, NUM_LIN), ("num_log", log, NUM_LOG),
                              ("nom", ohe, NOMINAL), ("bin", "passthrough", BINARY)],
                             remainder="drop", verbose_feature_names_out=True)


def build_cluster_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("lin", Pipeline([("w", Winsorizer(1, 99)), ("s", StandardScaler())]), CL_LIN),
        ("log", Pipeline([("w", Winsorizer(1, 99)),
                          ("l", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                          ("s", StandardScaler())]), CL_LOG),
        ("bin", "passthrough", CLUS_BIN)])


def _norm(c: str) -> str:
    return re.sub(r"\s+", " ", str(c).replace("﻿", "")).strip().lower()


def load_data(path: str | Path) -> pd.DataFrame:
    """Lee el CSV del repositorio UCI (id 697), normaliza nombres (BOM/espacios) y valida la estructura."""
    df = pd.read_csv(path, sep=None, engine="python")
    mapa = {_norm(c): c for c in CANON}
    df.columns = [mapa.get(_norm(c), str(c).strip()) for c in df.columns]
    faltan = set(CANON) - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas: {sorted(faltan)}")
    df = df[CANON].copy()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    return df
