"""Utilidades del dashboard: carga de artefactos, etiquetas y predicción de un estudiante nuevo."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import BINARY, CLASSES, CONT_COLS, INDIC, NOMINAL, add_features  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"

# Etiquetas del diccionario de datos del conjunto UCI (id 697, Realinho et al., 2022)
COURSES = {33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
           8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
           9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture", 9147: "Management",
           9238: "Social Service", 9254: "Tourism", 9500: "Nursing", 9556: "Oral Hygiene",
           9670: "Advertising and Marketing Management", 9773: "Journalism and Communication",
           9853: "Basic Education", 9991: "Management (evening)"}
MARITAL = {1: "Soltero/a", 2: "Casado/a", 3: "Viudo/a", 4: "Divorciado/a", 5: "Unión de hecho", 6: "Separado/a legalmente"}
CLUSTER_NAMES = {0: "0 · Carga alta y créditos convalidados", 1: "1 · Trayectoria regular",
                 2: "2 · Sin rendimiento en el 1.er semestre"}
CLUSTER_COLORS = {CLUSTER_NAMES[0]: "#7B3294", CLUSTER_NAMES[1]: "#009E73", CLUSTER_NAMES[2]: "#D55E00"}
CLASS_COLORS = {"Dropout": "#D55E00", "Enrolled": "#E69F00", "Graduate": "#0072B2"}
CLASS_ES = {"Dropout": "Abandono (Dropout)", "Enrolled": "Matriculado (Enrolled)", "Graduate": "Graduado (Graduate)"}
MODELS = {"SVM (RBF) — campeón a priori": "svm", "Random Forest": "rf", "Regresión logística — referencia Fase 2": "lr"}


def course_label(c: int) -> str:
    return f"{c} · {COURSES.get(int(c), 'Programa')}"


@st.cache_data(show_spinner=False)
def load_students() -> pd.DataFrame:
    df = pd.read_csv(ART / "estudiantes.csv")
    df["Curso"] = df["Course"].map(course_label)
    df["Clúster"] = df["cluster"].map(CLUSTER_NAMES)
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((ART / "metricas.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ART / name)


@st.cache_resource(show_spinner="Cargando modelos…")
def load_models() -> dict:
    """Carga los modelos guardados; si el entorno no es compatible, los reentrena (≈ 15 s, determinista)."""
    try:
        return joblib.load(ART / "modelos.joblib")
    except Exception:  # versión distinta de scikit-learn/numpy u objeto incompatible
        from src import train
        train.main()
        return joblib.load(ART / "modelos.joblib")


def build_row(inp: dict) -> pd.DataFrame:
    """Convierte las entradas del formulario en una fila con las 36 variables predictoras (más derivadas)."""
    base = {c: inp[c] for c in CONT_COLS + NOMINAL + [b for b in BINARY if b not in INDIC]}
    return add_features(pd.DataFrame([base]))


def predict_one(inp: dict, models: dict, which: str) -> dict:
    row = build_row(inp)
    m = models[which]
    proba = dict(zip(m.classes_, m.predict_proba(row)[0]))
    z = models["cluster_pre"].transform(row)
    cl = int(models["kmeans"].predict(z)[0])
    pc = models["pca"].transform(z)[0]
    return {"proba": proba, "clase": str(m.predict(row)[0]), "nota2S": float(models["reg"].predict(row)[0]),
            "cluster": cl, "pc": pc, "row": row}
