"""Pruebas de humo: los artefactos cargan, reproducen las métricas de la Fase 3 y el simulador responde."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

ART = ROOT / "dashboard" / "artifacts"


def test_metricas_reproducen_fase3():
    m = json.loads((ART / "metricas.json").read_text(encoding="utf-8"))
    assert abs(m["clasificacion"]["SVM (RBF)"]["prueba"]["f1_macro"] - 0.7023) < 5e-4
    assert abs(m["clasificacion"]["Regresión logística (Fase 2)"]["validacion"]["f1_macro"] - 0.6648) < 5e-4
    assert abs(m["regresion"]["prueba"]["r2"] - 0.7477) < 5e-4
    assert abs(m["clustering"]["silueta_entrenamiento"] - 0.3404) < 5e-4


def test_tabla_estudiantes():
    e = pd.read_csv(ART / "estudiantes.csv")
    assert e.shape[0] == 4424 and set(e["particion"]) == {"entrenamiento", "validación", "prueba"}
    assert e["cluster"].isin([0, 1, 2]).all()


def test_prediccion_un_estudiante():
    from dashboard.utils import predict_one
    models = joblib.load(ART / "modelos.joblib")
    e = pd.read_csv(ART / "estudiantes.csv").iloc[0]
    inp = {k: e[k] for k in ["Marital Status", "Application mode", "Application order", "Course", "Daytime/evening attendance",
                             "Previous qualification", "Previous qualification (grade)", "Mother's qualification", "Father's qualification",
                             "Mother's occupation", "Father's occupation", "Admission grade", "Displaced", "Educational special needs", "Debtor",
                             "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment", "International",
                             "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
                             "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
                             "Unemployment rate", "Inflation rate", "GDP"]}
    r = predict_one(inp, models, "svm")
    assert abs(sum(r["proba"].values()) - 1) < 1e-9 and r["cluster"] in (0, 1, 2)
