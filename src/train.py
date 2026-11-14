"""Reentrena los modelos campeones de la Fase 3 y genera los artefactos que consume el dashboard.

Uso (desde la raíz del repositorio):
    python -m src.train                       # entrena y guarda en dashboard/artifacts
    python -m src.train --reference RUTA.csv  # además compara contra predicciones_y_segmentos.csv de la Fase 3

Todo es determinista (semilla 42): la partición estratificada 70/15/15 coincide con las Fases 2 y 3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.features import (CLASSES, SEED, TARGET, TARGET_REG, USED, add_features, build_cluster_preprocessor,
                          build_preprocessor, load_data)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "artifacts"
K_STAR = 3


def split(df: pd.DataFrame):
    y = df[TARGET]
    X = df.drop(columns=TARGET)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def cls_metrics(model, F, y) -> dict:
    p = model.predict(F)
    P = model.predict_proba(F)
    order = list(model.classes_)
    return {
        "accuracy": accuracy_score(y, p), "balanced_acc": balanced_accuracy_score(y, p),
        "precision_macro": precision_score(y, p, average="macro"), "recall_macro": recall_score(y, p, average="macro"),
        "f1_macro": f1_score(y, p, average="macro"),
        "auc_macro": roc_auc_score(y, P, multi_class="ovr", average="macro", labels=order),
        "recall_dropout": recall_score(y, p, labels=["Dropout"], average="macro"),
    }


def main(reference: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data(ROOT / "data" / "data.csv")
    assert df.shape == (4424, 37) and df.isna().sum().sum() == 0
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(df)
    F_tr, F_va, F_te = (add_features(x) for x in (X_tr, X_va, X_te))
    assert not any("2nd sem" in c for c in USED)

    # ---------------- Clasificación ----------------
    lr = Pipeline([("pre", build_preprocessor()),
                   ("clf", LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced"))]).fit(F_tr, y_tr)
    rf = Pipeline([("pre", build_preprocessor()),
                   ("clf", RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=2, min_samples_split=2,
                                                  max_features="log2", max_samples=None, class_weight="balanced_subsample",
                                                  random_state=SEED, n_jobs=-1))]).fit(F_tr, y_tr)
    svm = Pipeline([("pre", build_preprocessor()),
                    ("clf", SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced", probability=True,
                                random_state=SEED))]).fit(F_tr, y_tr)
    # ---------------- Regresión ----------------
    reg = Pipeline([("pre", build_preprocessor()),
                    ("reg", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=5, max_features=0.5,
                                                  max_samples=None, random_state=SEED, n_jobs=-1))]).fit(F_tr, X_tr[TARGET_REG].values)
    # ---------------- Clustering ----------------
    cpre = build_cluster_preprocessor().fit(F_tr)
    Z = {k: cpre.transform(v) for k, v in {"tr": F_tr, "va": F_va, "te": F_te}.items()}
    km = KMeans(K_STAR, n_init=50, random_state=SEED).fit(Z["tr"])
    pca = PCA(n_components=3, random_state=SEED).fit(Z["tr"])

    # ---------------- Métricas ----------------
    met = {"clasificacion": {}, "regresion": {}, "clustering": {}}
    for name, m in [("Regresión logística (Fase 2)", lr), ("RandomForest", rf), ("SVM (RBF)", svm)]:
        met["clasificacion"][name] = {"validacion": cls_metrics(m, F_va, y_va), "prueba": cls_metrics(m, F_te, y_te)}
    for tag, F, X in [("validacion", F_va, X_va), ("prueba", F_te, X_te)]:
        yt, yp = X[TARGET_REG].values, reg.predict(F)
        met["regresion"][tag] = {"r2": r2_score(yt, yp), "rmse": mean_squared_error(yt, yp) ** 0.5, "mae": mean_absolute_error(yt, yp)}
    met["clustering"] = {"K": K_STAR, "silueta_entrenamiento": silhouette_score(Z["tr"], km.labels_),
                         "silueta_validacion": silhouette_score(Z["va"], km.predict(Z["va"])),
                         "silueta_prueba": silhouette_score(Z["te"], km.predict(Z["te"])),
                         "varianza_explicada_pca": pca.explained_variance_ratio_.tolist()}

    # ---------------- Tabla de estudiantes para el dashboard ----------------
    parts = []
    for tag, X, F, y, z in [("entrenamiento", X_tr, F_tr, y_tr, Z["tr"]), ("validación", X_va, F_va, y_va, Z["va"]),
                            ("prueba", X_te, F_te, y_te, Z["te"])]:
        t = F.copy()
        t.insert(0, "id_estudiante", F.index)
        t.insert(1, "particion", tag)
        t["Target"] = y.values
        for key, m in [("svm", svm), ("rf", rf), ("lr", lr)]:
            P = m.predict_proba(F)
            for i, c in enumerate(m.classes_):
                t[f"proba_{key}_{c}"] = P[:, i]
            t[f"pred_{key}"] = m.predict(F)
        t["nota2S_predicha"] = reg.predict(F)
        t["cluster"] = km.predict(z)
        pc = pca.transform(z)
        for i in range(3):
            t[f"PC{i + 1}"] = pc[:, i]
        parts.append(t)
    est = pd.concat(parts).sort_values("id_estudiante").reset_index(drop=True)
    est.to_csv(OUT / "estudiantes.csv", index=False, encoding="utf-8")

    joblib.dump({"lr": lr, "rf": rf, "svm": svm, "reg": reg, "cluster_pre": cpre, "kmeans": km, "pca": pca,
                 "classes": list(svm.classes_)}, OUT / "modelos.joblib", compress=3)
    json.dump(met, open(OUT / "metricas.json", "w"), indent=1, default=float)
    print(json.dumps(met, indent=1, default=float))

    # ---------------- Verificación contra la Fase 3 ----------------
    if reference:
        ref = pd.read_csv(reference).set_index("id_estudiante")
        ref_eval = ref[ref["target_predicho"].notna()].index
        e = est.set_index("id_estudiante").loc[ref.index]
        from sklearn.metrics import adjusted_rand_score
        print("ARI clúster vs Fase 3:", adjusted_rand_score(ref["cluster_kmeans"], e["cluster"]))
        print("Identidad exacta de etiquetas de clúster:", bool((ref["cluster_kmeans"] == e["cluster"]).all()))
        print("Acuerdo clase predicha SVM (validación+prueba):", (ref.loc[ref_eval, "target_predicho"] == e.loc[ref_eval, "pred_svm"]).mean())
        print("max|Δ proba Dropout SVM|:", np.abs(ref["proba_Dropout"] - e["proba_svm_Dropout"]).max())
        print("max|Δ nota 2S predicha|:", np.abs(ref["nota2S_predicha"] - e["nota2S_predicha"]).max())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=None)
    main(ap.parse_args().reference)
