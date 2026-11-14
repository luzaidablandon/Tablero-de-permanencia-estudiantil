"""Dashboard interactivo — Predicción de permanencia estudiantil (UCI id 697). Proyecto Integrador, Fase 4."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.utils import (CLASS_COLORS, CLASS_ES, CLUSTER_COLORS, CLUSTER_NAMES, COURSES, MARITAL, MODELS,
                             course_label, load_csv, load_metrics, load_models, load_students, predict_one)
from src.features import CLASSES

st.set_page_config(page_title="Permanencia estudiantil — Dashboard", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background:#F3F5F8; border:1px solid #E1E6EC; border-radius:10px; padding:10px 14px;}
div[data-testid="stMetricLabel"] p {font-size:0.82rem; color:#52606D;}
.small {font-size:0.85rem; color:#52606D;}
div[data-testid="stMetricValue"] {font-size:1.65rem;}
</style>""", unsafe_allow_html=True)

df_all = load_students()
met = load_metrics()
models = load_models()
HASH = {"entrenamiento": "Entrenamiento", "validación": "Validación", "prueba": "Prueba"}
RISK_CLUSTER = 2

# ------------------------------------------------------------------ Filtros (barra lateral)
FILTER_KEYS = ["f_part", "f_curso", "f_gen", "f_beca", "f_deu", "f_mat", "f_reg", "f_des", "f_tgt", "f_clu",
               "f_edad", "f_adm", "f_apr", "f_tasa", "f_nota"]
st.sidebar.title("Filtros dinámicos")
if st.sidebar.button("Restablecer filtros", width="stretch"):
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)
    st.rerun()

st.sidebar.markdown("**Categorías**")
parts = st.sidebar.multiselect("Partición", list(HASH), default=list(HASH), format_func=HASH.get, key="f_part",
                               help="Entrenamiento (70 %), validación (15 %) y prueba (15 %) de la Fase 2. Las métricas honestas del modelo se calculan en validación y prueba.")
cursos = st.sidebar.multiselect("Programa académico", sorted(df_all["Curso"].unique()), key="f_curso", placeholder="Todos")
gen = st.sidebar.multiselect("Género", ["Femenino", "Masculino"], key="f_gen", placeholder="Todos")
beca = st.sidebar.multiselect("Becario", ["Sí", "No"], key="f_beca", placeholder="Todos")
deu = st.sidebar.multiselect("Deudor", ["Sí", "No"], key="f_deu", placeholder="Todos")
mat = st.sidebar.multiselect("Matrícula al día", ["Sí", "No"], key="f_mat", placeholder="Todos")
reg = st.sidebar.multiselect("Régimen", ["Diurno", "Nocturno"], key="f_reg", placeholder="Todos")
des = st.sidebar.multiselect("Desplazado", ["Sí", "No"], key="f_des", placeholder="Todos")
tgt = st.sidebar.multiselect("Resultado real (Target)", CLASSES, key="f_tgt", placeholder="Todos", format_func=lambda c: CLASS_ES[c])
clu = st.sidebar.multiselect("Clúster", list(CLUSTER_NAMES.values()), key="f_clu", placeholder="Todos")

st.sidebar.markdown("**Variables numéricas**")
edad = st.sidebar.slider("Edad al ingreso", 17, 70, (17, 70), key="f_edad")
adm = st.sidebar.slider("Nota de admisión", 95.0, 190.0, (95.0, 190.0), key="f_adm")
apr = st.sidebar.slider("Unidades aprobadas 1.er sem.", 0, 26, (0, 26), key="f_apr")
tasa = st.sidebar.slider("Tasa de aprobación 1.er sem.", 0.0, 1.0, (0.0, 1.0), step=0.05, key="f_tasa")
nota = st.sidebar.slider("Nota media 1.er sem. (0–20)", 0.0, 20.0, (0.0, 20.0), step=0.5, key="f_nota")
st.sidebar.caption("El conjunto UCI no contiene fechas ni períodos académicos; por eso no hay filtro por rango de fechas. "
                   "La partición y el programa son los cortes disponibles.")


def yn(col, sel, yes="Sí", no="No", one_is_yes=True):
    if not sel or len(sel) == 2:
        return pd.Series(True, index=df_all.index)
    want = 1 if (sel[0] == yes) == one_is_yes else 0
    return df_all[col] == want


m = df_all["particion"].isin(parts)
if cursos: m &= df_all["Curso"].isin(cursos)
if gen: m &= df_all["Gender"].isin([1 if g == "Masculino" else 0 for g in gen])
m &= yn("Scholarship holder", beca) & yn("Debtor", deu) & yn("Tuition fees up to date", mat) & yn("Displaced", des)
if reg: m &= df_all["Daytime/evening attendance"].isin([1 if r == "Diurno" else 0 for r in reg])
if tgt: m &= df_all["Target"].isin(tgt)
if clu: m &= df_all["Clúster"].isin(clu)
m &= df_all["Age at enrollment"].between(*edad) & df_all["Admission grade"].between(*adm)
m &= df_all["Curricular units 1st sem (approved)"].between(*apr) & df_all["tasa_aprobacion_1s"].between(*tasa)
m &= df_all["Curricular units 1st sem (grade)"].between(*nota)
df = df_all[m].copy()
st.sidebar.markdown(f"**{len(df):,} de {len(df_all):,} estudiantes** en la selección".replace(",", "."))

# ------------------------------------------------------------------ Encabezado
st.title("🎓 Permanencia estudiantil: alerta temprana al cierre del 1.er semestre")
st.caption("Proyecto Integrador · Fase 4 · Conjunto UCI «Predict Students’ Dropout and Academic Success» (4.424 estudiantes). "
           "Herramienta académica de apoyo: no debe usarse para decidir sobre estudiantes reales sin validación institucional.")
tab_res, tab_pred, tab_seg, tab_mod = st.tabs(["📊 Resumen ejecutivo", "🔮 Predicción en vivo", "🧩 Segmentación (clustering)", "🧪 Modelos y explicabilidad"])


def pct(x):
    return f"{x * 100:,.1f} %".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt(x, d=2):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ------------------------------------------------------------------ 1. Resumen ejecutivo
with tab_res:
    if df.empty:
        st.warning("La selección no contiene estudiantes. Ajuste los filtros.")
    else:
        ev = df[df["particion"] != "entrenamiento"]
        st.subheader("Indicadores de la selección")
        c = st.columns(6)
        c[0].metric("Estudiantes", f"{len(df):,}".replace(",", "."))
        c[1].metric("Abandono real", pct((df["Target"] == "Dropout").mean()))
        c[2].metric("Graduación real", pct((df["Target"] == "Graduate").mean()))
        c[3].metric("Riesgo medio", pct(df["proba_svm_Dropout"].mean()),
                    help="Promedio de P(Dropout) del modelo SVM. En entrenamiento es optimista (los modelos ya vieron esos casos).")
        c[4].metric("Aprobación 1S", pct(df["tasa_aprobacion_1s"].mean()))
        c[5].metric("Clúster de riesgo", pct((df["cluster"] == RISK_CLUSTER).mean()), help="Clúster 2: sin rendimiento en el 1.er semestre.")
        st.subheader("Desempeño de la solución")
        d = st.columns(6)
        sv = met["clasificacion"]["SVM (RBF)"]["prueba"]
        d[0].metric("Macro-F1 (prueba)", fmt(sv["f1_macro"], 3), help="SVM, campeón fijado a priori. Referencia Fase 2 (regresión logística): 0,704. Métricas en el conjunto de prueba (n = 664).")
        d[1].metric("AUC-ROC (prueba)", fmt(sv["auc_macro"], 3))
        d[2].metric("Recall abandono", pct(sv["recall_dropout"]))
        d[3].metric("R² nota 2S", fmt(met["regresion"]["prueba"]["r2"], 3), help="Random Forest Regressor.")
        d[4].metric("MAE nota 2S", fmt(met["regresion"]["prueba"]["mae"], 2) + " pts")
        d[5].metric("Silueta K = 3", fmt(met["clustering"]["silueta_entrenamiento"], 3))
        if len(ev):
            acc = (ev["Target"] == ev["pred_svm"]).mean()
            mf1 = __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(ev["Target"], ev["pred_svm"], average="macro")
            st.info(f"En la selección actual (solo validación + prueba, n = {len(ev):,}): exactitud del SVM = **{pct(acc)}**, macro-F1 = **{fmt(mf1, 3)}**.".replace(",", "."))
        else:
            st.info("La selección no incluye validación ni prueba: las métricas del modelo se omiten porque en entrenamiento serían optimistas.")

        a, b = st.columns(2)
        with a:
            cnt = df["Target"].value_counts().reindex(CLASSES).fillna(0).reset_index()
            cnt.columns = ["Resultado", "n"]
            fig = px.pie(cnt, names="Resultado", values="n", hole=0.55, color="Resultado", color_discrete_map=CLASS_COLORS,
                         title="Resultado final de la selección")
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=False, height=340, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig, width="stretch")
        with b:
            g = df.groupby("Curso").agg(n=("Target", "size"), abandono=("Target", lambda s: (s == "Dropout").mean())).reset_index()
            g = g[g["n"] >= 20].sort_values("abandono")
            fig = px.bar(g, x="abandono", y="Curso", orientation="h", text=g["abandono"].map(lambda v: f"{v*100:.0f} %"),
                         title="Tasa de abandono por programa (n ≥ 20)", color_discrete_sequence=["#D55E00"], hover_data={"n": True})
            fig.update_traces(textposition="outside", textfont_size=12, cliponaxis=False)
            fig.update_layout(height=340, margin=dict(t=50, b=10, l=10, r=40), xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        a, b = st.columns(2)
        with a:
            bins = pd.cut(df["tasa_aprobacion_1s"], [-0.001, 0.25, 0.5, 0.75, 0.9999, 1.0], labels=["≤ 25 %", "25–50 %", "50–75 %", "75–99 %", "100 %"])
            t = pd.crosstab(bins, df["Target"], normalize="index").reindex(columns=CLASSES).fillna(0).reset_index()
            t = t.melt(id_vars="tasa_aprobacion_1s", var_name="Resultado", value_name="p")
            fig = px.bar(t, x="tasa_aprobacion_1s", y="p", color="Resultado", color_discrete_map=CLASS_COLORS, barmode="stack",
                         title="Resultado final según la tasa de aprobación del 1.er semestre", category_orders={"Resultado": CLASSES})
            fig.update_layout(height=340, margin=dict(t=50, b=10, l=10, r=10), yaxis_tickformat=".0%", xaxis_title="Tasa de aprobación 1S", yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        with b:
            t = pd.crosstab(df["riesgo_financiero"], df["Target"], normalize="index").reindex(columns=CLASSES).fillna(0).reset_index()
            t = t.melt(id_vars="riesgo_financiero", var_name="Resultado", value_name="p")
            t["riesgo_financiero"] = t["riesgo_financiero"].map({0: "0 · al día", 1: "1 · deuda o matrícula pendiente", 2: "2 · ambas"})
            fig = px.bar(t, x="riesgo_financiero", y="p", color="Resultado", color_discrete_map=CLASS_COLORS, barmode="stack",
                         title="Resultado final según el riesgo financiero", category_orders={"Resultado": CLASSES})
            fig.update_layout(height=340, margin=dict(t=50, b=10, l=10, r=10), yaxis_tickformat=".0%", xaxis_title="Riesgo financiero", yaxis_title=None)
            st.plotly_chart(fig, width="stretch")
        if len(ev):
            a, b = st.columns(2)
            with a:
                fig = px.histogram(ev, x="proba_svm_Dropout", color="Target", color_discrete_map=CLASS_COLORS, nbins=30, barmode="overlay",
                                   opacity=0.65, category_orders={"Target": CLASSES}, title="Riesgo predicho (P de abandono) según el resultado real")
                fig.update_layout(height=340, margin=dict(t=50, b=10, l=10, r=10), xaxis_title="P(Dropout) — SVM", yaxis_title="Estudiantes")
                st.plotly_chart(fig, width="stretch")
            with b:
                cm = pd.crosstab(ev["Target"], ev["pred_svm"]).reindex(index=CLASSES, columns=CLASSES).fillna(0)
                cmn = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
                fig = go.Figure(go.Heatmap(z=cmn.values, x=CLASSES, y=CLASSES, colorscale="Blues", zmin=0, zmax=1,
                                           text=[[f"{int(cm.values[i][j])}<br>{(cmn.values[i][j] if cmn.values[i][j]==cmn.values[i][j] else 0)*100:.0f} %" for j in range(3)] for i in range(3)],
                                           texttemplate="%{text}", showscale=False))
                fig.update_layout(title="Matriz de confusión del SVM (validación + prueba)", height=340, margin=dict(t=50, b=10, l=10, r=10),
                                  xaxis_title="Clase predicha", yaxis_title="Clase real", yaxis_autorange="reversed")
                st.plotly_chart(fig, width="stretch")
        with st.expander("Ver y descargar los datos filtrados"):
            show = ["id_estudiante", "particion", "Curso", "Age at enrollment", "Gender", "tasa_aprobacion_1s", "riesgo_financiero", "Target",
                    "pred_svm", "proba_svm_Dropout", "nota2S_predicha", "Clúster"]
            st.dataframe(df[show].rename(columns={"pred_svm": "Predicción SVM", "proba_svm_Dropout": "P(Dropout)", "nota2S_predicha": "Nota 2S predicha"}),
                         width="stretch", height=280)
            st.download_button("Descargar CSV de la selección", df[show].to_csv(index=False).encode("utf-8"), "seleccion.csv", "text/csv")

# ------------------------------------------------------------------ 2. Predicción en vivo
PRESETS = {
    "Perfil típico": dict(p_course=9500, p_age=20, p_gender="Femenino", p_beca=False, p_deu=False, p_mat=True, p_reg="Diurno", p_des=True,
                          p_adm=126.0, p_prev=133.0, p_enr=6, p_eva=8, p_apr=5, p_gr=12.3, p_cre=0, p_sin=0),
    "Riesgo medio": dict(p_course=9991, p_age=27, p_gender="Masculino", p_beca=False, p_deu=True, p_mat=True, p_reg="Diurno", p_des=False,
                         p_adm=118.0, p_prev=125.0, p_enr=6, p_eva=7, p_apr=4, p_gr=12.0, p_cre=0, p_sin=0),
    "Riesgo alto": dict(p_course=9991, p_age=27, p_gender="Masculino", p_beca=False, p_deu=True, p_mat=False, p_reg="Diurno", p_des=False,
                        p_adm=118.0, p_prev=125.0, p_enr=6, p_eva=6, p_apr=2, p_gr=11.0, p_cre=0, p_sin=0),
    "Riesgo bajo": dict(p_course=9500, p_age=19, p_gender="Femenino", p_beca=True, p_deu=False, p_mat=True, p_reg="Diurno", p_des=True,
                        p_adm=140.0, p_prev=140.0, p_enr=6, p_eva=7, p_apr=6, p_gr=14.0, p_cre=0, p_sin=0),
    "Sin actividad en 1S": dict(p_course=9147, p_age=24, p_gender="Masculino", p_beca=False, p_deu=False, p_mat=True, p_reg="Diurno", p_des=False,
                                p_adm=122.0, p_prev=128.0, p_enr=6, p_eva=0, p_apr=0, p_gr=0.0, p_cre=0, p_sin=0),
}


def set_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


for k, v in PRESETS["Perfil típico"].items():
    st.session_state.setdefault(k, v)

with tab_pred:
    st.subheader("Simulador de riesgo para un estudiante nuevo")
    st.caption("Ingrese los datos conocidos al cierre del 1.er semestre. El modelo calcula al instante la probabilidad de cada resultado final, "
               "la nota esperada del 2.º semestre y el segmento (clúster) al que pertenecería.")
    pb = st.columns(6)
    pb[0].markdown("**Casos de ejemplo:**")
    for i, nm in enumerate(PRESETS):
        pb[i + 1].button(nm, on_click=set_preset, args=(nm,), width="stretch", key=f"btn_{i}")
    model_name = st.radio("Modelo supervisado", list(MODELS), horizontal=True, key="p_model")
    which = MODELS[model_name]

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown("##### Perfil y situación del estudiante")
        c1, c2 = st.columns(2)
        course = c1.selectbox("Programa académico", sorted(COURSES), format_func=course_label, key="p_course")
        age = c2.slider("Edad al ingreso", 17, 70, key="p_age")
        gender = c1.radio("Género", ["Femenino", "Masculino"], horizontal=True, key="p_gender")
        reg_v = c2.radio("Régimen", ["Diurno", "Nocturno"], horizontal=True, key="p_reg")
        c1, c2, c3, c4 = st.columns(4)
        beca_v = c1.toggle("Becario", key="p_beca")
        deu_v = c2.toggle("Deudor", key="p_deu")
        mat_v = c3.toggle("Matrícula al día", key="p_mat")
        des_v = c4.toggle("Desplazado", key="p_des")
        c1, c2 = st.columns(2)
        adm_v = c1.slider("Nota de admisión (0–200)", 95.0, 190.0, step=0.5, key="p_adm")
        prev_v = c2.slider("Nota de la cualificación previa", 95.0, 190.0, step=0.5, key="p_prev")
        st.markdown("##### Rendimiento en el 1.er semestre")
        c1, c2, c3 = st.columns(3)
        enr = c1.number_input("Unidades matriculadas", 0, 26, key="p_enr")
        eva = c2.number_input("Evaluaciones presentadas", 0, 45, key="p_eva")
        apr_v = c3.number_input("Unidades aprobadas", 0, 26, key="p_apr")
        c1, c2, c3 = st.columns(3)
        gr = c1.slider("Nota media 1S (0–20)", 0.0, 20.0, step=0.1, key="p_gr")
        cre = c2.number_input("Unidades convalidadas", 0, 20, key="p_cre")
        sin = c3.number_input("Unidades sin evaluaciones", 0, 12, key="p_sin")
        with st.expander("Opciones avanzadas (valores por defecto: categoría más frecuente / mediana)"):
            c1, c2 = st.columns(2)
            amode = c1.selectbox("Modo de ingreso (código)", sorted(df_all["Application mode"].unique()), index=0, key="p_amode")
            aord = c2.number_input("Orden de aplicación (0–9)", 0, 9, 1, key="p_aord")
            mar = c1.selectbox("Estado civil", sorted(MARITAL), format_func=lambda k: f"{k} · {MARITAL[k]}", key="p_mar")
            pq = c2.selectbox("Cualificación previa (código)", sorted(df_all["Previous qualification"].unique()), key="p_pq", index=0)
            mq = c1.selectbox("Cualificación de la madre (código)", sorted(df_all["Mother's qualification"].unique()), key="p_mq", index=0)
            fq = c2.selectbox("Cualificación del padre (código)", sorted(df_all["Father's qualification"].unique()), key="p_fq", index=0)
            mo = c1.selectbox("Ocupación de la madre (código)", sorted(df_all["Mother's occupation"].unique()), key="p_mo", index=0)
            fo = c2.selectbox("Ocupación del padre (código)", sorted(df_all["Father's occupation"].unique()), key="p_fo", index=0)
            sne = c1.toggle("Necesidades educativas especiales", key="p_sne")
            intl = c2.toggle("Estudiante internacional", key="p_intl")
            u = c1.number_input("Tasa de desempleo (%)", 7.0, 17.0, 11.1, 0.1, key="p_u")
            inf = c2.number_input("Tasa de inflación (%)", -1.0, 4.0, 1.4, 0.1, key="p_inf")
            gdp = c1.number_input("PIB (variación %)", -5.0, 4.0, 0.32, 0.01, key="p_gdp")
    inp = {"Marital Status": mar, "Application mode": amode, "Application order": aord, "Course": course,
           "Daytime/evening attendance": 1 if reg_v == "Diurno" else 0, "Previous qualification": pq, "Previous qualification (grade)": prev_v,
           "Mother's qualification": mq, "Father's qualification": fq, "Mother's occupation": mo, "Father's occupation": fo,
           "Admission grade": adm_v, "Displaced": int(des_v), "Educational special needs": int(sne), "Debtor": int(deu_v),
           "Tuition fees up to date": int(mat_v), "Gender": 1 if gender == "Masculino" else 0, "Scholarship holder": int(beca_v),
           "Age at enrollment": age, "International": int(intl), "Curricular units 1st sem (credited)": cre,
           "Curricular units 1st sem (enrolled)": enr, "Curricular units 1st sem (evaluations)": eva,
           "Curricular units 1st sem (approved)": apr_v, "Curricular units 1st sem (grade)": gr,
           "Curricular units 1st sem (without evaluations)": sin, "Unemployment rate": u, "Inflation rate": inf, "GDP": gdp}
    res = predict_one(inp, models, which)
    st.session_state["live_point"] = res["pc"]
    st.session_state["live_cluster"] = res["cluster"]
    with right:
        st.markdown("##### Resultado del modelo")
        p = res["proba"]
        top = max(p, key=p.get)
        m1, m2, m3 = st.columns(3)
        m1.metric("Clase", CLASS_ES[res["clase"]].split(" (")[0], help="Resultado de predict() del modelo. Con la SVM puede diferir de la clase con mayor probabilidad calibrada en casos límite.")
        m2.metric("P(abandono)", pct(p["Dropout"]))
        m3.metric("Nota 2.º sem. esperada", f"{fmt(res['nota2S'], 1)} / 20", help="Random Forest Regressor. Promedia ceros y notas positivas: valores bajos indican riesgo de no aprobar unidades.")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=p["Dropout"] * 100, number={"suffix": " %"},
                                     title={"text": "Riesgo de abandono"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1F2933"},
                                            "steps": [{"range": [0, 30], "color": "#BFE3D0"}, {"range": [30, 60], "color": "#F6E3A8"}, {"range": [60, 100], "color": "#F2B8A2"}]}))
        fig.update_layout(height=230, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig, width="stretch")
        pr = pd.DataFrame({"Resultado": CLASSES, "p": [p[c] for c in CLASSES]})
        fig = px.bar(pr, x="p", y="Resultado", orientation="h", color="Resultado", color_discrete_map=CLASS_COLORS, text=pr["p"].map(lambda v: f"{v*100:.1f} %"))
        fig.update_layout(height=190, showlegend=False, margin=dict(t=10, b=10, l=10, r=10), xaxis_tickformat=".0%", xaxis_range=[0, 1], xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"**Segmento asignado:** {CLUSTER_NAMES[res['cluster']]}")
        if apr_v > enr + cre:
            st.warning("Las unidades aprobadas superan las matriculadas más las convalidadas; revise los datos ingresados.")
        if eva > 0 and enr == 0:
            st.warning("Hay evaluaciones pero ninguna unidad matriculada; revise los datos.")

        st.markdown("##### Escenarios «¿qué pasaría si…?»")
        scen = {"Situación ingresada": inp,
                "Aprueba todas las unidades matriculadas": {**inp, "Curricular units 1st sem (approved)": enr, "Curricular units 1st sem (evaluations)": max(eva, enr),
                                                            "Curricular units 1st sem (grade)": max(gr, 12.0) if enr else 0.0},
                "Regulariza deuda y matrícula": {**inp, "Debtor": 0, "Tuition fees up to date": 1},
                "Ambas": {**inp, "Curricular units 1st sem (approved)": enr, "Curricular units 1st sem (evaluations)": max(eva, enr),
                          "Curricular units 1st sem (grade)": max(gr, 12.0) if enr else 0.0, "Debtor": 0, "Tuition fees up to date": 1}}
        base_p = res["proba"]["Dropout"]
        rows = []
        for nm, x in scen.items():
            r_ = predict_one(x, models, which)
            rows.append({"Escenario": nm, "P(abandono)": r_["proba"]["Dropout"], "Δ": r_["proba"]["Dropout"] - base_p,
                         "Nota 2S": r_["nota2S"]})
        sc = pd.DataFrame(rows)
        st.dataframe(sc.style.format({"P(abandono)": "{:.1%}", "Δ": "{:+.1%}", "Nota 2S": "{:.1f}"}), hide_index=True, width="stretch")
        st.caption("Simulación del modelo (asociaciones aprendidas), no un efecto causal garantizado. Las probabilidades de la SVM provienen de calibración de Platt.")

# ------------------------------------------------------------------ 3. Segmentación
with tab_seg:
    st.subheader("Segmentación de estudiantes con K-Means (K = 3)")
    if df.empty:
        st.warning("La selección no contiene estudiantes.")
    else:
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        dim = c1.radio("Vista", ["2D (PC1–PC2)", "3D (PC1–PC3)"], horizontal=True)
        colorby = c2.selectbox("Colorear por", ["Clúster", "Resultado real", "Partición"])
        opac = c3.slider("Opacidad", 0.2, 1.0, 0.6, 0.05)
        show_live = c4.toggle("Mostrar estudiante simulado", value=True, help="El punto proviene de la pestaña «Predicción en vivo».")
        colmap = {"Clúster": ("Clúster", CLUSTER_COLORS), "Resultado real": ("Target", CLASS_COLORS), "Partición": ("particion", None)}
        col, cmap = colmap[colorby]
        plot = df.copy()
        if dim.startswith("2D"):
            fig = px.scatter(plot, x="PC1", y="PC2", color=col, color_discrete_map=cmap, opacity=opac, render_mode="webgl",
                             hover_data={"id_estudiante": True, "Curso": True, "tasa_aprobacion_1s": ":.2f", "Target": True, "PC1": False, "PC2": False})
            if show_live:
                pc = st.session_state["live_point"]
                fig.add_trace(go.Scattergl(x=[pc[0]], y=[pc[1]], mode="markers", name="Estudiante simulado",
                                         marker=dict(symbol="star", size=24, color="#FFD400", line=dict(width=2, color="black"))))
        else:
            fig = px.scatter_3d(plot, x="PC1", y="PC2", z="PC3", color=col, color_discrete_map=cmap, opacity=opac,
                                hover_data={"id_estudiante": True, "Curso": True, "Target": True})
            fig.update_traces(marker_size=3)
            if show_live:
                pc = st.session_state["live_point"]
                fig.add_trace(go.Scatter3d(x=[pc[0]], y=[pc[1]], z=[pc[2]], mode="markers", name="Estudiante simulado",
                                           marker=dict(symbol="diamond", size=9, color="#FFD400", line=dict(width=2, color="black"))))
        ve = met["clustering"]["varianza_explicada_pca"]
        fig.update_layout(height=560, margin=dict(t=30, b=10, l=10, r=10), legend_title_text="",
                          xaxis_title=f"PC1 ({ve[0]*100:.1f} % de la varianza)" if dim.startswith("2D") else None,
                          yaxis_title=f"PC2 ({ve[1]*100:.1f} %)" if dim.startswith("2D") else None)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Las dos primeras componentes explican {sum(ve[:2])*100:.1f} % de la varianza; las tres, {sum(ve)*100:.1f} %. "
                   "El PCA se ajustó solo con entrenamiento. Target y variables del 2.º semestre no participan en el clustering; se usan solo para perfilar.")

        st.markdown("##### Perfil numérico promedio por clúster (selección actual)")
        agg = df.groupby("Clúster").agg(
            Estudiantes=("id_estudiante", "size"),
            Edad=("Age at enrollment", "mean"),
            Matriculadas_1S=("Curricular units 1st sem (enrolled)", "mean"),
            Aprobadas_1S=("Curricular units 1st sem (approved)", "mean"),
            Nota_1S=("Curricular units 1st sem (grade)", "mean"),
            Tasa_aprobacion_1S=("tasa_aprobacion_1s", "mean"),
            Convalidadas_1S=("Curricular units 1st sem (credited)", "mean"),
            Becario=("Scholarship holder", "mean"), Deudor=("Debtor", "mean"), Matricula_al_dia=("Tuition fees up to date", "mean"),
            Abandono_real=("Target", lambda s: (s == "Dropout").mean()), Graduacion_real=("Target", lambda s: (s == "Graduate").mean()),
            Nota_2S_real=("Curricular units 2nd sem (grade)", "mean"),
        )
        agg.insert(1, "% de la selección", agg["Estudiantes"] / agg["Estudiantes"].sum())
        fm = {"Estudiantes": "{:,.0f}", "% de la selección": "{:.1%}", "Edad": "{:.1f}", "Matriculadas_1S": "{:.1f}", "Aprobadas_1S": "{:.1f}", "Nota_1S": "{:.1f}",
              "Tasa_aprobacion_1S": "{:.2f}", "Convalidadas_1S": "{:.1f}", "Becario": "{:.1%}", "Deudor": "{:.1%}", "Matricula_al_dia": "{:.1%}",
              "Abandono_real": "{:.1%}", "Graduacion_real": "{:.1%}", "Nota_2S_real": "{:.1f}"}
        lab = {"Estudiantes": "Estudiantes", "% de la selección": "% de la selección", "Edad": "Edad al ingreso (años)", "Matriculadas_1S": "Unidades matriculadas 1S",
               "Aprobadas_1S": "Unidades aprobadas 1S", "Nota_1S": "Nota media 1S", "Tasa_aprobacion_1S": "Tasa de aprobación 1S", "Convalidadas_1S": "Unidades convalidadas 1S",
               "Becario": "% becarios", "Deudor": "% deudores", "Matricula_al_dia": "% matrícula al día", "Abandono_real": "% abandono real (post hoc)",
               "Graduacion_real": "% graduación real (post hoc)", "Nota_2S_real": "Nota media 2S real (post hoc)"}
        prof = pd.DataFrame({lab[c]: [fm[c].format(v).replace(",", "X").replace(".", ",").replace("X", ".") for v in agg[c]] for c in agg.columns}, index=agg.index).T
        st.dataframe(prof, width="stretch", height=38 * (len(prof) + 1))
        st.caption("«Nota 2S real», «Abandono real» y «Graduación real» son validación externa posterior (no se usaron para formar los clústeres).")

        a, b = st.columns(2)
        ks = load_csv("kmeans_seleccion_K.csv")
        with a:
            fig = go.Figure(go.Scatter(x=ks["K"], y=ks["inercia"], mode="lines+markers", line_color="#0072B2"))
            fig.update_layout(title="Método del codo (inercia)", height=300, margin=dict(t=50, b=10), xaxis_title="K", yaxis_title="Inercia")
            fig.add_vline(x=6, line_dash="dash", line_color="#D55E00", annotation_text="codo (kneed): K = 6")
            st.plotly_chart(fig, width="stretch")
        with b:
            fig = go.Figure(go.Scatter(x=ks["K"], y=ks["silueta"], mode="lines+markers", line_color="#009E73"))
            fig.update_layout(title="Coeficiente de silueta", height=300, margin=dict(t=50, b=10), xaxis_title="K", yaxis_title="Silueta media")
            fig.add_vline(x=3, line_dash="dash", line_color="#D55E00", annotation_text="máximo: K = 3 (0,340)")
            st.plotly_chart(fig, width="stretch")
        st.markdown("""
**Lectura práctica.** El segmento 2 (sin rendimiento en el 1.er semestre) reúne a cerca del 19 % de los estudiantes y concentra
2,4–2,6 veces la tasa de abandono general en las tres particiones. El codo (K = 6) y la silueta (K = 3) no coinciden; se adoptó K = 3 por el
criterio fijado a priori. La estructura es débil pero estable (silueta 0,34; estabilidad *bootstrap* ARI 0,986). Un segundo algoritmo
(Ward) coincide con K-Means (ARI 0,86) y DBSCAN no resultó adecuado.""")

# ------------------------------------------------------------------ 4. Modelos
with tab_mod:
    st.subheader("Desempeño y explicabilidad de los modelos")
    rows = []
    for nm, v in met["clasificacion"].items():
        for part in ["validacion", "prueba"]:
            x = v[part]
            rows.append({"Modelo": nm, "Partición": part.capitalize().replace("Validacion", "Validación"), "Exactitud": x["accuracy"], "Exact. balanceada": x["balanced_acc"],
                         "Precisión macro": x["precision_macro"], "Recall macro": x["recall_macro"], "F1 macro": x["f1_macro"], "AUC-ROC macro": x["auc_macro"]})
    st.markdown("##### Clasificación (Dropout / Enrolled / Graduate)")
    st.dataframe(pd.DataFrame(rows).style.format({c: "{:.4f}" for c in ["Exactitud", "Exact. balanceada", "Precisión macro", "Recall macro", "F1 macro", "AUC-ROC macro"]}), hide_index=True, width="stretch")
    st.caption("Los tres modelos son estadísticamente equivalentes en validación (McNemar, p > 0,13) y sus intervalos de confianza en prueba se solapan. "
               "El campeón se fijó a priori con validación (SVM); Random Forest quedó nominalmente mejor en prueba.")
    st.markdown("##### Regresión (nota media del 2.º semestre) — Random Forest Regressor")
    st.dataframe(pd.DataFrame([{"Partición": k.capitalize().replace("Validacion", "Validación"), "R²": v["r2"], "RMSE": v["rmse"], "MAE": v["mae"]} for k, v in met["regresion"].items()])
                 .style.format({"R²": "{:.4f}", "RMSE": "{:.4f}", "MAE": "{:.4f}"}), hide_index=True)
    a, b = st.columns(2)
    for col, fn, title, lab in [(a, "importancia_permutacion_clasificacion.csv", "Importancia por permutación — clasificación (SVM)", "Caída de macro-F1"),
                                (b, "importancia_permutacion_regresion.csv", "Importancia por permutación — regresión (RF)", "Aumento de RMSE")]:
        d = load_csv(fn)
        d.columns = ["Variable", "Importancia", "DE"]
        d = d.sort_values("Importancia").tail(10)
        fig = px.bar(d, x="Importancia", y="Variable", orientation="h", error_x="DE", title=title, color_discrete_sequence=["#0072B2"])
        fig.update_layout(height=380, margin=dict(t=50, b=10, l=10, r=10), xaxis_title=lab, yaxis_title=None)
        col.plotly_chart(fig, width="stretch")
    a, b = st.columns([1.1, 1.4])
    a.markdown("##### SHAP — clase Dropout (Random Forest)")
    a.image(str(Path(__file__).parent / "assets" / "i3_shap_beeswarm_dropout.png"))
    b.markdown("##### Dependencia parcial de P(Dropout)")
    b.image(str(Path(__file__).parent / "assets" / "i4_pdp_dropout.png"))
    st.info("**Lectura:** el rendimiento del 1.er semestre (tasa de aprobación, tasa de éxito en evaluaciones, unidades aprobadas, nota) y el riesgo financiero "
            "dominan las predicciones. El riesgo cae con claridad solo cuando el estudiante aprueba la gran mayoría de lo matriculado (efecto umbral ≈ 75–90 %). "
            "Son asociaciones del modelo, no efectos causales.")
st.divider()
st.caption("Datos: Realinho, V., Machado, J., Baptista, L., Martins, M. V. (2022). Predicting student dropout and academic success. *Data*, 7(11), 146 — "
           "UCI Machine Learning Repository, id 697 (CC BY 4.0). Modelos entrenados con semilla 42 y partición 70/15/15 estratificada.")
