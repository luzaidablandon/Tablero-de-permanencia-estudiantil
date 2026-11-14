# Permanencia estudiantil: alerta temprana al cierre del 1.er semestre

Proyecto Integrador — Maestría en Inteligencia Artificial (Fase 4: dashboard interactivo, informe final y sustentación).

Aplicación web en **Streamlit** que integra, sobre el conjunto UCI *Predict Students' Dropout and Academic Success* (4.424 estudiantes, 37 variables):

| Módulo | Pestaña | Contenido |
|---|---|---|
| Filtros dinámicos | Barra lateral | Partición, programa, género, becario, deudor, matrícula al día, régimen, desplazado, resultado real, clúster, y rangos de edad, nota de admisión, unidades aprobadas, tasa de aprobación y nota del 1.er semestre |
| Tarjetas KPI ejecutivas | Resumen ejecutivo | 6 indicadores dinámicos de la selección y 6 de desempeño de la solución |
| Predicción en vivo | Predicción en vivo | Formulario con casos de ejemplo; probabilidad de Dropout/Enrolled/Graduate (SVM, Random Forest o regresión logística), nota esperada del 2.º semestre, clúster asignado y escenarios «¿qué pasaría si…?» |
| Segmentación | Segmentación (clustering) | Dispersión PCA 2D/3D con K-Means (K = 3), estudiante simulado superpuesto, perfiles numéricos promedio, codo y silueta |
| Modelos y explicabilidad | Modelos y explicabilidad | Métricas, importancia por permutación, SHAP y dependencia parcial |

> **Limitación declarada.** El conjunto de datos no contiene fechas ni períodos académicos, por lo que no existe un filtro por rango de fechas; los cortes disponibles son la partición, el programa y las demás categorías/variables numéricas.
> **Uso responsable.** Herramienta académica de apoyo; no debe usarse para decidir sobre estudiantes reales sin validación institucional.

## Estructura

```
data/            Datos originales (data.csv) y README con fuente, cita y licencia
notebooks/       Fase 2 (preprocesamiento y línea base) y Fase 3 (modelado, clustering, interpretabilidad), ya ejecutados
src/             features.py (preprocesamiento sin fuga) y train.py (entrena y genera los artefactos del dashboard)
dashboard/       app.py (Streamlit), utils.py, artifacts/ (modelos y tablas) y assets/ (figuras SHAP/PDP)
reports/         Informe técnico final
tests/           Pruebas de humo (reproducen las métricas de la Fase 3)
requirements.txt Versiones exactas para ejecutar el dashboard
```

El cuaderno de la Fase 1 (EDA) se encuentra en el archivo de entrega de esa fase y puede añadirse a `notebooks/` con el nombre `01_Fase1_EDA.ipynb`.

## Ejecución inmediata

Requiere Python 3.12 o 3.13.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Los artefactos (`dashboard/artifacts/modelos.joblib`, etc.) ya están incluidos. Si el entorno no puede cargarlos (versión distinta de scikit-learn), la aplicación reentrena los modelos automáticamente en unos 15 s.

Para regenerar todo desde los datos (semilla 42, partición 70/15/15 estratificada):

```bash
python -m src.train
pytest -q
```

Para abrir los cuadernos: `pip install -r requirements-notebooks.txt` y `jupyter lab`.

## Despliegue

Streamlit Community Cloud: repositorio público → *Create app* → archivo principal `dashboard/app.py` → *Advanced settings* → Python 3.13.

## Resultados principales (partición de prueba, n = 664)

| Tarea | Modelo | Resultado |
|---|---|---|
| Clasificación (3 clases) | SVM RBF (campeón a priori) | macro-F1 0,7023; AUC-ROC macro 0,8873; recall de abandono 0,7103 |
| Clasificación | Random Forest | macro-F1 0,7273; AUC-ROC macro 0,8903 |
| Referencia (Fase 2) | Regresión logística | macro-F1 0,7042; AUC-ROC macro 0,8885 |
| Regresión (nota 2.º semestre) | Random Forest | R² 0,7477; RMSE 2,5692; MAE 1,3827 |
| Clustering | K-Means, K = 3 | silueta 0,340 (entrenamiento) |

## Datos y licencia

Realinho, V., Machado, J., Baptista, L., Martins, M. V. (2022). *Predicting student dropout and academic success*. Data, 7(11), 146. UCI Machine Learning Repository, id 697. Licencia de los datos: CC BY 4.0. Código: MIT.
