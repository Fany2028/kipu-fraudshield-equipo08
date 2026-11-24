from pathlib import Path
import runpy
import joblib
import numpy as np
import json

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RAIZ = Path(__file__).resolve().parent.parent

# Reutilizar la limpieza e integración de datos
resultado = runpy.run_path(
    str(RAIZ / "datos" / "preparar.py")
)

df = resultado["datos_integrados"].loc[
    resultado["datos_integrados"]["es_fraude_limpio"].notna()
].copy()

numericas = [
    "monto_negativo", "monto_cero", "moneda_desconocida",
    "monto_ambiguo", "coordenadas_sospechosas",
    "dispositivo_id_faltante", "comercio_id_faltante",
    "ip_faltante"
]

categoricas = ["moneda", "canal", "mcc", "pais_transaccion"]

X = df[numericas + categoricas].copy()
y = df["es_fraude_limpio"].astype(int)

for columna in numericas:
    X[columna] = X[columna].astype(float)

for columna in categoricas:
    X[columna] = X[columna].astype("string").astype("object")
    X[columna] = X[columna].where(X[columna].notna(), np.nan)

# Misma división por usuario utilizada en Colab
divisor = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)

train, test = next(
    divisor.split(X, y, groups=df["usuario_id"].astype("string"))
)

preprocesador = ColumnTransformer([
    ("numericas", SimpleImputer(strategy="median"), numericas),
    ("categoricas", Pipeline([
        ("imputador", SimpleImputer(
            strategy="constant", fill_value="desconocido"
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categoricas)
])

modelo = Pipeline([
    ("preprocesamiento", preprocesador),
    ("clasificador", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

modelo.fit(X.iloc[train], y.iloc[train])

pred = modelo.predict(X.iloc[test])
prob = modelo.predict_proba(X.iloc[test])[:, 1]
real = y.iloc[test]

print("\n=== RESULTADOS RANDOM FOREST ===")
print("Entrenamiento:", len(train))
print("Prueba:", len(test))
print("Accuracy:", round(accuracy_score(real, pred), 4))
print("Precision:", round(precision_score(real, pred), 4))
print("Recall:", round(recall_score(real, pred), 4))
print("F1-score:", round(f1_score(real, pred), 4))
print("Average Precision:", round(average_precision_score(real, prob), 4))

print("\n=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(real, pred))

ruta = RAIZ / "modelo" / "kipu_fraudshield_ep3_conservador.joblib"
joblib.dump(modelo, ruta)
print("\nModelo guardado:", ruta)

# Guardar metadata del modelo
metadata = {
    "nombre": "Kipu FraudShield",
    "version": "ep3_prototipo",
    "algoritmo": "RandomForestClassifier",
    "variables": numericas + categoricas,
    "numericas": numericas,
    "categoricas": categoricas,
    "estado": "experimental",
    "limitaciones": (
        "Modelo entrenado con una muestra etiquetada reducida. "
        "Pendiente validacion temporal completa e independiente."
    )
}

ruta_metadata = RAIZ / "modelo" / "kipu_fraudshield_ep3_metadata.json"

with open(ruta_metadata, "w", encoding="utf-8") as archivo:
    json.dump(metadata, archivo, indent=4, ensure_ascii=False)

print("Metadata guardada:", ruta_metadata)