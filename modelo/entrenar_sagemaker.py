import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Rutas entregadas por SageMaker o usadas para pruebas locales
CARPETA_DATOS = Path(
    os.environ.get("SM_CHANNEL_TRAIN", "datos/procesados")
)
CARPETA_TEST = Path(
    os.environ.get("SM_CHANNEL_TEST", str(CARPETA_DATOS))
)
CARPETA_MODELO = Path(
    os.environ.get("SM_MODEL_DIR", "modelo/sagemaker_salida")
)

# Leer los archivos previamente tratados
train = pd.read_csv(CARPETA_DATOS / "train.csv")
test = pd.read_csv(CARPETA_TEST / "test.csv")

numericas = [
    "monto_negativo",
    "monto_cero",
    "moneda_desconocida",
    "monto_ambiguo",
    "coordenadas_sospechosas",
    "dispositivo_id_faltante",
    "comercio_id_faltante",
    "ip_faltante",
]

categoricas = [
    "moneda",
    "canal",
    "mcc",
    "pais_transaccion",
]

variables = numericas + categoricas
objetivo = "es_fraude_limpio"

def preparar_variables(df):
    X = df[variables].copy()
    y = df[objetivo].astype(int)

    for columna in numericas:
        X[columna] = X[columna].astype(float)

    for columna in categoricas:
        X[columna] = X[columna].astype("string").astype("object")
        X[columna] = X[columna].where(
            X[columna].notna(), np.nan
        )

    return X, y

X_train, y_train = preparar_variables(train)
X_test, y_test = preparar_variables(test)

# Mismo preprocesamiento del modelo original
preprocesador = ColumnTransformer([
    ("numericas", SimpleImputer(strategy="median"), numericas),
    ("categoricas", Pipeline([
        ("imputador", SimpleImputer(
            strategy="constant",
            fill_value="desconocido"
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categoricas)
])

# Mismos hiperparámetros del Random Forest original
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

modelo.fit(X_train, y_train)

predicciones = modelo.predict(X_test)
probabilidades = modelo.predict_proba(X_test)[:, 1]

metricas = {
    "accuracy": float(accuracy_score(y_test, predicciones)),
    "precision": float(precision_score(y_test, predicciones)),
    "recall": float(recall_score(y_test, predicciones)),
    "f1": float(f1_score(y_test, predicciones)),
    "average_precision": float(
        average_precision_score(y_test, probabilidades)
    ),
}

print("\n=== ENTRENAMIENTO SAGEMAKER ===")
print("Entrenamiento:", len(X_train))
print("Prueba:", len(X_test))
print("Metricas:", json.dumps(metricas, indent=2))
print("Matriz de confusion:")
print(confusion_matrix(y_test, predicciones))

# SageMaker empaqueta el contenido de SM_MODEL_DIR
CARPETA_MODELO.mkdir(parents=True, exist_ok=True)

joblib.dump(modelo, CARPETA_MODELO / "model.joblib")

metadata = {
    "version": "ep3_sagemaker_v1",
    "algoritmo": "RandomForestClassifier",
    "variables": variables,
    "metricas": metricas,
}

with open(
    CARPETA_MODELO / "metadata.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(metadata, archivo, indent=2)

print("Modelo guardado en:", CARPETA_MODELO)