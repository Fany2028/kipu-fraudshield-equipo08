import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

VARIABLES = [
    "monto_negativo",
    "monto_cero",
    "moneda_desconocida",
    "monto_ambiguo",
    "coordenadas_sospechosas",
    "dispositivo_id_faltante",
    "comercio_id_faltante",
    "ip_faltante",
    "moneda",
    "canal",
    "mcc",
    "pais_transaccion",
]

NUMERICAS = VARIABLES[:8]
CATEGORICAS = VARIABLES[8:]

def model_fn(model_dir):
    carpeta = Path(model_dir)

    modelo = joblib.load(carpeta / "model.joblib")

    with open(
        carpeta / "metadata.json",
        encoding="utf-8"
    ) as archivo:
        metadata = json.load(archivo)

    return {
        "modelo": modelo,
        "version": metadata["version"],
    }

def input_fn(request_body, content_type):
    if content_type.split(";")[0] != "application/json":
        raise ValueError("Se requiere application/json")

    datos = json.loads(request_body)

    if isinstance(datos, dict):
        datos = [datos]

    if not isinstance(datos, list) or not datos:
        raise ValueError("Se requiere una transaccion o una lista")

    for registro in datos:
        if not isinstance(registro, dict):
            raise ValueError("Cada transaccion debe ser un objeto")

        faltantes = [
            variable for variable in VARIABLES
            if variable not in registro
        ]
        if faltantes:
            raise ValueError(
                "Variables faltantes: " + ", ".join(faltantes)
            )

    df = pd.DataFrame(datos)[VARIABLES].copy()

    for columna in NUMERICAS:
        df[columna] = pd.to_numeric(
            df[columna], errors="raise"
        )

    for columna in CATEGORICAS:
        df[columna] = (
            df[columna].astype("string").astype("object")
        )
        df[columna] = df[columna].where(
            df[columna].notna(), np.nan
        )

    return df

def predict_fn(datos, artefactos):
    modelo = artefactos["modelo"]

    probabilidades = modelo.predict_proba(datos)[:, 1]
    predicciones = modelo.predict(datos)

    resultados = []

    for prediccion, probabilidad in zip(
        predicciones, probabilidades
    ):
        resultados.append({
            "es_fraude_predicho": int(prediccion),
            "probabilidad_fraude": float(probabilidad),
        })

    return {
        "version_modelo": artefactos["version"],
        "predicciones": resultados,
    }

def output_fn(prediction, accept):
    if accept not in (
        "application/json", "*/*", None
    ):
        raise ValueError("Formato de respuesta no soportado")

    return json.dumps(prediction), "application/json"