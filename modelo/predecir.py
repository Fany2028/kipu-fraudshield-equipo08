from pathlib import Path
import json

import joblib
import pandas as pd


RUTA_MODELO = Path(__file__).resolve().parent
ARCHIVO_MODELO = RUTA_MODELO / "kipu_fraudshield_ep3_conservador.joblib"
ARCHIVO_METADATA = RUTA_MODELO / "kipu_fraudshield_ep3_metadata.json"


# Cargar el modelo y sus metadatos una sola vez
modelo = joblib.load(ARCHIVO_MODELO)

with open(ARCHIVO_METADATA, encoding="utf-8") as archivo:
    metadata = json.load(archivo)


def predecir_transaccion(transaccion: dict) -> dict:
    """
    Recibe las 12 variables esperadas por el modelo
    y devuelve la clasificación y probabilidad de fraude.
    """
    variables = metadata["variables"]

    faltantes = [campo for campo in variables if campo not in transaccion]
    if faltantes:
        raise ValueError(
            f"Faltan variables requeridas por el modelo: {faltantes}"
        )

    entrada = pd.DataFrame([transaccion])
    entrada = entrada[variables]

    prediccion = int(modelo.predict(entrada)[0])

    clases = list(modelo.classes_)
    if 1 not in clases:
        raise ValueError("El modelo no tiene una clase de fraude (1).")

    indice_fraude = clases.index(1)
    probabilidad = float(
        modelo.predict_proba(entrada)[0, indice_fraude]
    )

    return {
        "es_fraude_predicho": prediccion,
        "probabilidad_fraude": probabilidad,
        "version_modelo": metadata["version"]
    }
