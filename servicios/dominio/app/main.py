from pathlib import Path
import json
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.database import obtener_conexion, inicializar_base_datos

MODEL_DIR = (
    Path(os.environ["MODEL_DIR"])
    if os.getenv("MODEL_DIR")
    else Path(__file__).resolve().parents[3] / "modelo"
)

MODEL_PATH = MODEL_DIR / "kipu_fraudshield_ep3_conservador.joblib"
METADATA_PATH = MODEL_DIR / "kipu_fraudshield_ep3_metadata.json"

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise RuntimeError("No se encontraron el modelo y sus metadatos.")

modelo = joblib.load(MODEL_PATH)

with open(METADATA_PATH, encoding="utf-8") as archivo:
    metadata = json.load(archivo)

app = FastAPI(
    title="Kipu FraudShield - Servicio de Dominio",
    version="1.0.0",
    description="Servicio local para evaluar transacciones."
)


class Transaccion(BaseModel):
    monto_negativo: int = Field(ge=0, le=1)
    monto_cero: int = Field(ge=0, le=1)
    moneda_desconocida: int = Field(ge=0, le=1)
    monto_ambiguo: int = Field(ge=0, le=1)
    coordenadas_sospechosas: int = Field(ge=0, le=1)
    dispositivo_id_faltante: int = Field(ge=0, le=1)
    comercio_id_faltante: int = Field(ge=0, le=1)
    ip_faltante: int = Field(ge=0, le=1)
    moneda: str
    canal: str
    mcc: str
    pais_transaccion: str


@app.on_event("startup")
def iniciar_servicio():
    inicializar_base_datos()


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "servicio": "dominio",
        "modelo": metadata["nombre"],
        "version_modelo": metadata["version"]
    }


@app.post("/api/predict")
def predict(transaccion: Transaccion):
    try:
        entrada = pd.DataFrame([transaccion.model_dump()])
        entrada = entrada[metadata["variables"]]

        probabilidades = modelo.predict_proba(entrada)[0]
        clases = list(modelo.classes_)

        if 1 not in clases:
            raise RuntimeError("El modelo no contiene la clase de fraude (1).")

        indice_fraude = clases.index(1)
        probabilidad = float(probabilidades[indice_fraude])
        prediccion = int(modelo.predict(entrada)[0])

        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO evaluaciones (
                        prediccion,
                        probabilidad_fraude,
                        version_modelo,
                        estado_modelo
                    )
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        prediccion,
                        probabilidad,
                        metadata["version"],
                        metadata["estado"]
                    )
                )

        return {
            "prediccion": prediccion,
            "probabilidad_fraude": round(probabilidad, 4),
            "version_modelo": metadata["version"],
            "estado_modelo": metadata["estado"]
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al evaluar la transaccion: {str(error)}"
        )


@app.get("/api/evaluaciones")
def listar_evaluaciones():
    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute("""
                    SELECT id, prediccion, probabilidad_fraude,
                           version_modelo, estado_modelo, fecha_evaluacion
                    FROM evaluaciones
                    ORDER BY fecha_evaluacion DESC
                    LIMIT 100
                """)
                filas = cursor.fetchall()

        return [
            {
                "id": fila[0],
                "prediccion": fila[1],
                "probabilidad_fraude": fila[2],
                "version_modelo": fila[3],
                "estado_modelo": fila[4],
                "fecha_evaluacion": fila[5].isoformat()
            }
            for fila in filas
        ]

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al consultar evaluaciones: {str(error)}"
        )


@app.get("/")
def inicio():
    return {
        "mensaje": "Kipu FraudShield activo",
        "documentacion": "/docs"
    }
