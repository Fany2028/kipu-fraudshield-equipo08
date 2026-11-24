import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from botocore.exceptions import BotoCoreError, ClientError

from app.database import obtener_conexion, inicializar_base_datos
from modelo.cliente_sagemaker import predecir_sagemaker

VERSION_ENDPOINT = os.getenv(
    "KIPU_SAGEMAKER_ENDPOINT",
    "kipu-fraudshield-ep3-endpoint-v3"
)

app = FastAPI(
    title="Kipu FraudShield - Servicio de Dominio",
    version="1.0.0",
    description="Servicio de evaluación de transacciones mediante SageMaker."
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
        "endpoint_sagemaker": VERSION_ENDPOINT
    }


@app.post("/api/predict")
def predict(transaccion: Transaccion):
    try:
        resultado = predecir_sagemaker(transaccion.model_dump())

        predicciones = resultado.get("predicciones", [])
        if not predicciones:
            raise ValueError("SageMaker no devolvió predicciones.")

        prediccion_modelo = predicciones[0]
        prediccion = int(prediccion_modelo["es_fraude_predicho"])
        probabilidad = float(prediccion_modelo["probabilidad_fraude"])
        version_modelo = resultado.get("version_modelo", VERSION_ENDPOINT)

        if prediccion not in (0, 1) or not 0 <= probabilidad <= 1:
            raise ValueError("SageMaker devolvió una predicción inválida.")

        estado_modelo = "sagemaker"

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
                        version_modelo,
                        estado_modelo
                    )
                )

        return {
            "prediccion": prediccion,
            "probabilidad_fraude": round(probabilidad, 4),
            "version_modelo": version_modelo,
            "estado_modelo": estado_modelo
        }

    except (BotoCoreError, ClientError):
        raise HTTPException(
            status_code=503,
            detail="El servicio de evaluación de SageMaker no está disponible."
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No fue posible evaluar la transacción."
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

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No fue posible consultar las evaluaciones."
        )


@app.get("/")
def inicio():
    return {
        "mensaje": "Kipu FraudShield activo",
        "documentacion": "/docs"
    }