import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from botocore.exceptions import BotoCoreError, ClientError

from app.database import obtener_conexion, inicializar_base_datos
from modelo.cliente_sagemaker import predecir_sagemaker
from modelo.predecir import predecir_transaccion

VERSION_ENDPOINT = os.getenv(
    "KIPU_SAGEMAKER_ENDPOINT",
    "kipu-fraudshield-ep3-endpoint-v3"
)
MODO_PREDICCION = os.getenv("KIPU_MODELO_PREDICCION", "local").lower()

app = FastAPI(
    title="Kipu FraudShield - Servicio de Dominio",
    version="1.0.0",
    description="Servicio de evaluación de transacciones mediante modelo local o SageMaker."
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
        "endpoint_sagemaker": VERSION_ENDPOINT,
        "modo_prediccion": MODO_PREDICCION
    }


@app.post("/api/predict")
def predict(transaccion: Transaccion):
    try:
        datos = transaccion.model_dump()

        if MODO_PREDICCION == "local":
            resultado = predecir_transaccion(datos)
            prediccion = int(resultado["es_fraude_predicho"])
            probabilidad = float(resultado["probabilidad_fraude"])
            version_modelo = resultado["version_modelo"]
            estado_modelo = "local"

        elif MODO_PREDICCION == "sagemaker":
            resultado = predecir_sagemaker(datos)
            predicciones = resultado.get("predicciones", [])

            if not predicciones:
                raise ValueError("SageMaker no devolvió predicciones.")

            prediccion_modelo = predicciones[0]
            prediccion = int(prediccion_modelo["es_fraude_predicho"])
            probabilidad = float(
                prediccion_modelo["probabilidad_fraude"]
            )
            version_modelo = resultado.get(
                "version_modelo", VERSION_ENDPOINT
            )
            estado_modelo = "sagemaker"

        else:
            raise ValueError("Modo de predicción no válido.")

        if prediccion not in (0, 1) or not 0 <= probabilidad <= 1:
            raise ValueError("El modelo devolvió una predicción inválida.")

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

    except HTTPException:
        raise
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