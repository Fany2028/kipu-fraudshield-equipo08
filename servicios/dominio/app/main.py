import os
import json

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from botocore.exceptions import BotoCoreError, ClientError
from secrets import compare_digest

from app.database import obtener_conexion, inicializar_base_datos
from modelo.cliente_sagemaker import predecir_sagemaker
from app.decisiones import obtener_decision
from app.cola import enviar_a_cola
from app.lista_negra import consultar_lista_negra

VERSION_ENDPOINT = os.getenv(
    "KIPU_SAGEMAKER_ENDPOINT",
    "kipu-fraudshield-ep3-endpoint-v3"
)

app = FastAPI(
    title="Kipu FraudShield - Servicio de Dominio",
    version="1.0.0",
    description="Servicio de evaluación de transacciones mediante SageMaker."
)

def validar_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    api_key_esperada = os.getenv("KIPU_API_KEY")

    if not api_key_esperada:
        raise HTTPException(
            status_code=503,
            detail="Autenticación no configurada."
        )

    if not x_api_key or not compare_digest(
        x_api_key, api_key_esperada
    ):
        raise HTTPException(
            status_code=401,
            detail="API Key inválida o ausente."
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
    ip: str | None = None
    dispositivo_id: str | None = None
    comercio_id: str | None = None
    tarjeta_enmascarada: str | None = None


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


@app.post("/api/predict", dependencies=[Depends(validar_api_key)])
def predict(transaccion: Transaccion):
    try:
        datos_modelo = transaccion.model_dump(
            include={
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
                "pais_transaccion"
            }
        )

        coincidencias_lista_negra = consultar_lista_negra(
            transaccion.model_dump()
        )
        if coincidencias_lista_negra:
            razones_regla = [
                f"Coincidencia en lista negra vigente: {tipo}"
                for tipo in coincidencias_lista_negra
            ]

            with obtener_conexion() as conexion:
                with conexion.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO evaluaciones (
                            prediccion,
                            probabilidad_fraude,
                            version_modelo,
                            estado_modelo,
                            estado_llm,
                            decision,
                            origen_decision,
                            razones
                        )
                        VALUES (NULL, NULL, %s, %s, %s, %s, %s, %s::jsonb)
                        RETURNING id
                        """,
                        (
                            "no_aplica_regla",
                            "omitido_por_regla",
                            "pendiente",
                            "bloquear",
                            "lista_negra",
                            json.dumps(razones_regla, ensure_ascii=False)
                        )
                    )

                    evaluacion_id = cursor.fetchone()[0]

            enviado = enviar_a_cola({
                "evaluacion_id": evaluacion_id,
                "decision": "bloquear",
                "senales": [
                    f"lista_negra_{tipo}"
                    for tipo in coincidencias_lista_negra
                ]
            })

            if not enviado:
                with obtener_conexion() as conexion:
                    with conexion.cursor() as cursor:
                        cursor.execute(
                            """
                            UPDATE evaluaciones
                            SET estado_llm = 'degradado',
                                error_llm = 'RabbitMQ no disponible'
                            WHERE id = %s
                            """,
                            (evaluacion_id,)
                        )

            return {
                "evaluacion_id": evaluacion_id,
                "prediccion": None,
                "probabilidad_fraude": None,
                "decision": "bloquear",
                "origen_decision": "lista_negra",
                "razones": razones_regla,
                "version_modelo": "no_aplica_regla",
                "estado_modelo": "omitido_por_regla",
                "estado_llm": "pendiente" if enviado else "degradado"
            }
        
        resultado = predecir_sagemaker(datos_modelo)

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
        decision = obtener_decision(probabilidad)
        estado_llm = "pendiente" if decision in ("revisar", "bloquear") else "no_solicitado"

        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO evaluaciones (
                        prediccion,
                        probabilidad_fraude,
                        version_modelo,
                        estado_modelo,
                        estado_llm
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        prediccion,
                        probabilidad,
                        version_modelo,
                        estado_modelo,
                        estado_llm
                    )
                )

                evaluacion_id = cursor.fetchone()[0]
                
        senales = [
            campo
            for campo, valor in transaccion.model_dump().items()
            if campo in (
                "monto_negativo",
                "monto_cero",
                "moneda_desconocida",
                "monto_ambiguo",
                "coordenadas_sospechosas",
                "dispositivo_id_faltante",
                "comercio_id_faltante",
                "ip_faltante"
            ) and valor == 1
        ]
        
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE evaluaciones
                    SET decision = %s,
                        origen_decision = 'sagemaker'
                    WHERE id = %s
                    """,
                    (decision, evaluacion_id)
                )

        if decision in ("revisar", "bloquear"):
            enviado = enviar_a_cola({
                "evaluacion_id": evaluacion_id,
                "decision": decision,
                "senales": senales
            })

            if not enviado:
                with obtener_conexion() as conexion:
                    with conexion.cursor() as cursor:
                        cursor.execute(
                            """
                            UPDATE evaluaciones
                            SET estado_llm = 'degradado',
                                error_llm = 'RabbitMQ no disponible'
                            WHERE id = %s
                            """,
                            (evaluacion_id,)
                        )

        return {
            "evaluacion_id": evaluacion_id,
            "prediccion": prediccion,
            "probabilidad_fraude": round(probabilidad, 4),
            "decision": decision,
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

@app.get("/api/predicciones/{evaluacion_id}", dependencies=[Depends(validar_api_key)])
def consultar_prediccion(evaluacion_id: int):
    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        id,
                        prediccion,
                        probabilidad_fraude,
                        version_modelo,
                        estado_modelo,
                        estado_llm,
                        resumen_llm,
                        error_llm,
                        fecha_evaluacion,
                        decision,
                        origen_decision,
                        razones
                    FROM evaluaciones
                    WHERE id = %s
                    """,
                    (evaluacion_id,)
                )
                fila = cursor.fetchone()

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No fue posible consultar la predicción."
        )

    if fila is None:
        raise HTTPException(
            status_code=404,
            detail="La evaluación no existe."
        )

    return {
        "evaluacion_id": fila[0],
        "prediccion": fila[1],
        "probabilidad_fraude": fila[2],
        "version_modelo": fila[3],
        "estado_modelo": fila[4],
        "estado_llm": fila[5],
        "resumen_llm": fila[6],
        "error_llm": fila[7],
        "fecha_evaluacion": fila[8].isoformat(),
        "decision": fila[9],
        "origen_decision": fila[10],
        "razones": fila[11]
    }

@app.get("/api/evaluaciones", dependencies=[Depends(validar_api_key)])
def listar_evaluaciones():
    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        id, prediccion, probabilidad_fraude,
                        version_modelo, estado_modelo, fecha_evaluacion,
                        decision, origen_decision, razones,
                        estado_llm, resumen_llm
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
                "fecha_evaluacion": fila[5].isoformat(),
                "decision": fila[6],
                "origen_decision": fila[7],
                "razones": fila[8],
                "estado_llm": fila[9],
                "resumen_llm": fila[10]
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