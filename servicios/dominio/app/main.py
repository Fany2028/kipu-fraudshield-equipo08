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
    monto: float | None = Field(default=None, ge=0, allow_inf_nan=False)
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
                            razones,
                            monto
                        )
                        VALUES (NULL, NULL, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        RETURNING id
                        """,
                        (
                            "no_aplica_regla",
                            "omitido_por_regla",
                            "pendiente",
                            "bloquear",
                            "lista_negra",
                            json.dumps(razones_regla, ensure_ascii=False),
                            transaccion.monto
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
        
        campos = [
            "monto_negativo", "monto_cero",
            "moneda_desconocida", "monto_ambiguo",
            "coordenadas_sospechosas",
            "dispositivo_id_faltante",
            "comercio_id_faltante", "ip_faltante"
        ]
        variantes = [datos_modelo]
        for campo in campos:
            alternativa = datos_modelo.copy()
            alternativa[campo] = 1 - alternativa[campo]
            variantes.append(alternativa)

        resultado = predecir_sagemaker(variantes)

        predicciones = resultado.get("predicciones", [])
        if not predicciones:
            raise ValueError("SageMaker no devolvió predicciones.")

        prediccion_modelo = predicciones[0]
        prediccion = int(prediccion_modelo["es_fraude_predicho"])
        probabilidad = float(prediccion_modelo["probabilidad_fraude"])
        version_modelo = resultado.get("version_modelo", VERSION_ENDPOINT)

        if prediccion not in (0, 1) or not 0 <= probabilidad <= 1:
            raise ValueError("SageMaker devolvió una predicción inválida.")

        if len(predicciones) != len(variantes):
            raise ValueError("Cantidad de predicciones inesperada.")

        sensibilidad = []
        for indice, campo in enumerate(campos, start=1):
            alternativa = float(
                predicciones[indice]["probabilidad_fraude"]
            )
            if not 0 <= alternativa <= 1:
                raise ValueError("Probabilidad alternativa invalida.")

            diferencia = (alternativa - probabilidad) * 100
            etiqueta = campo.replace("_", " ")
            razon = (
                f"{etiqueta}: cambiar de {datos_modelo[campo]} "
                f"a {1 - datos_modelo[campo]} modifica "
                f"la probabilidad de {probabilidad * 100:.2f}% "
                f"a {alternativa * 100:.2f}% "
                f"({diferencia:+.2f} puntos porcentuales)."
            )
            sensibilidad.append((abs(diferencia), razon))

        sensibilidad.sort(key=lambda x: x[0], reverse=True)
        razones_modelo = [
            razon for magnitud, razon in sensibilidad
            if magnitud >= 0.01
        ][:3]

        if not razones_modelo:
            razones_modelo = [
                "Las variaciones binarias no produjeron "
                "cambios relevantes en la probabilidad."
            ]

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
                        estado_llm,
                        razones,
                        monto
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    RETURNING id
                    """,
                    (
                        prediccion,
                        probabilidad,
                        version_modelo,
                        estado_modelo,
                        estado_llm,
                        json.dumps(razones_modelo, ensure_ascii=False),
                        transaccion.monto
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
                "senales": senales,
                "probabilidad_fraude": probabilidad,
                "razones_modelo": razones_modelo
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
            "razones": razones_modelo,
            "version_modelo": version_modelo,
            "estado_modelo": estado_modelo
        }

    except (BotoCoreError, ClientError) as error:
        print(
            f"[ERROR SAGEMAKER] {type(error).__name__}: {error}",
            flush=True
        )
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



@app.get("/api/health")
def health_detallado():
    import time
    import boto3
    import pika
    from botocore.config import Config

    servicios = {
        "postgresql": "no_disponible",
        "rabbitmq": "no_disponible",
        "sagemaker": "no_verificado",
        "llm": "no_verificado"
    }

    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        servicios["postgresql"] = "ok"
    except Exception:
        pass

    conexion_cola = None
    try:
        credenciales = pika.PlainCredentials(
            os.environ["RABBITMQ_USER"],
            os.environ["RABBITMQ_PASSWORD"]
        )
        parametros = pika.ConnectionParameters(
            host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
            credentials=credenciales,
            socket_timeout=2,
            connection_attempts=1
        )
        conexion_cola = pika.BlockingConnection(parametros)
        canal = conexion_cola.channel()
        estado_cola = canal.queue_declare(
            queue="resumenes_fraude",
            passive=True
        )
        servicios["rabbitmq"] = "ok"
        consumidores = estado_cola.method.consumer_count
        servicios["llm"] = (
            "worker_conectado_proveedor_no_verificado"
            if consumidores > 0 else "worker_no_conectado"
        )
    except Exception:
        pass
    finally:
        if conexion_cola is not None and conexion_cola.is_open:
            conexion_cola.close()

    try:
        cliente = boto3.client(
            "sagemaker",
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            config=Config(
                connect_timeout=2,
                read_timeout=3,
                retries={"max_attempts": 0}
            )
        )
        respuesta = cliente.describe_endpoint(
            EndpointName=VERSION_ENDPOINT
        )
        servicios["sagemaker"] = respuesta["EndpointStatus"]
    except ClientError as error:
        codigo_error = error.response.get("Error", {}).get("Code")
        servicios["sagemaker"] = (
            "no_verificado_por_permisos"
            if codigo_error in ("AccessDenied", "AccessDeniedException")
            else "no_disponible"
        )
    except Exception:
        servicios["sagemaker"] = "no_verificado"

    funcionando = (
        servicios["postgresql"] == "ok"
        and servicios["rabbitmq"] == "ok"
        and servicios["sagemaker"] == "InService"
        and servicios["llm"].startswith("worker_conectado")
    )

    return {
        "estado": "ok" if funcionando else "degradado",
        "servicios": servicios,
        "endpoint_sagemaker": VERSION_ENDPOINT,
        "nota": "El estado del worker no verifica directamente la API NVIDIA."
    }


@app.get("/api/revision", dependencies=[Depends(validar_api_key)])
def listar_revision():
    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, decision, probabilidad_fraude,
                           fecha_evaluacion, estado_llm,
                           resumen_llm, razones, origen_decision, monto
                    FROM evaluaciones
                    WHERE decision IN ('revisar', 'bloquear')
                    ORDER BY
                        CASE WHEN decision = 'bloquear' THEN 0 ELSE 1 END,
                        probabilidad_fraude DESC NULLS LAST,
                        monto DESC NULLS LAST,
                        fecha_evaluacion DESC
                    LIMIT 100
                    """
                )
                filas = cursor.fetchall()

        return [
            {
                "evaluacion_id": fila[0],
                "decision": fila[1],
                "probabilidad_fraude": fila[2],
                "fecha_evaluacion": fila[3].isoformat(),
                "estado_llm": fila[4],
                "resumen_llm": fila[5],
                "razones": fila[6],
                "origen_decision": fila[7],
                "monto": float(fila[8]) if fila[8] is not None else None
            }
            for fila in filas
        ]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No fue posible consultar la cola de revision."
        )



@app.get("/api/dashboard", dependencies=[Depends(validar_api_key)])
def obtener_dashboard():
    try:
        with obtener_conexion() as conexion:
            with conexion.cursor() as cursor:
                cursor.execute("""
                    SELECT
                        COUNT(*),
                        COUNT(*) FILTER (WHERE decision = 'aprobar'),
                        COUNT(*) FILTER (WHERE decision = 'revisar'),
                        COUNT(*) FILTER (WHERE decision = 'bloquear'),
                        COUNT(*) FILTER (WHERE estado_llm = 'degradado')
                    FROM evaluaciones
                """)

                total, aprobadas, revision, bloqueadas, degradadas = (
                    cursor.fetchone()
                )

                cursor.execute("""
                    SELECT
                        (fecha_evaluacion AT TIME ZONE
                         'America/Santiago')::date AS dia,
                        COUNT(*) AS total,
                        COUNT(*) FILTER (
                            WHERE decision = 'bloquear'
                        ) AS bloqueadas
                    FROM evaluaciones
                    GROUP BY dia
                    ORDER BY dia DESC
                    LIMIT 14
                """)

                registros = cursor.fetchall()

        por_dia = [
            {
                "fecha": fila[0].isoformat(),
                "total": fila[1],
                "bloqueadas": fila[2],
                "tasa_bloqueo": round(
                    fila[2] * 100 / fila[1], 2
                ) if fila[1] else 0
            }
            for fila in registros
        ]

        return {
            "total_evaluaciones": total,
            "decisiones": {
                "aprobar": aprobadas,
                "revisar": revision,
                "bloquear": bloqueadas
            },
            "tasa_bloqueo_global": round(
                bloqueadas * 100 / total, 2
            ) if total else 0,
            "evaluaciones_degradadas": degradadas,
            "bloqueos_por_dia": por_dia
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="No fue posible obtener las estadisticas."
        )


@app.get("/")
def inicio():
    return {
        "mensaje": "Kipu FraudShield activo",
        "documentacion": "/docs"
    }