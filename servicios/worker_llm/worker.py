import json
import os
import time

import pika
import psycopg
from openai import RateLimitError

from cliente_nvidia import generar_resumen

NOMBRE_COLA = "resumenes_fraude"

SENALES_PERMITIDAS = {
    "monto_negativo": "Monto negativo",
    "monto_cero": "Monto igual a cero",
    "moneda_desconocida": "Moneda desconocida",
    "monto_ambiguo": "Monto ambiguo",
    "coordenadas_sospechosas": "Coordenadas sospechosas",
    "dispositivo_id_faltante": "Identificador de dispositivo faltante",
    "comercio_id_faltante": "Identificador de comercio faltante",
    "ip_faltante": "Dirección IP faltante",
    "lista_negra_ip": "IP incluida en lista negra vigente",
    "lista_negra_dispositivo": "Dispositivo incluido en lista negra vigente",
    "lista_negra_comercio": "Comercio incluido en lista negra vigente",
    "lista_negra_tarjeta": "Tarjeta incluida en lista negra vigente",
}

def procesar_mensaje(canal, metodo, propiedades, cuerpo):
    try:
        mensaje = json.loads(cuerpo)
        evaluacion_id = int(mensaje["evaluacion_id"])
        decision = mensaje["decision"]

        if decision not in ("aprobar", "revisar", "bloquear"):
            raise ValueError("Decision no valida")

        senales = mensaje.get("senales", [])
        razones = [
            descripcion
            for clave, descripcion in SENALES_PERMITIDAS.items()
            if clave in senales
        ]

        razones_modelo = mensaje.get("razones_modelo", [])
        if isinstance(razones_modelo, list):
            razones.extend(
                razon[:350]
                for razon in razones_modelo[:3]
                if isinstance(razon, str)
            )

        probabilidad = mensaje.get("probabilidad_fraude")
        if probabilidad is not None:
            probabilidad = float(probabilidad)
            if not 0 <= probabilidad <= 1:
                raise ValueError("Probabilidad fuera de rango")
            razones.append(
                f"Probabilidad estimada por SageMaker: {probabilidad * 100:.2f}%"
            )
            razones.append(
                f"Decision aplicada mediante umbrales: {decision}"
            )

    except (ValueError, KeyError, TypeError):
        print("Mensaje invalido descartado.")
        canal.basic_ack(delivery_tag=metodo.delivery_tag)
        return

    resumen = None
    error = None

    for intento in range(3):
        try:
            resumen = generar_resumen(decision, razones)
            break

        except RateLimitError:
            error = "NVIDIA devolvio HTTP 429"
            if intento < 2:
                time.sleep(2 ** (intento + 1))

        except Exception as excepcion:
            error = type(excepcion).__name__
            break

    estado = "completado" if resumen else "degradado"

    try:
        with psycopg.connect(os.environ["DATABASE_URL"]) as conexion:
            with conexion.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE evaluaciones
                    SET estado_llm = %s,
                        resumen_llm = %s,
                        error_llm = %s
                    WHERE id = %s
                    """,
                    (estado, resumen, error, evaluacion_id)
                )

    except Exception as excepcion:
        print("Error guardando resultado:", type(excepcion).__name__)
        canal.basic_nack(
            delivery_tag=metodo.delivery_tag,
            requeue=True
        )
        time.sleep(2)
        return

    canal.basic_ack(delivery_tag=metodo.delivery_tag)
    print("Evaluacion procesada:", evaluacion_id, estado)

def iniciar_worker():
    credenciales = pika.PlainCredentials(
        os.environ["RABBITMQ_USER"],
        os.environ["RABBITMQ_PASSWORD"]
    )

    parametros = pika.ConnectionParameters(
        host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
        credentials=credenciales,
        heartbeat=60
    )

    conexion = pika.BlockingConnection(parametros)
    canal = conexion.channel()

    canal.queue_declare(
        queue=NOMBRE_COLA,
        durable=True
    )

    canal.basic_qos(prefetch_count=1)

    canal.basic_consume(
        queue=NOMBRE_COLA,
        on_message_callback=procesar_mensaje
    )

    print("Worker NVIDIA esperando mensajes...")
    canal.start_consuming()

if __name__ == "__main__":
    iniciar_worker()