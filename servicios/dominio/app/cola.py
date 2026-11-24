import json
import os
import pika

NOMBRE_COLA = "resumenes_fraude"

def enviar_a_cola(mensaje: dict) -> bool:
    """Publica un mensaje para que lo procese el worker."""

    conexion = None

    try:
        credenciales = pika.PlainCredentials(
            os.environ["RABBITMQ_USER"],
            os.environ["RABBITMQ_PASSWORD"]
        )

        parametros = pika.ConnectionParameters(
            host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
            credentials=credenciales,
            connection_attempts=1,
            socket_timeout=2,
            blocked_connection_timeout=2
        )

        conexion = pika.BlockingConnection(parametros)
        canal = conexion.channel()

        canal.queue_declare(
            queue=NOMBRE_COLA,
            durable=True
        )

        canal.basic_publish(
            exchange="",
            routing_key=NOMBRE_COLA,
            body=json.dumps(mensaje).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=2
            )
        )

        return True

    except Exception:
        return False

    finally:
        if conexion is not None and conexion.is_open:
            conexion.close()