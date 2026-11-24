from app.database import obtener_conexion

def consultar_lista_negra(datos: dict) -> list[str]:
    ip = (datos.get("ip") or "").strip()
    dispositivo = (datos.get("dispositivo_id") or "").strip()
    comercio = (datos.get("comercio_id") or "").strip()
    tarjeta = (datos.get("tarjeta_enmascarada") or "").strip()

    if not any((ip, dispositivo, comercio, tarjeta)):
        return []

    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT tipo
                FROM lista_negra
                WHERE vigente = 1
                  AND (
                      (tipo = 'ip' AND TRIM(valor) = %s)
                      OR (tipo = 'dispositivo' AND TRIM(valor) = %s)
                      OR (tipo = 'comercio' AND TRIM(valor) = %s)
                      OR (tipo = 'tarjeta' AND TRIM(valor) = %s)
                  )
                """,
                (ip, dispositivo, comercio, tarjeta)
            )

            coincidencias = cursor.fetchall()

    return [fila[0] for fila in coincidencias]