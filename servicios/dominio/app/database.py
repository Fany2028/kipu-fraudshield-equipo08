import os
import psycopg

DATABASE_URL = os.environ["DATABASE_URL"]


def obtener_conexion():
    return psycopg.connect(DATABASE_URL)


def inicializar_base_datos():
    with obtener_conexion() as conexion:
        with conexion.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluaciones (
                    id BIGSERIAL PRIMARY KEY,
                    prediccion INTEGER NOT NULL,
                    probabilidad_fraude DOUBLE PRECISION NOT NULL,
                    version_modelo VARCHAR(100) NOT NULL,
                    estado_modelo VARCHAR(50) NOT NULL,
                    fecha_evaluacion TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)
