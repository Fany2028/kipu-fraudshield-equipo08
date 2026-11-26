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

            cursor.execute("""
                ALTER TABLE evaluaciones
                ADD COLUMN IF NOT EXISTS estado_llm
                VARCHAR(30) NOT NULL DEFAULT 'no_solicitado';
            """)

            cursor.execute("""
                ALTER TABLE evaluaciones
                ADD COLUMN IF NOT EXISTS resumen_llm TEXT;
            """)

            cursor.execute("""
                ALTER TABLE evaluaciones
                ADD COLUMN IF NOT EXISTS error_llm TEXT;
            """)
            cursor.execute("""
                ALTER TABLE evaluaciones
                ALTER COLUMN prediccion DROP NOT NULL
            """)

            cursor.execute("""
                ALTER TABLE evaluaciones
                ALTER COLUMN probabilidad_fraude DROP NOT NULL
            """)

            cursor.execute("""
                ALTER TABLE evaluaciones
                ADD COLUMN IF NOT EXISTS decision VARCHAR(20)
            """)

            cursor.execute("""
                ALTER TABLE evaluaciones
                ADD COLUMN IF NOT EXISTS origen_decision VARCHAR(40)
            """)

            cursor.execute("""
                ALTER TABLE evaluaciones
                ADD COLUMN IF NOT EXISTS razones JSONB
            """)
            cursor.execute("""
                ALTER TABLE evaluaciones
                ADD COLUMN IF NOT EXISTS monto NUMERIC(18,2)
            """)
