
import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from pandas.api import types as ptypes


BASE_DIR = Path(__file__).resolve().parent

TABLAS = {
    "transacciones": ("transacciones", "transaccion_id"),
    "usuarios": ("usuarios", "usuario_id"),
    "dispositivos": ("dispositivos", "dispositivo_id"),
    "lista_negra": ("lista_negra", "entrada_id"),
    "transacciones_integradas": ("integradas", "transaccion_id"),
}


def tipo_postgres(serie):
    dtype = serie.dtype

    if ptypes.is_bool_dtype(dtype):
        return "BOOLEAN"
    if ptypes.is_integer_dtype(dtype):
        return "BIGINT"
    if ptypes.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    if ptypes.is_datetime64_any_dtype(dtype):
        if isinstance(dtype, pd.DatetimeTZDtype):
            return "TIMESTAMPTZ"
        return "TIMESTAMP"

    return "TEXT"


def convertir_valor(valor):
    if pd.isna(valor):
        return None

    if isinstance(valor, pd.Timestamp):
        return valor.to_pydatetime()

    if isinstance(valor, np.generic):
        return valor.item()

    return valor


def crear_tabla(cursor, nombre, dataframe, clave):
    columnas = []

    for columna in dataframe.columns:
        tipo = tipo_postgres(dataframe[columna])
        definicion = sql.SQL("{} {}").format(
            sql.Identifier(columna),
            sql.SQL(tipo)
        )

        if columna == clave:
            definicion += sql.SQL(" PRIMARY KEY")

        columnas.append(definicion)

    consulta = sql.SQL("CREATE TABLE {} ({})").format(
        sql.Identifier(nombre),
        sql.SQL(", ").join(columnas)
    )

    cursor.execute(consulta)


def insertar_datos(cursor, nombre, dataframe):
    columnas = list(dataframe.columns)

    consulta = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier(nombre),
        sql.SQL(", ").join(
            sql.Identifier(columna) for columna in columnas
        ),
        sql.SQL(", ").join(
            sql.Placeholder() for _ in columnas
        )
    )

    registros = (
        tuple(convertir_valor(valor) for valor in fila)
        for fila in dataframe.itertuples(index=False, name=None)
    )

    cursor.executemany(consulta, registros)


def main():
    database_url = os.environ.get("DATABASE_URL")

    if not database_url:
        raise RuntimeError("No se encontró la variable DATABASE_URL.")

    print("Preparando los datos...")
    resultado = runpy.run_path(str(BASE_DIR / "preparar.py"))

    datos_limpios = resultado["datos_limpios"]
    datos_integrados = resultado["datos_integrados"]

    dataframes = {
        "transacciones": datos_limpios["transacciones"],
        "usuarios": datos_limpios["usuarios"],
        "dispositivos": datos_limpios["dispositivos"],
        "lista_negra": datos_limpios["lista_negra"],
        "transacciones_integradas": datos_integrados,
    }

    print("\nValidando claves primarias...")

    for nombre, dataframe in dataframes.items():
        clave = TABLAS[nombre][1]

        if dataframe[clave].isna().any():
            raise ValueError(f"{nombre}: existen claves nulas.")

        if dataframe[clave].duplicated().any():
            raise ValueError(f"{nombre}: existen claves duplicadas.")

        print(f"{nombre}: {len(dataframe)} registros, clave {clave} OK")

    print("\nConectando a PostgreSQL...")

    with psycopg.connect(database_url) as conexion:
        with conexion.cursor() as cursor:

            print("\nReemplazando tablas de datos...")

            for nombre in dataframes:
                cursor.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(nombre)
                    )
                )

            for nombre, dataframe in dataframes.items():
                clave = TABLAS[nombre][1]

                crear_tabla(cursor, nombre, dataframe, clave)
                insertar_datos(cursor, nombre, dataframe)

                print(
                    f"Tabla {nombre}: {len(dataframe)} registros cargados."
                )

    print("\nCarga finalizada correctamente.")
    print("Las tablas de datos fueron cargadas.")
    print("La tabla evaluaciones no fue modificada.")


if __name__ == "__main__":
    main()
