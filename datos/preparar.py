from pathlib import Path
import pandas as pd

# Identificar la carpeta principal del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Carpeta donde se almacenarán los archivos CSV originales
RUTA_DATOS = RUTA_PROYECTO / "datos" / "originales"

print("Proyecto Kipu FraudShield")
print(f"Carpeta principal: {RUTA_PROYECTO}")
print(f"Carpeta de datos: {RUTA_DATOS}")

# Cargar los cuatro archivos CSV originales
datos = {
    "transacciones": pd.read_csv(RUTA_DATOS / "transacciones.csv"),
    "usuarios": pd.read_csv(RUTA_DATOS / "usuarios.csv"),
    "dispositivos": pd.read_csv(RUTA_DATOS / "dispositivos.csv"),
    "lista_negra": pd.read_csv(RUTA_DATOS / "lista_negra.csv")
}

# Comprobar las dimensiones de cada dataset
for nombre, df in datos.items():
    print(f"{nombre}: {df.shape[0]} filas y {df.shape[1]} columnas")

# Conservar los datasets originales
datos_originales = {
    nombre: df.copy()
    for nombre, df in datos.items()
}

# Crear copias para aplicar la limpieza
datos_limpios = {
    nombre: df.copy()
    for nombre, df in datos.items()
}

print("\n=== COPIAS PARA LIMPIEZA ===")

for nombre, df in datos_limpios.items():
    print(f"{nombre}: {df.shape[0]} filas y {df.shape[1]} columnas")

# Eliminación de duplicados exactos
print("\n=== VALIDACIÓN DE DUPLICADOS ===")

for nombre, df in datos_limpios.items():

    filas_antes = len(df)
    duplicados = int(df.duplicated().sum())

    datos_limpios[nombre] = df.drop_duplicates().copy()

    filas_despues = len(datos_limpios[nombre])

    # Verificar que la limpieza fue correcta
    assert filas_despues == filas_antes - duplicados
    assert not datos_limpios[nombre].duplicated().any()

    print(f"\nTabla: {nombre}")
    print(f"Registros iniciales: {filas_antes}")
    print(f"Duplicados eliminados: {duplicados}")
    print(f"Registros finales: {filas_despues}")
    print("Estado: OK")

# Normalización de categorías
# 1. Limpiar textos
columnas = {
    "transacciones": ["canal", "resultado_autorizacion", "pais_transaccion"],
    "usuarios": ["segmento", "pais"],
    "dispositivos": ["tipo", "pais_declarado"],
    "lista_negra": ["tipo", "motivo"]
}

for tabla, nombres in columnas.items():
    for columna in nombres:
        datos_limpios[tabla][columna] = (
            datos_limpios[tabla][columna]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace("", pd.NA)
        )

# 2. Unificar segmentos de usuarios
datos_limpios["usuarios"]["segmento"] = (
    datos_limpios["usuarios"]["segmento"].replace({
        "individuo": "persona",
        "comercio pequeño": "comercio_pequeno",
        "comercio grande": "comercio_grande"
    })
)

# 3. Unificar países
equivalencias_paises = {
    "chile": "cl", "argentina": "ar",
    "peru": "pe", "perú": "pe",
    "brasil": "br", "brazil": "br",
    "colombia": "co", "venezuela": "ve",
    "usa": "us", "eeuu": "us",
    "estados unidos": "us",
    "rusia": "ru", "russia": "ru",
    "nigeria": "ng"
}

columnas_pais = {
    "transacciones": "pais_transaccion",
    "usuarios": "pais",
    "dispositivos": "pais_declarado"
}

for tabla, columna in columnas_pais.items():
    datos_limpios[tabla][columna] = (
        datos_limpios[tabla][columna].replace(equivalencias_paises)
    )

# 4. Verificar resultados
print("\n=== PAÍSES NORMALIZADOS ===")

resumen_paises = pd.DataFrame({
    tabla: datos_limpios[tabla][columna].value_counts()
    for tabla, columna in columnas_pais.items()
}).fillna(0).astype(int)

print(resumen_paises.to_string())

print("\n=== SEGMENTOS NORMALIZADOS ===")
print(datos_limpios["usuarios"]["segmento"].value_counts(dropna=False))

print("\n=== TIPOS DE DISPOSITIVO ===")
print(datos_limpios["dispositivos"]["tipo"].value_counts(dropna=False))

# Limpieza y normalización de montos y monedas
tx = datos_limpios["transacciones"]

# 1. Conservar montos originales
if "monto_original" not in tx.columns:
    tx["monto_original"] = tx["monto"].copy()

# 2. Normalizar monedas
tx["moneda"] = (
    tx["moneda"].astype("string")
    .str.strip()
    .str.upper()
    .replace("", pd.NA)
)

# 3. Limpiar formatos de montos
texto = (
    tx["monto_original"].astype("string")
    .str.strip()
    .str.replace("$", "", regex=False)
)

# Corregir separadores de miles en CLP
clp_miles = (
    tx["moneda"].eq("CLP").fillna(False)
    & texto.str.fullmatch(r"^-?\d{1,3}\.\d{3}$").fillna(False)
)

texto.loc[clp_miles] = (
    texto.loc[clp_miles].str.replace(".", "", regex=False)
)

# Convertir comas decimales y montos numéricos
tx["monto"] = pd.to_numeric(
    texto.str.replace(",", ".", regex=False),
    errors="coerce"
).astype("Float64")

# 4. Marcar valores problemáticos
tx["moneda_desconocida"] = (
    tx["moneda"].isna().astype(int)
)

original = (
    tx["monto_original"].astype("string")
    .str.strip()
    .str.replace("$", "", regex=False)
)

tx["monto_ambiguo"] = (
    tx["moneda"].isna()
    & original.str.fullmatch(
        r"^-?\d{1,3}[.,]\d{3}$"
    ).fillna(False)
).astype(int)

tx["monto_negativo"] = (
    tx["monto"].lt(0).fillna(False).astype(int)
)

tx["monto_cero"] = (
    tx["monto"].eq(0).fillna(False).astype(int)
)

# 5. Mostrar resultados
print("\n=== RESUMEN DE LIMPIEZA DE MONTOS ===")
print("Transacciones:", len(tx))
print("Montos nulos:", tx["monto"].isna().sum())
print("CLP corregidos:", clp_miles.sum())
print("Monedas desconocidas:", tx["moneda_desconocida"].sum())
print("Montos ambiguos:", tx["monto_ambiguo"].sum())
print("Montos negativos:", tx["monto_negativo"].sum())
print("Montos en cero:", tx["monto_cero"].sum())

print("\n=== DISTRIBUCIÓN DE MONEDAS ===")
print(tx["moneda"].value_counts(dropna=False))

print("\n=== EJEMPLOS DE CONVERSIÓN ===")
print(tx[["monto_original", "monto", "moneda"]].head(10).to_string(index=False))

# Limpieza y normalización de fechas
# 1. Fechas de usuarios y lista negra
fechas_simples = {
    "usuarios": "fecha_registro",
    "lista_negra": "fecha_alta"
}

for tabla, columna in fechas_simples.items():
    df = datos_limpios[tabla]
    original = f"{columna}_original"

    if original not in df.columns:
        df[original] = df[columna].copy()

    df[columna] = pd.to_datetime(
        df[original].astype("string").str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    print(f"\n=== FECHAS: {tabla.upper()} ===")
    print("Fechas convertidas:", df[columna].notna().sum())
    print("Fechas no convertidas:", df[columna].isna().sum())

# Segunda parte: fechas de transacciones y dispositivos
fechas_timestamps = {
    "transacciones": "timestamp",
    "dispositivos": "primera_vez"
}

for tabla, columna in fechas_timestamps.items():
    df = datos_limpios[tabla]
    original = f"{columna}_original"

    if original not in df.columns:
        df[original] = df[columna].copy()

    texto = df[original].astype("string").str.strip()

    # Identificar fechas con zona horaria
    tiene_zona = texto.str.contains(
        r"(?:Z|[+-]\d{2}:\d{2})$",
        regex=True,
        na=False
    )

    df[f"{columna}_tiene_zona"] = tiene_zona

    # Convertir fechas con zona horaria a UTC
    df[f"{columna}_utc"] = pd.to_datetime(
        texto.where(tiene_zona),
        format="mixed",
        utc=True,
        errors="coerce"
    )

    # Convertir fechas sin zona, sin inventar una
    sin_zona = texto.where(~tiene_zona)

    formato_dia_primero = sin_zona.str.contains(
        "/", regex=False, na=False
    )

    fechas_locales = pd.Series(
        pd.NaT, index=df.index, dtype="datetime64[ns]"
    )

    fechas_locales.loc[formato_dia_primero] = pd.to_datetime(
        sin_zona.loc[formato_dia_primero],
        format="mixed",
        dayfirst=True,
        errors="coerce"
    )

    fechas_locales.loc[~formato_dia_primero] = pd.to_datetime(
        sin_zona.loc[~formato_dia_primero],
        format="mixed",
        errors="coerce"
    )

    df[f"{columna}_sin_zona"] = fechas_locales

    # Validar resultados
    validas = (
        df[f"{columna}_utc"].notna()
        | df[f"{columna}_sin_zona"].notna()
    )

    print(f"\n=== FECHAS: {tabla.upper()} ===")
    print("Registros:", len(df))
    print("Fechas con zona:", tiene_zona.sum())
    print("Fechas sin zona:", (~tiene_zona).sum())
    print("Fechas normalizadas:", validas.sum())
    print("Fechas sin convertir:", (~validas).sum())

    # Comprobar que ninguna fecha tenga dos representaciones
    fechas_duplicadas = (
        df[f"{columna}_utc"].notna()
        & df[f"{columna}_sin_zona"].notna()
    )

    print(
        "Fechas con ambas representaciones:",
        fechas_duplicadas.sum()
    )

    assert not fechas_duplicadas.any(), (
        f"Error: fechas duplicadas en {tabla}"
    )

# Limpieza y validación de valores faltantes e inválidos
tx = datos_limpios["transacciones"]
usuarios = datos_limpios["usuarios"]

# 1. Normalizar variables booleanas
mapa_bool = {
    "1": 1, "1.0": 1, "si": 1, "sí": 1, "true": 1,
    "0": 0, "0.0": 0, "no": 0, "false": 0
}

booleanas = {
    "usuarios": ["telefono_verificado", "verificado"],
    "dispositivos": ["emulador"],
    "lista_negra": ["vigente"]
}

for tabla, columnas in booleanas.items():
    df = datos_limpios[tabla]

    for col in columnas:
        original = f"{col}_original"

        if original not in df.columns:
            df[original] = df[col].copy()

        df[col] = (
            df[original].astype("string")
            .str.strip().str.lower()
            .map(mapa_bool).astype("Int64")
        )

# Validación de valores booleanos normalizados
print("\n=== VALIDACIÓN DE BOOLEANOS ===")

for tabla, columnas in booleanas.items():
    for columna in columnas:
        serie = datos_limpios[tabla][columna]

        print(f"\n{tabla}.{columna}")
        print("Verdaderos (1):", int(serie.eq(1).sum()))
        print("Falsos (0):", int(serie.eq(0).sum()))
        print("Desconocidos:", int(serie.isna().sum()))

        # Comprobar que no existan otros valores
        assert serie.dropna().isin([0, 1]).all()

print("\nValidación de booleanos: OK")

# Tratamiento de niveles KYC
if "nivel_kyc_original" not in usuarios.columns:
    usuarios["nivel_kyc_original"] = usuarios["nivel_kyc"].copy()

kyc = pd.to_numeric(
    usuarios["nivel_kyc_original"],
    errors="coerce"
)

usuarios["kyc_especial"] = (
    kyc.eq(-1).fillna(False).astype(int)
)

usuarios["kyc_desconocido"] = (
    kyc.isna().astype(int)
)

usuarios["nivel_kyc_limpio"] = (
    kyc.where(kyc.isin([0, 1, 2, 3])).astype("Int64")
)

print("\n=== VALIDACIÓN KYC ===")
print("KYC especiales:", usuarios["kyc_especial"].sum())
print("KYC desconocidos:", usuarios["kyc_desconocido"].sum())

# Tratamiento de intentos previos
col = "intentos_previos_1h"

if f"{col}_original" not in tx.columns:
    tx[f"{col}_original"] = tx[col].copy()

intentos = pd.to_numeric(
    tx[f"{col}_original"],
    errors="coerce"
)

tx["intentos_previos_desconocidos"] = (
    intentos.isna().astype(int)
)

tx["intentos_previos_invalidos"] = (
    intentos.notna() &
    ((intentos < 0) | (intentos % 1 != 0))
).astype(int)

print("\n=== VALIDACIÓN DE INTENTOS PREVIOS ===")
print("Intentos desconocidos:", tx["intentos_previos_desconocidos"].sum())
print("Intentos inválidos:", tx["intentos_previos_invalidos"].sum())

# Validación de coordenadas geográficas
for col in ["latitud", "longitud"]:
    if f"{col}_original" not in tx.columns:
        tx[f"{col}_original"] = tx[col].copy()

lat = pd.to_numeric(
    tx["latitud_original"], errors="coerce"
)

lon = pd.to_numeric(
    tx["longitud_original"], errors="coerce"
)

# Identificar coordenadas faltantes
tx["coordenadas_faltantes"] = (
    lat.isna() | lon.isna()
).astype(int)

# Identificar coordenadas (0, 0)
tx["coordenadas_cero"] = (
    lat.eq(0) & lon.eq(0)
).astype(int)

# Identificar coordenadas fuera de los rangos válidos
tx["coordenadas_fuera_rango"] = (
    lat.notna() & lon.notna() &
    (~lat.between(-90, 90) | ~lon.between(-180, 180))
).astype(int)

# Consolidar las anomalías
tx["coordenadas_sospechosas"] = (
    tx[
        [
            "coordenadas_faltantes",
            "coordenadas_cero",
            "coordenadas_fuera_rango"
        ]
    ].any(axis=1).astype(int)
)

print("\n=== VALIDACIÓN DE COORDENADAS ===")
print("Coordenadas faltantes:", tx["coordenadas_faltantes"].sum())
print("Coordenadas (0,0):", tx["coordenadas_cero"].sum())
print("Coordenadas fuera de rango:", tx["coordenadas_fuera_rango"].sum())
print("Total sospechosas:", tx["coordenadas_sospechosas"].sum())

# Identificar campos faltantes
identificadores = [
    "dispositivo_id",
    "comercio_id",
    "tarjeta_enmascarada",
    "ip"
]

for col in identificadores:
    tx[f"{col}_faltante"] = tx[col].isna().astype(int)

# Mostrar resultados
print("\n=== IDENTIFICADORES FALTANTES ===")

for col in identificadores:
    print(
        f"{col}:",
        int(tx[f"{col}_faltante"].sum())
    )


# Tratamiento de variables categóricas faltantes
categoricas = {
    "usuarios": ["email_dominio"],
    "dispositivos": ["sistema_operativo"],
    "lista_negra": ["agregado_por", "motivo"]
}

for tabla, columnas in categoricas.items():
    df = datos_limpios[tabla]

    for col in columnas:
        limpio = (
            df[col].astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

        # Indicador de valor faltante
        df[f"{col}_faltante"] = limpio.isna().astype(int)

        # Versión limpia de la categoría
        df[f"{col}_limpio"] = limpio.fillna("desconocido")

# Mostrar resultados
print("\n=== CATEGORÍAS FALTANTES ===")

for tabla, columnas in categoricas.items():
    for col in columnas:
        print(
            f"{tabla}.{col}:",
            int(datos_limpios[tabla][f"{col}_faltante"].sum())
        )

# Tratamiento del historial de transacciones
col = "transacciones_previas"

if f"{col}_original" not in usuarios.columns:
    usuarios[f"{col}_original"] = usuarios[col].copy()

historial = pd.to_numeric(
    usuarios[f"{col}_original"],
    errors="coerce"
)

# Identificar valores desconocidos
usuarios["transacciones_previas_desconocidas"] = (
    historial.isna().astype(int)
)

# Identificar valores negativos o decimales
usuarios["transacciones_previas_invalidas"] = (
    historial.notna() &
    ((historial < 0) | (historial % 1 != 0))
).astype(int)

# Conservar únicamente conteos válidos
usuarios["transacciones_previas_limpio"] = historial.where(
    (historial >= 0) & (historial % 1 == 0)
).astype("Int64")

print("\n=== VALIDACIÓN DEL HISTORIAL ===")
print(
    "Historial desconocido:",
    usuarios["transacciones_previas_desconocidas"].sum()
)
print(
    "Historial inválido:",
    usuarios["transacciones_previas_invalidas"].sum()
)

# Limpieza y validación de etiquetas de fraude
tx = datos_limpios["transacciones"]

# 1. Conservar etiquetas originales
if "es_fraude_original" not in tx.columns:
    tx["es_fraude_original"] = tx["es_fraude"].copy()

# 2. Normalizar etiquetas
etiquetas = (
    tx["es_fraude_original"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace("", pd.NA)
)

mapa_fraude = {
    "0": 0, "0.0": 0, "no": 0,
    "1": 1, "1.0": 1, "si": 1, "sí": 1
}

tx["es_fraude_limpio"] = (
    etiquetas.map(mapa_fraude).astype("Int64")
)

# 3. Separar datos etiquetados y sin etiquetar
tx_etiquetadas = tx.loc[
    tx["es_fraude_limpio"].notna()
].copy()

tx_sin_etiqueta = tx.loc[
    tx["es_fraude_limpio"].isna()
].copy()

tx_etiquetadas["es_fraude_limpio"] = (
    tx_etiquetadas["es_fraude_limpio"].astype(int)
)

# 4. Mostrar resultados
print("\n=== ETIQUETAS NORMALIZADAS ===")
print(tx["es_fraude_limpio"].value_counts(dropna=False))

print("\n=== SEPARACIÓN DE DATOS ===")
print("Transacciones etiquetadas:", len(tx_etiquetadas))
print("Transacciones sin etiqueta:", len(tx_sin_etiqueta))


# Validación final de etiquetas e identificadores
no_reconocidas = (
    etiquetas.notna()
    & tx["es_fraude_limpio"].isna()
)

ids_duplicados = int(
    tx_etiquetadas["transaccion_id"].duplicated().sum()
)

ids_faltantes = int(
    tx_etiquetadas["transaccion_id"].isna().sum()
)

usuarios_unicos = int(
    tx_etiquetadas["usuario_id"].nunique()
)

print("\n=== VALIDACIÓN FINAL DE ETIQUETAS ===")
print("Etiquetas no reconocidas:", int(no_reconocidas.sum()))
print("Total conservado:", len(tx))

print("\n=== VALIDACIÓN DE IDENTIFICADORES ===")
print("Transacciones etiquetadas:", len(tx_etiquetadas))
print("IDs duplicados:", ids_duplicados)
print("IDs faltantes:", ids_faltantes)
print("Usuarios diferentes:", usuarios_unicos)

assert len(tx_etiquetadas) + len(tx_sin_etiqueta) == len(tx)
assert not no_reconocidas.any()
assert ids_duplicados == 0
assert ids_faltantes == 0

print("Estado: OK")

# Integración de transacciones con usuarios
tx = datos_limpios["transacciones"]
usuarios = datos_limpios["usuarios"]

# Seleccionar características relevantes de los usuarios
columnas_usuarios = [
    "usuario_id",
    "pais",
    "segmento",
    "fecha_registro",
    "edad_cuenta_dias",
    "nivel_kyc_limpio",
    "kyc_especial",
    "kyc_desconocido",
    "verificado",
    "telefono_verificado",
    "transacciones_previas_limpio",
    "transacciones_previas_desconocidas",
    "email_dominio_limpio"
]

# Integración mediante LEFT JOIN
datos_integrados = tx.merge(
    usuarios[columnas_usuarios],
    on="usuario_id",
    how="left",
    validate="many_to_one",
    indicator="_relacion_usuario"
)

# Validar resultados
print("\n=== INTEGRACIÓN TRANSACCIONES + USUARIOS ===")
print("Transacciones originales:", len(tx))
print("Registros integrados:", len(datos_integrados))

print("\nEstado de integración:")
print(datos_integrados["_relacion_usuario"].value_counts())

print("\nTransacciones duplicadas después del JOIN:")
print(datos_integrados["transaccion_id"].duplicated().sum())

# Integración de transacciones con dispositivos
dispositivos = datos_limpios["dispositivos"]

# Seleccionar características relevantes
columnas_dispositivos = [
    "dispositivo_id",
    "tipo",
    "sistema_operativo_limpio",
    "usuarios_distintos",
    "emulador",
    "pais_declarado",
    "primera_vez_utc",
    "primera_vez_sin_zona"
]

# Integrar dispositivos mediante LEFT JOIN
datos_integrados = datos_integrados.merge(
    dispositivos[columnas_dispositivos],
    on="dispositivo_id",
    how="left",
    validate="many_to_one",
    indicator="_relacion_dispositivo"
)

# Validar resultados
print("\n=== INTEGRACIÓN DE DISPOSITIVOS ===")
print("Registros integrados:", len(datos_integrados))

print("\nEstado de integración:")
print(datos_integrados["_relacion_dispositivo"].value_counts())

print("\nTransacciones duplicadas:")
print(datos_integrados["transaccion_id"].duplicated().sum())

# Preparación de la lista negra
lista_negra = datos_limpios["lista_negra"]

# Conservar el tipo original
lista_negra["tipo_original"] = lista_negra["tipo"].copy()

# Normalizar tipos de identificadores
lista_negra["tipo"] = (
    lista_negra["tipo"]
    .astype("string")
    .str.strip()
    .str.lower()
    .replace({
        "device": "dispositivo",
        "merchant": "comercio",
        "card": "tarjeta"
    })
)

print("\n=== TIPOS NORMALIZADOS ===")
print(lista_negra["tipo"].value_counts(dropna=False))

print("\nCantidad de categorías:")
print(lista_negra["tipo"].nunique(dropna=True))

print("\nDuplicados por tipo y valor:")
print(
    lista_negra.duplicated(
        subset=["tipo", "valor"],
        keep=False
    ).sum()
)

print("\nTotal de registros:", len(lista_negra))


# Coincidencias históricas de la lista negra
tx = datos_integrados
ln = datos_limpios["lista_negra"]

relaciones = {
    "ip": "ip",
    "dispositivo": "dispositivo_id",
    "comercio": "comercio_id",
    "tarjeta": "tarjeta_enmascarada"
}

# Obtener fecha de transacción
fecha_tx = tx["timestamp_utc"].dt.date.astype("string")

fecha_sin_zona = (
    tx["timestamp_sin_zona"].dt.date.astype("string")
)

fecha_tx = fecha_tx.fillna(fecha_sin_zona)

# Buscar coincidencias anteriores a la transacción
for tipo, columna in relaciones.items():

    entradas = ln.loc[
        ln["tipo"] == tipo,
        ["valor", "fecha_alta"]
    ].copy()

    primeras_fechas = (
        entradas.groupby("valor")["fecha_alta"].min()
    )

    fecha_alta = (
        tx[columna]
        .astype("string")
        .str.strip()
        .map(primeras_fechas)
    )

    indicador = (
        fecha_alta.dt.date.astype("string") < fecha_tx
    ).fillna(False)

    tx[f"lista_negra_{tipo}_historica"] = (
        indicador.astype(int)
    )

    print(
        f"Coincidencias históricas ({tipo}):",
        tx[f"lista_negra_{tipo}_historica"].sum()
    )

# Indicador general
columnas_historicas = [
    f"lista_negra_{tipo}_historica"
    for tipo in relaciones
]

tx["lista_negra_coincidencia_historica"] = (
    tx[columnas_historicas].max(axis=1)
)

print(
    "\nTransacciones con alguna coincidencia histórica:",
    tx["lista_negra_coincidencia_historica"].sum()
)

print("Total de transacciones:", len(tx))

# Validación final de integración
assert len(datos_integrados) == 28324
assert datos_integrados["transaccion_id"].is_unique
assert datos_integrados[columnas_historicas].notna().all().all()
assert (
    datos_integrados[columnas_historicas].max(axis=1)
    == datos_integrados["lista_negra_coincidencia_historica"]
).all()

print("\nIntegración validada: OK")