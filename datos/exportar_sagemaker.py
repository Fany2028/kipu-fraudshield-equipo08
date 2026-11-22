from pathlib import Path
import runpy
from sklearn.model_selection import GroupShuffleSplit

# Identificar la carpeta principal del proyecto
RAIZ = Path(__file__).resolve().parent.parent

# Ejecutar la preparación de datos existente
resultado = runpy.run_path(
    str(RAIZ / "datos" / "preparar.py")
)

# Seleccionar únicamente las transacciones etiquetadas
df = resultado["datos_integrados"].copy()
df = df.loc[df["es_fraude_limpio"].notna()].copy()

# Variables utilizadas por nuestro Random Forest
numericas = [
    "monto_negativo",
    "monto_cero",
    "moneda_desconocida",
    "monto_ambiguo",
    "coordenadas_sospechosas",
    "dispositivo_id_faltante",
    "comercio_id_faltante",
    "ip_faltante"
]

categoricas = [
    "moneda",
    "canal",
    "mcc",
    "pais_transaccion"
]

columnas = numericas + categoricas + ["es_fraude_limpio"]

# Mantener la separación por usuario del entrenamiento original
divisor = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    divisor.split(
        df,
        df["es_fraude_limpio"],
        groups=df["usuario_id"].astype("string")
    )
)

entrenamiento = df.iloc[train_idx][columnas].copy()
prueba = df.iloc[test_idx][columnas].copy()

# Asegurar que las etiquetas sean 0 o 1
entrenamiento["es_fraude_limpio"] = (
    entrenamiento["es_fraude_limpio"].astype(int)
)

prueba["es_fraude_limpio"] = (
    prueba["es_fraude_limpio"].astype(int)
)

# Crear la carpeta de salida
SALIDA = RAIZ / "datos" / "procesados"
SALIDA.mkdir(parents=True, exist_ok=True)

# Exportar los CSV para SageMaker
entrenamiento.to_csv(
    SALIDA / "train.csv",
    index=False,
    encoding="utf-8"
)

prueba.to_csv(
    SALIDA / "test.csv",
    index=False,
    encoding="utf-8"
)

print("\n=== DATOS PREPARADOS PARA SAGEMAKER ===")
print("Entrenamiento:", len(entrenamiento))
print("Prueba:", len(prueba))
print("Variables de entrada:", len(numericas + categoricas))
print("Carpeta de salida:", SALIDA)