from pathlib import Path
import json
import joblib
import pandas as pd

# Ubicar los archivos del modelo
RUTA_MODELO = Path(__file__).resolve().parent

# Cargar el modelo previamente entrenado
modelo = joblib.load(
    RUTA_MODELO / "kipu_fraudshield_ep3_conservador.joblib"
)

# Cargar la información de las variables
with open(
    RUTA_MODELO / "kipu_fraudshield_ep3_metadata.json",
    encoding="utf-8"
) as archivo:
    metadata = json.load(archivo)

# Ejemplo de transacción con las 12 variables del modelo
transaccion = {
    "monto_negativo": 0,
    "monto_cero": 0,
    "moneda_desconocida": 0,
    "monto_ambiguo": 0,
    "coordenadas_sospechosas": 0,
    "dispositivo_id_faltante": 0,
    "comercio_id_faltante": 0,
    "ip_faltante": 0,
    "moneda": "CLP",
    "canal": "web",
    "mcc": "5411",
    "pais_transaccion": "cl"
}

# Preparar la entrada respetando las variables del entrenamiento
entrada = pd.DataFrame([transaccion])
entrada = entrada[metadata["variables"]]

# Calcular la probabilidad de fraude
indice_fraude = list(modelo.classes_).index(1)
probabilidad = modelo.predict_proba(entrada)[0, indice_fraude]

print("\n=== PREDICCIÓN KIPU FRAUDSHIELD ===")
print("Versión:", metadata["version"])
print(f"Probabilidad de fraude: {probabilidad:.2%}")