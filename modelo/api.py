from pathlib import Path
import json
import os
import secrets
import joblib
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from modelo.cliente_sagemaker import predecir_sagemaker

RUTA = Path(__file__).resolve().parent

# Cargar el modelo y sus variables
modelo = joblib.load(
    RUTA / "kipu_fraudshield_ep3_conservador.joblib"
)

with open(
    RUTA / "kipu_fraudshield_ep3_metadata.json",
    encoding="utf-8"
) as archivo:
    metadata = json.load(archivo)

app = FastAPI(title="Kipu FraudShield")

# Limitar las solicitudes por dirección IP
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)

# Verificar la clave de acceso a la API
def verificar_api_key(x_api_key: str | None = Header(default=None)):
    clave_esperada = os.getenv("KIPU_API_KEY")

    if (
        not clave_esperada
        or not x_api_key
        or not secrets.compare_digest(x_api_key, clave_esperada)
    ):
        raise HTTPException(
            status_code=401,
            detail="Clave de API ausente o invalida"
        )

# Definir las variables que recibe la API
class Transaccion(BaseModel):
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

# Comprobar el estado del servicio
@app.get("/health")
def health():
    return {"estado": "ok", "version": metadata["version"]}

# Realizar una predicción
@app.post("/predict", dependencies=[Depends(verificar_api_key)])
@limiter.limit("5/minute")
def predict(request: Request, transaccion: Transaccion):
    entrada = pd.DataFrame([transaccion.model_dump()])
    entrada = entrada[metadata["variables"]]

    probabilidad = modelo.predict_proba(entrada)[0, 1]
    prediccion = modelo.predict(entrada)[0]

    return {
        "prediccion": int(prediccion),
        "probabilidad_fraude": round(float(probabilidad), 4),
        "version": metadata["version"]
    }

# Realizar predicciones mediante AWS SageMaker
@app.post(
    "/predict/sagemaker",
    dependencies=[Depends(verificar_api_key)]
)
@limiter.limit("5/minute")
def predict_sagemaker(
    request: Request,
    transaccion: Transaccion
):
    try:
        resultado = predecir_sagemaker(
            transaccion.model_dump()
        )

        return resultado

    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="No fue posible consultar SageMaker"
        ) from error