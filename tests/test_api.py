from fastapi.testclient import TestClient
from modelo.api import app

cliente = TestClient(app)

TRANSACCION_VALIDA = {
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

# Prueba 1: Comprobar que la API está operativa
def test_health():
    respuesta = cliente.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "ok"

# Prueba 2: Predicción con datos válidos y autenticación
def test_predict_valido(monkeypatch):
    clave_prueba = "clave-ficticia-solo-para-tests"
    monkeypatch.setenv("KIPU_API_KEY", clave_prueba)

    respuesta = cliente.post(
        "/predict",
        json=TRANSACCION_VALIDA,
        headers={"X-API-Key": clave_prueba}
    )

    assert respuesta.status_code == 200

    datos = respuesta.json()

    assert "prediccion" in datos
    assert "probabilidad_fraude" in datos
    assert "version" in datos
    assert 0 <= datos["probabilidad_fraude"] <= 1

# Prueba 3: Rechazar un valor fuera de rango
def test_predict_invalido(monkeypatch):
    clave_prueba = "clave-ficticia-solo-para-tests"
    monkeypatch.setenv("KIPU_API_KEY", clave_prueba)

    transaccion = TRANSACCION_VALIDA.copy()
    transaccion["monto_negativo"] = 2

    respuesta = cliente.post(
        "/predict",
        json=transaccion,
        headers={"X-API-Key": clave_prueba}
    )

    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == [
        "body",
        "monto_negativo"
    ]

# Prueba 4: Rechazar solicitudes sin clave de API
def test_predict_sin_clave():
    respuesta = cliente.post(
        "/predict",
        json=TRANSACCION_VALIDA
    )

    assert respuesta.status_code == 401
    assert respuesta.json()["detail"] == "Clave de API ausente o invalida"

# Prueba 5: Comprobar el límite de solicitudes HTTP 429
def test_predict_rate_limit(monkeypatch):
    clave_prueba = "clave-ficticia-solo-para-tests"
    monkeypatch.setenv("KIPU_API_KEY", clave_prueba)

    codigos_respuesta = []

    # Enviar seis solicitudes consecutivas
    for _ in range(6):
        respuesta = cliente.post(
            "/predict",
            json=TRANSACCION_VALIDA,
            headers={"X-API-Key": clave_prueba}
        )

        codigos_respuesta.append(respuesta.status_code)

    # Al menos una solicitud debe superar el límite
    assert 429 in codigos_respuesta
