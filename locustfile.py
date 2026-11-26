import os
from locust import HttpUser, task, constant

TRANSACCION_PRUEBA = {
    "monto": 25000,
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

class PruebaKipu(HttpUser):
    wait_time = constant(12)

    @task
    def evaluar_transaccion(self):
        clave = os.environ.get("KIPU_API_KEY")

        if not clave:
            raise RuntimeError(
                "Falta configurar KIPU_API_KEY"
            )

        self.client.post(
            "/api/predict",
            json=TRANSACCION_PRUEBA,
            headers={"X-API-Key": clave},
            name="POST /api/predict",
            timeout=30
        )