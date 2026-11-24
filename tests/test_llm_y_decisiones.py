import json
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock

from servicios.dominio.app.decisiones import obtener_decision

RUTA_WORKER = Path(__file__).resolve().parents[1] / "servicios" / "worker_llm"
sys.path.insert(0, str(RUTA_WORKER))

worker = importlib.import_module("worker")

# Prueba 9: NVIDIA falla, pero el sistema registra modo degradado
def test_nvidia_modo_degradado(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://prueba:prueba@localhost/prueba"
    )

    def simular_fallo(decision, razones):
        raise RuntimeError("NVIDIA no disponible")

    monkeypatch.setattr(worker, "generar_resumen", simular_fallo)

    conexion_falsa = MagicMock()
    cursor_falso = (
        conexion_falsa.__enter__.return_value
        .cursor.return_value.__enter__.return_value
    )

    monkeypatch.setattr(
        worker.psycopg,
        "connect",
        lambda url: conexion_falsa
    )

    canal = MagicMock()
    metodo = MagicMock()
    metodo.delivery_tag = 123

    mensaje = {
        "evaluacion_id": 10,
        "decision": "bloquear",
        "senales": ["moneda_desconocida"]
    }

    worker.procesar_mensaje(
        canal,
        metodo,
        None,
        json.dumps(mensaje).encode("utf-8")
    )

    parametros = cursor_falso.execute.call_args.args[1]

    assert parametros[0] == "degradado"
    assert parametros[1] is None
    assert parametros[2] == "RuntimeError"
    assert parametros[3] == 10

    canal.basic_ack.assert_called_once_with(delivery_tag=123)
    canal.basic_nack.assert_not_called()

# Prueba 10: Decisiones según probabilidad de fraude
def test_decisiones_automaticas(monkeypatch):
    monkeypatch.setenv("UMBRAL_REVISION", "0.30")
    monkeypatch.setenv("UMBRAL_BLOQUEO", "0.70")

    assert obtener_decision(0.10) == "aprobar"
    assert obtener_decision(0.50) == "revisar"
    assert obtener_decision(0.90) == "bloquear"