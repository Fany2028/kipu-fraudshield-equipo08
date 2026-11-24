import os

def obtener_decision(probabilidad: float) -> str:
    """Determina la acción según el riesgo de fraude."""

    umbral_revision = float(
        os.getenv("UMBRAL_REVISION", "0.30")
    )
    umbral_bloqueo = float(
        os.getenv("UMBRAL_BLOQUEO", "0.70")
    )

    if not 0 <= umbral_revision < umbral_bloqueo <= 1:
        raise ValueError("Umbrales de fraude invalidos")

    if probabilidad >= umbral_bloqueo:
        return "bloquear"

    if probabilidad >= umbral_revision:
        return "revisar"

    return "aprobar"