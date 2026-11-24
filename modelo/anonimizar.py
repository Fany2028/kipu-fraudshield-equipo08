import re

def anonimizar_texto(texto: str) -> str:
    """Oculta datos personales antes de enviarlos al LLM."""

    # Ocultar correos electrónicos
    texto = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[CORREO]",
        texto
    )

    # Ocultar tarjetas de 16 dígitos
    texto = re.sub(
        r"\b(?:\d{4}[- ]?){3}\d{4}\b",
        "[TARJETA]",
        texto
    )

    # Ocultar RUT chilenos
    texto = re.sub(
        r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b",
        "[RUT]",
        texto
    )

    # Ocultar teléfonos móviles chilenos
    texto = re.sub(
        r"(?<!\d)(?:\+?56[\s-]?)?9[\s-]?\d{4}[\s-]?\d{4}(?!\d)",
        "[TELEFONO]",
        texto
    )

    return texto