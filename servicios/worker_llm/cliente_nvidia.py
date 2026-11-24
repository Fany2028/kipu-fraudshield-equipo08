import os
from openai import OpenAI

def generar_resumen(decision: str, razones: list[str]) -> str:
    """Genera una explicación breve de una decisión de fraude."""

    cliente = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
        timeout=8.0,
        max_retries=0
    )

    respuesta = cliente.chat.completions.create(
        model=os.environ["NVIDIA_MODEL"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente para analistas de fraude. "
                    "Explica brevemente la decisión utilizando "
                    "solo las razones proporcionadas. "
                    "No inventes información."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Decisión: {decision}. "
                    f"Razones: {'; '.join(razones)}"
                )
            }
        ],
        temperature=0.2,
        max_tokens=160
    )

    texto = respuesta.choices[0].message.content

    if not texto:
        raise ValueError("NVIDIA no devolvió una explicación.")

    return texto.strip()