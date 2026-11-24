from modelo.anonimizar import anonimizar_texto


# Prueba 8: Anonimización de datos personales
def test_anonimizacion_datos_personales():

    texto_original = (
        "RUT: 12.345.678-5, "
        "correo: cliente@example.com, "
        "telefono: +56 9 8765 4321, "
        "tarjeta: 4111 1111 1111 1111"
    )

    texto_anonimizado = anonimizar_texto(texto_original)

    # Comprobar que desaparecieron los datos originales
    assert "12.345.678-5" not in texto_anonimizado
    assert "cliente@example.com" not in texto_anonimizado
    assert "+56 9 8765 4321" not in texto_anonimizado
    assert "4111 1111 1111 1111" not in texto_anonimizado

    # Comprobar que fueron reemplazados
    assert "[RUT]" in texto_anonimizado
    assert "[CORREO]" in texto_anonimizado
    assert "[TELEFONO]" in texto_anonimizado
    assert "[TARJETA]" in texto_anonimizado