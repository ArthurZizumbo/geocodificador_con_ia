# tests/test_data_preprocessing.py

import pytest
from geocodificador_ine_mvp.data_preprocessing import normalizar_texto_avanzado # type: ignore

# Usamos parametrize de pytest para probar múltiples casos de forma limpia
@pytest.mark.parametrize("input_str, expected_output", [
    # Casos básicos
    ("AVENIDA JUÁREZ, LEÓN", "avenida juarez leon"),
    ("C. Hidalgo esq. con Carr. a Toluca", "calle hidalgo esquina con carretera a toluca"),
    ("Reforma #123", "reforma 123"),
    ("Domicilio Conocido S/N", "domicilio conocido sin numero"),
    
    # Casos de normalización avanzada que descubrimos
    ("cjon diego rivera", "callejon diego rivera"),
    ("mz33a", "manzana 33 a"),
    ("lt3", "lote 3"),
    ("conj hab", "conjunto habitacional"),
    ("ampliacion polvorilla", "ampliacion polvorilla"),
    ("blvd pdte adolfo lopez mateos", "boulevard presidente adolfo lopez mateos"),

    # Casos borde
    ("", ""),
    (None, ""),
    ("123", "123"),
])
def test_normalizar_texto_avanzado(input_str, expected_output):
    """Valida que la función normalizar_texto_avanzado funcione para varios casos."""
    assert normalizar_texto_avanzado(input_str) == expected_output
