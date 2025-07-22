# tests/conftest.py
import pytest
from pathlib import Path
import os
from dotenv import load_dotenv

# Importamos las funciones principales de nuestros scripts
from geocodificador_ine_mvp.data_preprocessing import process_data # type: ignore
from geocodificador_ine_mvp.build_vector_store import build_vector_store # type: ignore

load_dotenv()

@pytest.fixture(scope="session", autouse=True)
def setup_artifacts():
    """
    Este fixture se ejecuta automáticamente para generar los artefactos del MVP v2.
    Llama directamente a las funciones de los scripts en Python.
    """
    print("\n⚙️  Ejecutando workflow en Python para generar artefactos de prueba para MVP v2...")

    # Definir rutas usando pathlib para consistencia
    root_dir = Path(__file__).parent.parent
    input_dir = root_dir / os.getenv('DATA_INPUT_DIR', 'data/input')
    output_dir = root_dir / os.getenv('DATA_OUTPUT_DIR', 'data/output')
    processed_data_path = output_dir / 'domicilios_unificado_preprocesado.csv'
    vector_store_path = root_dir / os.getenv('VECTOR_STORE_PATH', 'vector_store/')

    try:
        # --- Paso 1: Ejecutar la lógica de preprocesamiento en modo multi-archivo ---
        print("\n--- Paso 1: Ejecutando el preprocesamiento de datos con Polars ---")
        process_data(
            multi_file=True,
            input_dir=input_dir,
            output_path=processed_data_path,
            input_path=None, # No se usa en modo multi-archivo
            # max_workers=os.cpu_count() # Usar todos los cores para las pruebas
        )
        print("✔️  Paso 1 completado.")

        # --- Paso 2: Ejecutar la lógica para construir la base de datos vectorial particionada ---
        print("\n--- Paso 2: Construyendo la base de datos vectorial particionada ---")
        build_vector_store(
            input_path=processed_data_path, 
            db_path=vector_store_path
        )
        print("✔️  Paso 2 completado.")

    except Exception as e:
        pytest.fail(f"La ejecución del workflow en Python falló: {e}")

    print("\n✔️  Artefactos generados exitosamente para las pruebas.")