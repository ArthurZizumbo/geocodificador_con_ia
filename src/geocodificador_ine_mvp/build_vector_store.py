# src/geocodificador_ine_mvp/build_vector_store.py
import os
import polars as pl
import typer
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

app = typer.Typer()

# Definir solo las columnas que necesitamos para construir el vector store esto hace que la carga de datos sea mucho más rápida y consuma menos memoria.
REQUIRED_COLUMNS_WITH_DTYPES = {
    'domicilio_unico_id': pl.Utf8, 
    'domicilio_canonico': pl.Utf8,
    'latitud': pl.Float64,
    'longitud': pl.Float64,
    'cve_entidad': pl.Utf8
}

@app.command()
def build_vector_store(
    input_path: Optional[Path] = typer.Option(None, "--input-path", "-i", help="Ruta al CSV unificado y preprocesado."),
    db_path: Optional[Path] = typer.Option(None, "--db-path", "-d", help="Directorio donde se guardará la base de datos vectorial."),
    batch_size: int = typer.Option(5000, "--batch-size", "-b", help="Tamaño de los lotes para la inserción de datos.")
):
    """
    Lee un CSV de domicilios canónicos, genera sus embeddings y los almacena en una
    base de datos vectorial ChromaDB particionada por entidad.
    """
    # Si no hay entradas carga los archivos desde las variables de entorno
    if not input_path:
        input_path = Path(os.getenv('PROCESSED_DATA_PATH', 'data/output/domicilios_unificado_preprocesado.csv'))
    if not db_path:
        db_path = Path(os.getenv('VECTOR_STORE_PATH', 'vector_store/'))
    
    # Cargar el archivo procesado
    typer.echo(f"➡️  Cargando datos desde: {input_path}")
    if not input_path.exists():
        typer.secho(f"❌ Error: El archivo no existe: {input_path}", fg=typer.colors.RED)
        raise typer.Exit(1)
    
    # Usar scan_csv y select para leer solo las columnas necesarias.
    df = pl.scan_csv(
        input_path, 
        schema_overrides=REQUIRED_COLUMNS_WITH_DTYPES
    ).select(
        list(REQUIRED_COLUMNS_WITH_DTYPES.keys())
    ).collect()
    # Limpiar nulos usando el nuevo ID y el domicilio canónico.
    df_clean = df.drop_nulls(subset=['domicilio_canonico', 'domicilio_unico_id', 'latitud', 'longitud'])
    typer.echo(f"✔️  {df_clean.height:,} domicilios canónicos cargados.")

    # Inicializar cliente de ChromaDB
    typer.echo(f"➡️  Inicializando base de datos vectorial en: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))

    # Cargar el modelo de embeddings
    model_name = os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
    typer.echo(f"➡️  Cargando el modelo de embeddings: {model_name}...")
    embedding_model = SentenceTransformer(model_name)
    typer.echo("✔️  Modelo cargado.")

    # Agrupar el DataFrame por entidad
    unique_entidades = df_clean.get_column("cve_entidad").unique().to_list()
    typer.echo(f"➡️  Procesando {len(unique_entidades)} entidades...")
    
    for cve_entidad in unique_entidades:
        data = df_clean.filter(pl.col("cve_entidad") == cve_entidad)
        collection_name = f"{os.getenv('CHROMA_COLLECTION_PREFIX', 'domicilios_ine_')}{str(cve_entidad).lower()}"
        typer.echo(f"\n--- Procesando Entidad: {cve_entidad} ---")
        typer.echo(f"  - Nombre de la colección: {collection_name}")
        typer.echo(f"  - Número de domicilios: {len(data):,}")

        # Preparar los datos para esta entidad
        documents = data['domicilio_canonico'].to_list()
        ids = data['domicilio_unico_id'].cast(pl.Utf8).to_list()
        metadatas = data.select(['latitud', 'longitud']).to_dicts()

        # Generar embeddings para los documentos de esta entidad
        typer.echo("  - Generando embeddings...")
        embeddings = embedding_model.encode(documents, show_progress_bar=True, normalize_embeddings=True)

        # Crear o obtener la colección para esta entidad
        collection = client.get_or_create_collection(name=collection_name)

        # Añadir los datos a la colección en lotes
        typer.echo(f"  - Añadiendo datos a la colección en lotes de {batch_size}...")
        for i in range(0, len(documents), batch_size):
            end_index = i + batch_size
            collection.add(
                ids=ids[i:end_index],
                embeddings=embeddings[i:end_index].tolist(),
                metadatas=metadatas[i:end_index], #type: ignore
                documents=documents[i:end_index]
            )
        
        typer.secho(f"  ✔️  Entidad {cve_entidad} procesada. Total en colección: {collection.count()}", fg=typer.colors.GREEN)

    typer.secho("\n🎉 ¡Proceso de construcción de la base de datos particionada completado!", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()