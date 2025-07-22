
# Documentación del Proyecto: Geocodificador INE MVP

Este documento detalla la estructura, contenido y propósito de cada archivo dentro del proyecto `geocodificador_ine_mvp`.

## Resumen del Proyecto

El proyecto es una API de geocodificación de alta precisión diseñada para el INE. Utiliza un enfoque de búsqueda híbrida que combina la búsqueda semántica (embeddings vectoriales) y la búsqueda léxica (BM25) para encontrar la correspondencia más precisa para una dirección dada. La base de datos está particionada por entidad federativa para optimizar el rendimiento.

## Estructura del Proyecto

```
C:/Users/arthu/Proyectos/INE/UbicaTuDomicilioIA/geocodificador_ine_mvp/
├───.dockerignore
├───.env
├───.gitignore
├───CLAUDE.md
├───Dockerfile
├───poetry.lock
├───pyproject.toml
├───README.md
├───requirements.txt
├───.pytest_cache/
├───data/
│   ├───input/
│   └───output/
├───notebooks/
├───scripts/
│   └───evaluate_model.py
├───src/
│   └───geocodificador_ine_mvp/
│       ├───__init__.py
│       ├───build_vector_store.py
│       ├───data_preprocessing.py
│       ├───main.py
│       └───search_models.py
├───tests/
│   ├───conftest.py
│   ├───test_data_preprocessing.py
│   └───test_search_models.py
├───utils/
│   └───logger.py
└───vector_store/
```

---

## Archivos Principales y de Configuración

### `Dockerfile`

**Propósito:** Define el entorno de contenedor para la aplicación. Utiliza una compilación de varias etapas para crear una imagen de Docker ligera y eficiente para producción.

**Contenido:**
```dockerfile
# ---- Etapa 1: Builder (Sin cambios) ----
# Usamos una imagen completa para instalar dependencias con Poetry
# FROM python:3.12-slim as builder
FROM python:3.12-slim as base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

FROM base

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos el entorno virtual con las dependencias ya instaladas desde la etapa 'builder'
# COPY --from=builder /app/.venv ./.venv
COPY --from=base /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copiamos los artefactos y el código fuente necesarios
COPY src/ ./src/

# Exponemos el puerto de la API
EXPOSE 8000

# Comando para iniciar el servidor. Uvicorn se encuentra gracias a la variable PATH.
CMD ["uvicorn", "src.geocodificador_ine_mvp.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### `pyproject.toml`

**Propósito:** Archivo de configuración del proyecto para `poetry`. Define metadatos, dependencias de producción y desarrollo, y configuraciones de herramientas como `pytest`.

**Contenido:**
```toml
[tool.poetry]
name = "geocodificador_ine_mvp"
version = "0.1.0"
description = "Geocodificador de alta precisión para el INE usando búsqueda semántica híbrida."
authors = ["Arthur Zizumbo"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.9"
fastapi = "*"
uvicorn = {extras = ["standard"], version = "*"}
sentence-transformers = "*"
faiss-cpu = "*"
numpy = "*"
pandas = "*"
scikit-learn = "*"
typer = "*"
chromadb = "^1.0.15"
rank-bm25 = "^0.2.2"
haversine = "^2.9.0"
python-dotenv = "^1.1.1"
pip = "^25.1.1"
python-json-logger = "^3.3.0"
polars = "^1.31.0"

[tool.poetry.group.dev.dependencies]
pytest = "*"
black = "*"

[tool.pytest.ini_options]
pythonpath = "src"
testpaths = "tests"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### `requirements.txt`

**Propósito:** Lista todas las dependencias de Python necesarias para el proyecto. Es utilizado por entornos que no usan `poetry`, como el `Dockerfile`, para instalar los paquetes requeridos.

**Contenido:**
```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
annotated-types==0.7.0
# ... (lista completa de dependencias)
```

---

## Código Fuente (`src`)

### `src/geocodificador_ine_mvp/data_preprocessing.py`

**Propósito:** Contiene la lógica para la limpieza, normalización y unificación de los datos de domicilios. Utiliza la biblioteca `Polars` para un procesamiento de datos de alto rendimiento.

**Funciones Clave:**
- `normalizar_texto_avanzado(texto)`: Estandariza una cadena de dirección expandiendo abreviaturas, eliminando acentos y caracteres especiales.
- `process_multiple_csv_files(input_dir)`: Lee y unifica múltiples archivos CSV de un directorio, manejando diferentes codificaciones.
- `apply_transformations_and_deduplicate_lazy(df_lazy)`: Define un plan de transformaciones (lazy) en Polars para normalizar, deduplicar y enriquecer los datos.
- `process_data(...)`: Comando principal de Typer que orquesta el proceso de carga, transformación y guardado de los datos procesados.

**Contenido:**
```python
import re
import unicodedata
import polars as pl
import typer
from pathlib import Path
from typing import Optional, List
import os
from dotenv import load_dotenv
import multiprocessing

# Cargar variables de entorno desde un archivo .env si existe
load_dotenv()

# --- Creamos una aplicación Typer para la interfaz de línea de comandos ---
app = typer.Typer()

# --- Constantes y Diccionarios ---

# Mapeo de claves numéricas de entidades a sus abreviaturas estándar
CLAVES_ENTIDADES = {
    1: 'AGS', 2: 'BC', 3: 'BCS', 4: 'CAM', 5: 'COA',
    6: 'COL', 7: 'CHS', 8: 'CHH', 9: 'CMX', 10: 'DGO',
    11: 'GTO', 12: 'GRO', 13: 'HGO', 14: 'JAL', 15: 'MEX',
    16: 'MCH', 17: 'MOR', 18: 'NAY', 19: 'NL', 20: 'OAX',
    21: 'PUE', 22: 'QRO', 23: 'ROO', 24: 'SLP', 25: 'SIN',
    26: 'SON', 27: 'TAB', 28: 'TMS', 29: 'TLX', 30: 'VER',
    31: 'YUC', 32: 'ZAC'
}

# Diccionario centralizado para las sustituciones de texto.
# Usamos expresiones regulares para capturar variaciones (ej. con/sin punto).
ABREVIATURAS_REGEX = {
    # Vialidades
    r'\bav\.?\b': 'avenida', r'\bc\.?\b': 'calle', r'\bblvd\.?\b': 'boulevard',
    r'\bcalz\.?\b': 'calzada', r'\bcda\.?\b': 'cerrada', r'\bpriv\.?\b': 'privada',
    r'\bprol\.?\b': 'prolongacion', r'\band\.?\b': 'andador', r'\bcto\.?\b': 'circuito',
    r'\bcarr\.?\b': 'carretera', r'\bcjon\.?\b': 'callejon', r'\bhda\.?\b': 'hacienda',
    # Asentamientos
    r'\bcol\.?\b': 'colonia', r'\bfracc\.?\b': 'fraccionamiento', r'\bconj\.?\b': 'conjunto', r'\bhab\.?\b': 'habitacional',
    r'\bconj\.?\s?hab\.?\b': 'conjunto habitacional', r'\bu\s?hab\.?\b': 'unidad habitacional',
    r'\buh\.?\b': 'unidad habitacional', r'\bbo\.?\b': 'barrio', r'\bbarr\.?\b': 'barrio', r'\bpblo\.?\b': 'pueblo',
    r'\bsecc\.?\b': 'seccion', r'\bej\.?\b': 'ejido', r'\brcho\.?\b': 'rancho', r'\bampl\.?\b': 'ampliacion', r'\brdcial\.?\b': 'residencial',
    # Interiores y estructuras
    r'\bmz(n|a)?\.?\b': 'manzana', r'\blte\.?\b': 'lote', r'\blt\.?\b': 'lote', r'\bedif\.?\b': 'edificio',
    r'\bd(ep|p)to\.?\b': 'departamento', r'\bint\.?\b': 'interior', r'\bpb\.?\b': 'planta baja',
    # Conectores y varios
    r'\besq\.?\b': 'esquina', r'\s+y\s+': ' ', r'\bs/n\b': 'sin numero',
    r'\b(num|no)\.?\b': '', r'#': '', r'\bkm\.?\b': 'kilometro',
    # Puntos Cardinales y otros
    r'\bnte\.?\b': 'norte', r'\bpte\.?\b': 'poniente', r'\bote\.?\b': 'oriente',
    r'\bpdte\.?\b': 'presidente',
}

# Definir las columnas que necesitamos de la fuente de datos.
REQUIRED_COLUMNS = ['id','dom_id', 'entidad', 'domicilio', 'latitud', 'longitud']

# Se mantiene solo para la normalización de acentos
def normalizar_acentos_udf(texto: str) -> str:
    """Elimina acentos de una cadena de texto."""
    if not texto:
        return ""
    # Descompone los caracteres en su forma base y diacríticos, y luego filtra los diacríticos.
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def normalizar_texto_avanzado(texto: str) -> str:
    """Limpia y estandariza una cadena de texto de una dirección de forma robusta."""
    if not isinstance(texto, str): 
        return ""
    texto = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', texto)
    texto = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', texto)
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    for abrev, completo in ABREVIATURAS_REGEX.items():
        texto = re.sub(abrev, completo, texto)
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def process_multiple_csv_files(input_dir: Path) -> pl.LazyFrame:
    """
    Procesa todos los archivos CSV de un directorio y los unifica en un solo LazyFrame.
    Maneja múltiples codificaciones de forma robusta.
    
    Args:
        input_dir: Directorio que contiene los archivos CSV.
        
    Returns:
        Un LazyFrame unificado con todos los domicilios.
    """
    typer.echo(f"🔍 Buscando archivos CSV en: {input_dir}")
    
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        typer.secho(f"❌ No se encontraron archivos CSV en {input_dir}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    typer.echo(f"📁 Se encontraron {len(csv_files)} archivos CSV.")
    
    lazy_frames: List[pl.LazyFrame] = []
    # Lista de codificaciones a probar.
    encodings_to_try = ['latin1', 'utf-8']

    for file in csv_files:
        typer.echo(f"  📄 Leyendo y decodificando: {file.name}")
        lf = None
        # Leer el archivo en Python y dárselo a Polars en UTF-8 ---
        try:
            # Leer el encabezado para verificar si todas las columnas requeridas existen
            header = pl.read_csv(file, has_header=True, n_rows=0, encoding=encodings_to_try[0]).columns
             # Usar solo las columnas requeridas que realmente existen en el archivo
            cols_to_read = [col for col in REQUIRED_COLUMNS if col in header]
            if 'domicilio' not in cols_to_read:
                typer.secho(f"  ⚠️  Advertencia: La columna 'domicilio' no se encontró en {file.name}. Saltando archivo.", fg=typer.colors.YELLOW)
                continue

            file_bytes = file.read_bytes()
            decoded_content = None
            for encoding in encodings_to_try:
                try:
                    decoded_content = file_bytes.decode(encoding)
                    typer.echo(f"    → Decodificado con: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if decoded_content:
                utf8_bytes = decoded_content.encode('utf-8')
                lf = pl.read_csv(utf8_bytes, ignore_errors=True, schema_overrides={'dom_id': pl.Utf8}).lazy()
                lazy_frames.append(lf)
            else:
                typer.secho(f"  ❌ No se pudo decodificar {file.name} con ninguna codificación probada.", fg=typer.colors.YELLOW)

        except Exception as e:
            typer.secho(f"  ❌ Error fatal procesando {file.name}: {e}", fg=typer.colors.RED)

    if not lazy_frames:
        typer.secho("❌ No se pudo procesar ningún archivo CSV exitosamente.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"\n🔄 Concatenando {len(lazy_frames)} planes de consulta (LazyFrames)...")
    df_lazy = pl.concat(lazy_frames, how="diagonal_relaxed")
    
    total_rows = df_lazy.select(pl.len()).collect().item()
    typer.echo(f"✅ Total de registros unificados (antes de procesar): {total_rows:,}")
    
    return df_lazy

def apply_transformations_and_deduplicate_lazy(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    Aplica todas las transformaciones Y la deduplicación como parte del plan perezoso.
    """
    typer.echo("➡️  Definiendo el plan de transformaciones y deduplicación (Lazy)...")
    
    entidad_mapping_df = pl.DataFrame({
        'entidad': list(CLAVES_ENTIDADES.keys()),
        'cve_entidad': list(CLAVES_ENTIDADES.values())
    }).lazy()

    domicilio_expr = (
        pl.col('domicilio').cast(pl.Utf8)
        .str.replace_all(r'([a-zA-Z])(\d)', r'$1 $2')
        .str.replace_all(r'(\d)([a-zA-Z])', r'$1 $2')
        .str.to_lowercase()
        .map_elements(normalizar_acentos_udf, return_dtype=pl.Utf8, strategy="threading")
    )
    
    for pattern, replacement in ABREVIATURAS_REGEX.items():
        domicilio_expr = domicilio_expr.str.replace_all(pattern, replacement)
        
    domicilio_expr = (
        domicilio_expr
        .str.replace_all(r'[^a-z0-9\s]', '')
        .str.replace_all(r'\s+', ' ').str.strip_chars()
    )

    df_processed_lazy = (
        df_lazy
        # Renombrar 'id' a 'source_id' para claridad.
        .rename({"id": "source_id"})
        .join(entidad_mapping_df, on='entidad', how='left')
        .with_columns([
            domicilio_expr.alias('domicilio_canonico')
        ])
        # Asegurarse de que el domicilio canónico no sea nulo o vacío
        .filter(pl.col('domicilio_canonico').is_not_null() & (pl.col('domicilio_canonico') != ""))
        # **CAMBIO CLAVE 3**: Deduplicar usando la columna correcta.
        .unique(subset=['domicilio_canonico'], keep='first', maintain_order=False)
        # **CAMBIO CLAVE 4**: Crear un ID único y determinista DESPUÉS de la deduplicación.
        .with_columns(
            pl.col("domicilio_canonico").hash().alias("domicilio_unico_id")
        )
        .sort('cve_entidad', maintain_order=False)
    )
    
    return df_processed_lazy

# --- Comando principal de la CLI ---
@app.command()
def process_data(
    input_path: Optional[Path] = typer.Argument(None, help="Ruta a un archivo CSV único."),
    output_path: Optional[Path] = typer.Argument(None, help="Ruta de salida del CSV procesado."),
    multi_file: bool = typer.Option(False, "--multi-file", "-m", help="Activa el modo de procesamiento de múltiples archivos."),
    input_dir: Optional[Path] = typer.Option(None, "--input-dir", "-i", help="Directorio con múltiples CSVs (usado con -m)."),
    n_threads: Optional[int] = typer.Option(None, "--threads", "-t", help="Número de threads para Polars (por defecto: todos los cores).")
):
    """
    Procesa el padrón de domicilios para limpiar, normalizar y crear un
    dataset canónico deduplicado, usando expresiones nativas de Polars para máxima eficiencia.
    """
    if n_threads:
        os.environ["POLARS_MAX_THREADS"] = str(n_threads)
    else:
        os.environ["POLARS_MAX_THREADS"] = str(multiprocessing.cpu_count())
    
    typer.echo(f"🖥️  Configurado para usar {os.environ['POLARS_MAX_THREADS']} threads para operaciones de Polars.")
    
    # --- Parte 1: Carga de Datos (Modo único o múltiple) ---
    if multi_file:
        if not input_dir:
            input_dir = Path(os.getenv('DATA_INPUT_DIR', './data/input'))
        if not output_path:
            output_dir = Path(os.getenv('DATA_OUTPUT_DIR', './data/output'))
            output_path = output_dir / 'domicilios_unificado_preprocesado.csv'
        
        typer.echo("🚀 Modo: Procesamiento de múltiples archivos CSV")
        df_lazy = process_multiple_csv_files(input_dir)
    else:
        if not input_path or not output_path:
            typer.secho("❌ Error: En modo de archivo único, 'input_path' y 'output_path' son obligatorios.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
            
        typer.echo("📄 Modo: Procesamiento de archivo único")
        if not input_path.exists():
            typer.secho(f"❌ Error: El archivo de entrada no existe en {input_path}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
            
        typer.echo(f"➡️  Cargando datos desde: {input_path}")
        df_lazy = pl.scan_csv(input_path).select(REQUIRED_COLUMNS)
    
    # --- Parte 2: Aplicar Transformaciones y Deduplicación (Lazy) ---
    df_final_lazy = apply_transformations_and_deduplicate_lazy(df_lazy)
    
    # --- Parte 3: Ejecución en Streaming ---
    typer.echo("➡️  Ejecutando el plan COMPLETO (transformación + deduplicación) en modo STREAMING...")
    
    df_final = df_final_lazy.collect(engine="streaming")
    
    typer.echo("✔️  Recolección en streaming completada.")
    
    # --- Parte 4: Selección Final y Estadísticas ---
    columnas_finales = [
        'domicilio_unico_id','domicilio_canonico', 
        'latitud', 'longitud', 'cve_entidad', 'source_id', 'dom_id' 
    ]
    columnas_existentes_en_df = [col for col in columnas_finales if col in df_final.columns]
    df_final_seleccionado = df_final.select(columnas_existentes_en_df)
    typer.echo("\n" + "="*60)
    typer.echo("📈 RESUMEN ESTADÍSTICO DEL PROCESAMIENTO")
    typer.echo("="*60)
    typer.echo(f"📤 Registros de salida (únicos): {df_final_seleccionado.height:,}")
    
    if 'cve_entidad' in df_final_seleccionado.columns:
        typer.echo("\n🗺️  Cobertura geográfica final:")
        n_entidades = df_final_seleccionado['cve_entidad'].n_unique()
        typer.echo(f"  • Entidades cubiertas: {n_entidades}/32")
        
        top_5 = df_final_seleccionado['cve_entidad'].value_counts().head(5)
        if not top_5.is_empty():
            typer.echo("\n  Top 5 entidades con más registros:")
            total_final = df_final_seleccionado.height
            for row in top_5.iter_rows(named=True):
                porcentaje = (row['count'] / total_final) * 100
                typer.echo(f"  • {row['cve_entidad']}: {row['count']:,} ({porcentaje:.1f}%)")
    typer.echo("="*60 + "\n")

    # --- Parte 5: Guardado ---
    typer.echo(f"➡️  Guardando archivo en: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final_seleccionado.write_csv(output_path)
    typer.secho(f"\n🎉 ¡Proceso completado! Archivo final con {df_final_seleccionado.height:,} registros únicos guardado.", fg=typer.colors.GREEN)
    typer.echo(f"⚡ Procesamiento ejecutado con {os.environ['POLARS_MAX_THREADS']} threads de Polars.")
    
if __name__ == "__main__":
    app()
```

### `src/geocodificador_ine_mvp/build_vector_store.py`

**Propósito:** Script para construir la base de datos vectorial `ChromaDB`. Lee los domicilios preprocesados, genera embeddings para cada uno usando un modelo `SentenceTransformer` y los almacena en colecciones separadas por entidad.

**Funciones Clave:**
- `build_vector_store(...)`: Comando de Typer que carga los datos, inicializa el modelo de embeddings y el cliente de ChromaDB, y luego itera por cada entidad para generar y almacenar los vectores en lotes.

**Contenido:**
```python
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
```

### `src/geocodificador_ine_mvp/search_models.py`

**Propósito:** Implementa la lógica de búsqueda híbrida.

**Clases y Funciones Clave:**
- `HybridSearcher`: Clase que encapsula la lógica de búsqueda.
  - `__init__`: Inicializa el cliente de ChromaDB, carga el modelo de embeddings y una caché para los índices léxicos.
  - `_get_lexical_index`: Construye (y cachea) un índice léxico BM25 para una entidad específica bajo demanda.
  - `search`: Realiza la búsqueda combinando resultados de la búsqueda vectorial (semántica) y la búsqueda por palabras clave (léxica) mediante una fusión de puntajes.
- `test_search`: Comando de Typer para probar la búsqueda desde la línea de comandos.

**Contenido:**
```python
# src/geocodificador_ine_mvp/search_models.py

import os
import typer
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.errors import NotFoundError
from rank_bm25 import BM25Okapi
import polars as pl
import numpy as np

load_dotenv()

app = typer.Typer()

# Se añade el diccionario para la conversión mapeo de claves numéricas de entidades a sus abreviaturas estándar
CLAVES_ENTIDADES = {
    1: 'AGS', 2: 'BC', 3: 'BCS', 4: 'CAM', 5: 'COA',
    6: 'COL', 7: 'CHS', 8: 'CHH', 9: 'CMX', 10: 'DGO',
    11: 'GTO', 12: 'GRO', 13: 'HGO', 14: 'JAL', 15: 'MEX',
    16: 'MCH', 17: 'MOR', 18: 'NAY', 19: 'NL', 20: 'OAX',
    21: 'PUE', 22: 'QRO', 23: 'ROO', 24: 'SLP', 25: 'SIN',
    26: 'SON', 27: 'TAB', 28: 'TMS', 29: 'TLX', 30: 'VER',
    31: 'YUC', 32: 'ZAC'
}

class HybridSearcher:
    def __init__(self, db_path: Path):
        """
        Inicializa el buscador. Se conecta a ChromaDB y carga el modelo de embeddings.
        El índice BM25 se creará dinámicamente en cada búsqueda.
        """
        typer.echo("➡️  Inicializando el Buscador Híbrido (con Fusión de Resultados)...")
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        model_name = os.getenv('EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
        typer.echo(f"➡️  Cargando el modelo de embeddings: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        
        # --- Caché para los índices BM25 por entidad ---
        self._lexical_indexes = {}
        
        typer.echo("✔️  Buscador Híbrido inicializado y listo.")
    
    def _get_lexical_index(self, cve_entidad: str) -> Optional[Dict[str, Any]]:
        """Carga o construye un índice léxico para una entidad específica."""
        if cve_entidad in self._lexical_indexes:
            return self._lexical_indexes[cve_entidad]

        typer.echo(f"  - Construyendo índice BM25 para '{cve_entidad}' por primera vez...")
        try:
            # Asumimos que los datos están en la misma carpeta que la DB vectorial
            corpus_path = Path(os.getenv('PROCESSED_DATA_PATH', 'data/output/domicilios_unificado_preprocesado.csv'))
            
            # Cargar solo los datos de la entidad necesaria
            df_entidad = pl.read_csv(
                corpus_path,
                schema_overrides={'domicilio_unico_id': pl.Utf8} 
            ).filter(pl.col("cve_entidad") == cve_entidad)
            
            corpus = df_entidad["domicilio_canonico"].to_list()
            corpus_ids = df_entidad["domicilio_unico_id"].to_list()
            
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            
            lexical_index = {"bm25": bm25, "corpus_ids": corpus_ids}
            self._lexical_indexes[cve_entidad] = lexical_index
            typer.echo(f"  - Índice BM25 para '{cve_entidad}' construido y cacheado.")
            return lexical_index
        except Exception as e:
            typer.secho(f"  - ❌ Error construyendo índice BM25 para '{cve_entidad}': {e}", fg=typer.colors.RED)
            return None

    def search(self, query: str, cve_entidad: str, top_k: int = 5, alpha: float = 0.6) -> List[Dict[str, any]]: # type: ignore
        """
        Realiza una búsqueda híbrida combinando la búsqueda vectorial y el re-ranking dinámico con BM25.
        """
        if not query or not cve_entidad:
            return []

        # 1. Búsqueda Vectorial para obtener candidatos
        collection_name = f"{os.getenv('CHROMA_COLLECTION_PREFIX', 'domicilios_ine_')}{cve_entidad.lower()}"
        try:
            collection = self.client.get_collection(name=collection_name)
        except (ValueError, NotFoundError):
            return []

        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
        vector_results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k, 
            include=["metadatas", "documents", "distances"]
        )
        
        # --- 2. Búsqueda Léxica ---
        lexical_index = self._get_lexical_index(cve_entidad)
        lexical_results = {}
        if lexical_index:
            tokenized_query = query.split(" ")
            doc_scores = lexical_index["bm25"].get_scores(tokenized_query)
            
            # Obtener los top_k resultados léxicos
            top_n_indices = np.argsort(doc_scores)[::-1][:top_k]
            for i in top_n_indices:
                if doc_scores[i] > 0:
                    doc_id = lexical_index["corpus_ids"][i]
                    lexical_results[doc_id] = doc_scores[i]

        if not vector_results['ids'] or not vector_results['ids'][0]:
            return []
        
        # --- 3. Fusión de Resultados ---
        fused_scores = {}
        # Procesar resultados vectoriales
        if vector_results['ids'] and vector_results['ids'][0]:
            for i, doc_id in enumerate(vector_results['ids'][0]):
                score = 1 - vector_results['distances'][0][i] # type: ignore
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (score * alpha)

        # Procesar resultados léxicos
        if lexical_results:
            max_lex_score = max(lexical_results.values()) if lexical_results else 1.0
            for doc_id, score in lexical_results.items():
                # Normalizar score léxico
                normalized_score = score / max_lex_score
                fused_scores[doc_id] = fused_scores.get(doc_id, 0) + (normalized_score * (1 - alpha))

        # --- 4. Ordenar y enriquecer los resultados finales ---
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]
        if not sorted_ids: 
            return []

        final_results_data = collection.get(ids=sorted_ids, include=["metadatas", "documents"])
        
        # Crear un mapa de ID a datos para una búsqueda rápida
        data_map = {
            final_results_data['ids'][i]: {
                "document": final_results_data['documents'][i], # type: ignore
                "metadata": final_results_data['metadatas'][i] or {} # type: ignore
            }
            for i in range(len(final_results_data['ids']))
        }
        
        final_results = []
        for doc_id in sorted_ids:
            if doc_id in data_map:
                data = data_map[doc_id]
                final_results.append({
                    "id_global_domicilio": doc_id,
                    "domicilio_canonico": data["document"],
                    "score": round(fused_scores[doc_id], 4),
                    "latitud": data["metadata"].get('latitud'),
                    "longitud": data["metadata"].get('longitud'),
                })

        return final_results

# --- Bloque de prueba para validación rápida ---
@app.command()
def test_search(
    query: str = typer.Argument(..., help="Dirección a buscar."),
    # --- CAMBIO: Se recibe un entero ---
    entidad_id: int = typer.Argument(..., help="ID numérico de la entidad (ej. 9 para CMX, 17 para MOR).")
):
    """
    Comando para probar la funcionalidad de búsqueda desde la terminal.
    """
    db_path = Path(os.getenv('VECTOR_STORE_PATH', 'vector_store/'))
    searcher = HybridSearcher(db_path=db_path)
    
    # Se convierte el ID numérico a la clave de texto ---
    cve_entidad = CLAVES_ENTIDADES.get(entidad_id)
    if not cve_entidad:
        typer.secho(f"Error: ID de entidad '{entidad_id}' no es válido.", fg=typer.colors.RED)
        raise typer.Exit()

    typer.echo(f"\nBuscando: '{query}' en la entidad '{cve_entidad}' (ID: {entidad_id})...")
    # La clase HybridSearcher sigue recibiendo el string, que es su formato interno
    search_results = searcher.search(query=query, cve_entidad=cve_entidad)
    
    typer.echo("\n--- Resultados ---")
    if not search_results:
        typer.echo("No se encontraron resultados.")
    else:
        for result in search_results:
            print(result)

if __name__ == "__main__":
    app()
```

### `src/geocodificador_ine_mvp/main.py`

**Propósito:** Punto de entrada de la API web construida con `FastAPI`.

**Componentes Clave:**
- `lifespan`: Un gestor de contexto asíncrono que carga el `HybridSearcher` al iniciar la aplicación para que esté listo para servir peticiones.
- `GeocodeRequest` / `GeocodeResponse`: Modelos de Pydantic para la validación de datos de entrada y salida.
- `@app.post("/geocodificar")`: El endpoint principal que recibe una dirección y un ID de entidad, la normaliza y utiliza el `HybridSearcher` para encontrar los domicilios correspondientes.
- `@app.get("/health")`: Un endpoint simple para verificar el estado del servicio.

**Contenido:**
```python
# src/geocodificador_ine_mvp/main.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

from .search_models import HybridSearcher
from .data_preprocessing import normalizar_texto_avanzado


load_dotenv()

# --- Diccionario de Mapeo: ID Numérico a Clave de 3 Letras ---
CLAVES_ENTIDADES = {
    1: 'AGS', 2: 'BC', 3: 'BCS', 4: 'CAM', 5: 'COA',
    6: 'COL', 7: 'CHS', 8: 'CHH', 9: 'CMX', 10: 'DGO',
    11: 'GTO', 12: 'GRO', 13: 'HGO', 14: 'JAL', 15: 'MEX',
    16: 'MCH', 17: 'MOR', 18: 'NAY', 19: 'NL', 20: 'OAX',
    21: 'PUE', 22: 'QRO', 23: 'ROO', 24: 'SLP', 25: 'SIN',
    26: 'SON', 27: 'TAB', 28: 'TMS', 29: 'TLX', 30: 'VER',
    31: 'YUC', 32: 'ZAC'
}

# --- Modelos de Datos con Pydantic ---
class GeocodeRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Dirección a geocodificar.")
    # --- Se recibe la entidad como número entero ---
    entidad_id: int = Field(..., description="ID numérico de la entidad (ej. 9 para CMX, 17 para MOR).")
    top_k: int = Field(5, gt=0, le=10, description="Número de resultados a devolver.")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Peso para la búsqueda semántica (0.0 a 1.0).")

class GeocodeResponse(BaseModel):
    results: List[Dict[str, Any]]

# --- 1. Contexto para el Modelo ---
# En lugar de un 'state' global, usaremos un diccionario que el 'lifespan' manejará.
model_context = {}

# --- 2. Implementación del Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al iniciar la aplicación
    print("Cargando modelo y preparando el buscador...")
    db_path = Path(os.getenv('VECTOR_STORE_PATH', 'vector_store/'))
    
    
    model_context["searcher"] = HybridSearcher(db_path=db_path)
    
    print("¡Servicio listo para recibir peticiones!")
    
    yield
    
    # Código que se ejecuta al apagar la aplicación
    model_context.clear()
    print("Servicio detenido y recursos liberados.")

# --- 3. Inicialización de FastAPI con el Lifespan ---
app = FastAPI(
    title="API de Geocodificación del INE (v2)",
    description="Servicio para geocodificar domicilios usando un modelo híbrido particionado por entidad.",
    version="0.2.0",
    lifespan=lifespan
)

# --- Endpoint de Geocodificación  ---
@app.post("/geocodificar", response_model=GeocodeResponse)
def geocode_address(request: GeocodeRequest):
    """
    Recibe una dirección y un ID de entidad, y devuelve los domicilios más probables.
    """
    searcher = model_context.get("searcher")
    if not searcher:
        raise HTTPException(status_code=503, detail="El servicio no está listo. Intente de nuevo en unos momentos.")
    
    # --- Se convierte el ID numérico a la clave de texto ---
    cve_entidad = CLAVES_ENTIDADES.get(request.entidad_id)
    if not cve_entidad:
        raise HTTPException(status_code=400, detail=f"El 'entidad_id' {request.entidad_id} no es válido.")

    try:
        # Se aplica la misma lógica de limpieza que a los datos en la base de datos.
        normalized_query = normalizar_texto_avanzado(request.query)
        
        # Se pasa la cve_entidad (string) al método de búsqueda
        search_results = searcher.search(
            query=normalized_query,
            cve_entidad=cve_entidad,
            top_k=request.top_k,
            alpha=request.alpha
        )
        return {"results": search_results}
    except Exception as e:
        print(f"Error interno durante la búsqueda: {e}")
        raise HTTPException(status_code=500, detail="Ocurrió un error interno durante la búsqueda.")

# --- Endpoint de Salud (sin cambios) ---
@app.get("/health")
def health_check():
    """Verifica que el servicio esté activo."""
    return {"status": "ok"}
```

---

## Scripts y Pruebas

### `scripts/evaluate_model.py`

**Propósito:** Script para evaluar el rendimiento del modelo de geocodificación. Compara los resultados de la API con un conjunto de datos de evaluación y calcula métricas como `Accuracy@1`, `MRR` y el error de distancia promedio.

**Funciones Clave:**
- `evaluate(eval_path)`: Comando de Typer que lee el archivo de evaluación, itera sobre cada caso de prueba, llama a la API de geocodificación y compara los resultados para generar un reporte final.

**Contenido:**
```python
# scripts/evaluate_model.py
import pandas as pd
import requests
import typer
from pathlib import Path
from haversine import haversine, Unit
import numpy as np

app = typer.Typer()
API_URL = "http://localhost:8000/geocodificar"

@app.command()
def evaluate(eval_path: Path = typer.Argument(..., help="Ruta al CSV con los datos de evaluación.")):
    """Evalúa el rendimiento del modelo de geocodificación con un reporte detallado."""
    if not eval_path.exists():
        typer.secho(f"❌ Error: No se encontró el archivo de evaluación en {eval_path}", fg=typer.colors.RED); raise typer.Exit(1)
            
    df_eval = pd.read_csv(eval_path)
    df_eval['id_global_domicilio_real'] = df_eval['entidad'].map({15: 'MEX', 9: 'CMX', 31: 'YUC', 7: 'CHS'}) + '-' + df_eval['id_domicilio'].astype(str)

    distances = []
    reciprocal_ranks = []

    typer.echo(f"\n▶️  Iniciando evaluación detallada de {len(df_eval)} casos de prueba...\n")
    
    for row in df_eval.itertuples():
        typer.echo(f"--- Caso de Prueba: '{row.query}' ---")
        expected_id = row.id_global_domicilio_real
        typer.echo(f"  - Esperado: {expected_id}")

        payload = {"query": row.query, "top_k": 5}
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            results = response.json()["results"]
        except requests.exceptions.RequestException as e:
            typer.secho(f"  - ❌ Error en API: {e}", fg=typer.colors.RED); continue

        if not results:
            typer.secho("  - ❌ Fallo: La API no devolvió resultados.", fg=typer.colors.RED)
            reciprocal_ranks.append(0)
            continue

        top_ids = [res["id_global_domicilio"] for res in results]
        top_result_id = top_ids[0]
        typer.echo(f"  - Obtenido:  {top_result_id} (Score: {results[0]['score']:.4f})")
        
        # --- Lógica de Reporte Detallado ---
        if top_result_id == expected_id:
            typer.secho("  - ✅ Acierto en @1", fg=typer.colors.GREEN)
        else:
            typer.secho("  - ❌ Fallo en @1", fg=typer.colors.RED)
            if expected_id in top_ids:
                rank = top_ids.index(expected_id) + 1
                typer.echo(f"    (La respuesta correcta estaba en la posición #{rank})")
            else:
                typer.echo("    (La respuesta correcta no apareció en el top 5)")
        
        # --- Cálculo de métricas (sin cambios) ---
        try:
            rank = top_ids.index(expected_id) + 1
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            reciprocal_ranks.append(0)

        if top_result_id == expected_id:
            pred_coords = (results[0].get('latitud'), results[0].get('longitud'))
            real_coords = (row.lat_real, row.lon_real)
            if pred_coords[0] is not None and pred_coords[1] is not None:
                dist = haversine(pred_coords, real_coords, unit=Unit.METERS)
                distances.append(dist)
                typer.echo(f"  - Distancia: {dist:.2f} metros")
        
        typer.echo("-" * 40)


    # Reporte final (sin cambios)
    accuracy_at_1 = sum(1 for r in reciprocal_ranks if r == 1) / len(reciprocal_ranks)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    avg_distance_error = np.mean(distances) if distances else float('inf')

    typer.echo("\n--- 📈 Reporte de Evaluación Final ---")
    typer.secho(f"Accuracy@1: {accuracy_at_1:.2%}", fg=typer.colors.GREEN)
    typer.secho(f"Mean Reciprocal Rank (MRR): {mrr:.4f}", fg=typer.colors.GREEN)
    typer.secho(f"Error Promedio de Distancia (en aciertos @1): {avg_distance_error:.2f} metros", fg=typer.colors.CYAN)
    typer.echo("------------------------------------")

if __name__ == "__main__":
    app()
```

### `tests/conftest.py`

**Propósito:** Archivo de configuración para `pytest`. Define fixtures que se utilizan en las pruebas.

**Fixtures Clave:**
- `setup_artifacts()`: Una fixture de sesión que se ejecuta una sola vez antes de todas las pruebas. Orquesta la ejecución de los scripts `data_preprocessing` y `build_vector_store` para asegurar que las pruebas se ejecuten sobre artefactos de datos consistentes y recién generados.

**Contenido:**
```python
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
```

### `tests/test_data_preprocessing.py`

**Propósito:** Pruebas unitarias para las funciones de normalización de texto.

**Funciones Clave:**
- `test_normalizar_texto_avanzado()`: Utiliza `pytest.mark.parametrize` para probar la función `normalizar_texto_avanzado` con una variedad de casos de entrada, asegurando que la limpieza y estandarización de direcciones sea correcta y robusta.

**Contenido:**
```python
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

```

### `tests/test_search_models.py`

**Propósito:** Pruebas de integración para el `HybridSearcher`.

**Funciones Clave:**
- `test_search_finds_correct_document_cmx()`: Valida que una búsqueda específica para la Ciudad de México devuelva el documento correcto.
- `test_search_handles_different_entity_mor()`: Asegura que el buscador puede manejar correctamente las búsquedas en diferentes particiones (entidades).
- `test_tuning_alpha_improves_ranking()`: Verifica que el ajuste del parámetro `alpha` (que balancea la búsqueda semántica y léxica) tiene el efecto deseado en los resultados.
- `test_search_handles_non_existent_collection()`: Comprueba que el sistema maneja de forma segura las consultas a entidades que no existen.

**Contenido:**
```python
# tests/test_search_models.py
import pytest
from pathlib import Path
import os

from geocodificador_ine_mvp.search_models import HybridSearcher # type: ignore


@pytest.fixture(scope="module")
def searcher():
    """Inicializa el HybridSearcher una vez por módulo de prueba.
    Apunta a los artefactos generados por el workflow."""
    db_path = Path(os.getenv('VECTOR_STORE_PATH', 'vector_store/'))
    return HybridSearcher(db_path=db_path)

def test_search_finds_correct_document_cmx(searcher):
    """Prueba una consulta específica para CMX y valida que el resultado correcto esté en primer lugar."""
    query = "carretera san mateo santa rosa 49 b colonia san mateo tlaltenango"
    # El ID correcto de nuestro archivo de prueba
    expected_id = "3862935497666623127" 

    results = searcher.search(query=query, cve_entidad="CMX", top_k=1, alpha=0.5)

    assert len(results) > 0, "La búsqueda no debería devolver una lista vacía"
    assert results[0]["id_global_domicilio"] == expected_id
    assert results[0]["score"] > 0.4, "El score de confianza debería ser alto"
    
def test_search_handles_different_entity_mor(searcher):
    """
    Prueba una consulta específica para MOR para asegurar que la selección de colección funciona.
    """
    query = "privada ingenio san ignacio 2"
    expected_id = "5145817463627586555" # ID de prueba para Morelos

    results = searcher.search(query=query, cve_entidad="MOR", top_k=1, alpha=0.5)
    
    assert len(results) > 0
    assert results[0]["id_global_domicilio"] == expected_id

def test_tuning_alpha_improves_ranking(searcher):
    """
    Prueba el caso de "clavel" y valida que ajustar el alpha mejora el ranking.
    """
    query = "calle plan de san luis manzana 108 lote 52 colonia san lorenzo la cebada 16035 xochimilco ciudad de mexico"
    
    # Con un alpha alto (semántico), esperamos un resultado.
    expected_id_semantic = "6649470028573682094"
    results_semantic_heavy = searcher.search(query=query, cve_entidad="CMX", top_k=1, alpha=0.7)
    assert len(results_semantic_heavy) > 0
    assert results_semantic_heavy[0]["id_global_domicilio"] == expected_id_semantic

    # Con un alpha bajo (léxico), esperamos el resultado que es textualmente más preciso.
    expected_id_lexical = "6649470028573682094"
    results_lexical_heavy = searcher.search(query=query, cve_entidad="CMX", top_k=1, alpha=0.3)
    assert len(results_lexical_heavy) > 0
    assert results_lexical_heavy[0]["id_global_domicilio"] == expected_id_lexical

def test_search_handles_non_existent_collection(searcher):
    """
    Prueba que el buscador devuelve una lista vacía si se le pide una entidad que no existe.
    """
    query = "cualquier cosa"
    results = searcher.search(query=query, cve_entidad="XYZ", top_k=5) # XYZ no existe
    assert results == []
```

---

## Utilidades

### `utils/logger.py`

**Propósito:** Configura un logger centralizado para la aplicación que formatea los logs en formato JSON, facilitando su recolección y análisis en sistemas de monitoreo.

**Funciones Clave:**
- `setup_logger()`: Crea y configura una instancia de logger con un `jsonlogger`.

**Contenido:**
```python
# src/geocodificador_ine_mvp/utils/logger.py
import logging
from pythonjsonlogger import jsonlogger

def setup_logger():
    logger = logging.getLogger("geocodificador")
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter() # type: ignore
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
```
