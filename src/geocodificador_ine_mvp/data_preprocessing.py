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