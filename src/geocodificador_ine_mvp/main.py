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

# Carga variables de .env solo si estamos en modo de desarrollo
if os.getenv('ENV_MODE') == 'development':
    print("Cargando variables de entorno desde .env...")
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