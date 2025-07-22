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