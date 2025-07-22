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