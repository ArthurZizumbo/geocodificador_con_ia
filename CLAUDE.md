# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a geocoding MVP for INE (Instituto Nacional de Estadística) that uses hybrid semantic search to match addresses with high precision. The system combines vector embeddings with lexical search (BM25) to find the best matches for address queries.

## Key Technologies

- **Python 3.9+** with Poetry for dependency management
- **FastAPI** for the REST API server
- **ChromaDB** for vector storage and semantic search
- **Sentence Transformers** for multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
- **BM25** for lexical search ranking
- **Typer** for CLI applications
- **Pytest** for testing

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry
poetry install

# Or using pip with requirements.txt
pip install -r requirements.txt
```

### Running the API Server
```bash
# Start the FastAPI server
uvicorn src.geocodificador_ine_mvp.main:app --host 0.0.0.0 --port 8000 --reload
```

### Data Processing Pipeline
```bash
# 1. Preprocess and normalize addresses
python -m src.geocodificador_ine_mvp.data_preprocessing data/raw_addresses.csv data/padron_canonico_deduplicado.csv

# 2. Build vector store from processed data
python -m src.geocodificador_ine_mvp.build_vector_store data/padron_canonico_deduplicado.csv vector_store/

# 3. Evaluate model performance
python scripts/evaluate_model.py data/eval_data.csv
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_search_models.py
```

### Code Quality
```bash
# Format code with Black
black src/ tests/ scripts/

# Run linting
black --check src/ tests/ scripts/
```

### Docker
```bash
# Build Docker image
docker build -t geocodificador-ine .

# Run container
docker run -p 8000:8000 geocodificador-ine
```

## Architecture

### Core Components

1. **HybridSearcher** (`src/geocodificador_ine_mvp/search_models.py`):
   - Combines ChromaDB vector search with BM25 lexical search
   - Loads `paraphrase-multilingual-MiniLM-L12-v2` model for embeddings
   - Implements weighted fusion of semantic and lexical scores

2. **Data Preprocessing** (`src/geocodificador_ine_mvp/data_preprocessing.py`):
   - Normalizes Mexican addresses with domain-specific rules
   - Handles abbreviations, accents, and standardizes formats
   - Creates canonical address representations
   - Deduplicates similar addresses

3. **Vector Store Builder** (`src/geocodificador_ine_mvp/build_vector_store.py`):
   - Generates embeddings for processed addresses
   - Stores in ChromaDB with metadata
   - Processes data in batches for memory efficiency

4. **FastAPI Server** (`src/geocodificador_ine_mvp/main.py`):
   - REST API with `/geocodificar` endpoint
   - Uses application lifespan for model loading
   - Returns enriched results with coordinates

### Data Flow

1. Raw address data → Data preprocessing → Canonical addresses
2. Canonical addresses → Vector store builder → ChromaDB embeddings
3. Query → HybridSearcher → Vector + BM25 search → Ranked results
4. Results → API response with coordinates and metadata

### Key Files Structure

- `src/geocodificador_ine_mvp/`: Main Python package
- `data/`: CSV files with address data
- `vector_store/`: ChromaDB persistent storage
- `tests/`: Unit tests with pytest
- `scripts/`: Evaluation and utility scripts

## Important Notes

- The system is designed for Mexican addresses and includes Mexico-specific normalization rules
- Address normalization includes handling of Mexican state codes (CVE_ENTIDAD mapping)
- The hybrid search combines semantic similarity (alpha) and lexical matching (1-alpha)
- Vector store must be built before starting the API server
- All CLI tools use Typer for consistent command-line interface

## Testing Strategy

- Unit tests for normalization functions in `test_data_preprocessing.py`
- Integration tests for search functionality in `test_search_models.py`
- Model evaluation script tests API endpoints with real data
- Tests require pre-built vector store and processed data files