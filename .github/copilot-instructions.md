# Copilot Coding Agent Instructions for endo-ecosystem-poc

## Project Overview
This repository is a Proof of Concept (PoC) for the Bold Wave Productions Endo Ecosystem. It aggregates, normalizes, and processes biomedical and regulatory data from multiple sources, supporting RAG (Retrieval-Augmented Generation) workflows and vector search.

## Architecture & Key Components
- **src/**: Main Python source code.
  - **connectors/**: Data ingestion modules for sources (e.g., `crossref.py`, `pubmed.py`, `ctgov.py`, etc.).
  - **pipelines/**: Data processing, embedding, retrieval, and RAG logic (e.g., `embeddings.py`, `rag_ask.py`).
  - **common/**: Shared utilities (e.g., normalization, storage, logging).
  - **ui/**: Minimal UI logic (e.g., `rag_panel.py`).
- **data/**: Stores processed data, logs, and vector stores (e.g., `vector_store.faiss`).
- **raw/**: Hierarchically organized raw data dumps by source and date.
- **tests/**: Pytest-based unit tests for ingestion and pipelines.

## Developer Workflows
- **Environment Setup**: Use Python virtual environments. Activate with `.venv\Scripts\activate` (Windows).
- **Dependencies**: Install via `requirements.txt`. Use `pip install -r requirements.txt`.
- **Testing**: Run all tests with `pytest`. Example: `pytest tests/`.
- **Health Check**: Validate pipeline health with `python -m src.pipelines.health_check`.
- **Linting**: Use `ruff` for linting. Configured via `ruff.toml`.
- **Configuration**: Environment variables (see `env.example.copy`) and `config.py`.

## Patterns & Conventions
- **Data Ingestion**: Each connector module implements a `pull` or `fetch` function for its source. See `src/connectors/`.
- **Normalization**: Use `common/normalize.py` for standardizing data formats.
- **Logging**: RAG logs are written to `data/rag_logs.jsonl`.
- **Vector Search**: FAISS vector store at `data/vector_store.faiss`.
- **Testing**: Tests are organized by source/component, mirroring `src/` structure.
- **No Web UI**: UI is minimal and script-based; no frontend framework.

## Integration Points
- **External APIs**: Connectors integrate with public biomedical APIs (e.g., PubMed, Crossref, NIH Reporter).
- **Azure OpenAI**: Uses environment variables for API keys and endpoints.
- **Docker**: Dockerfile provided for containerization; not required for local dev.

## Examples
- To ingest PubMed data: `python -m src.connectors.pubmed`
- To run RAG query: `python -m src.pipelines.rag_ask --query "your question"`
- To check health: `python -m src.pipelines.health_check`

## Key Files
- `src/main.py`: Entry point for orchestrating workflows.
- `src/config.py`: Central config management.
- `src/pipelines/rag_ask.py`: RAG query logic.
- `src/common/normalize.py`: Data normalization utilities.
- `requirements.txt`: Python dependencies.

## Contributing
- Follow existing module patterns for new connectors or pipelines.
- Add tests in `tests/` matching the structure of new code.
- Document new environment variables in `env.example.copy`.

---
For questions or missing conventions, review `README.md` and source files in `src/`.
