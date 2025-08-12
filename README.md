# BeagleMind API (RAG)

FastAPI service for Retrieval-Augmented Generation workflows backed by Milvus vector store and ONNX models (offline-friendly).

## Project Structure

```
rag_api/
├── app/
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models for request/response
│   ├── routes/
│   │   ├── __init__.py
│   │   └── retrieval.py        # API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   └── retrieval_service.py # Core retrieval logic
│   ├── __init__.py
│   └── config.py               # Configuration settings
├── main.py                     # FastAPI application entry point
├── run.py                      # Development server runner
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Prerequisites

- Python 3.12 (for local runs) or Docker (for containerized runs)
- Milvus Standalone (run via Docker Compose)
- ONNX models and local tokenizers already included in `onnx/` (no internet required)

## Milvus Setup (required)

Run Milvus locally:


```bash
# Download the installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

# Start the Docker container
bash standalone_embed.sh start
```

## Installation (local)

### Run with Docker

Milvus must be started separately (see above). This container only runs the API.

```bash
docker build -t beaglemind-api .
docker run -d --name beaglemind-api \
  -p 8000:8000 \
  -e MILVUS_HOST=host.docker.internal \
  -e MILVUS_PORT=19530 \
  beaglemind-api
```

If running Milvus on Linux host, use `--network host` or set `MILVUS_HOST=localhost` and add `--add-host=host.docker.internal:host-gateway` if needed.

## User Guide: End‑to‑End Workflow

This guide walks you through checking the API health, ingesting the official BeagleBoard documentation repository, monitoring ingestion status, and finally performing a semantic retrieval query.

### 1. Health Check (ensure API is running)

GET `/health`

Expected response:
```json
{ "status": "healthy" }
```

### 2. Ingest the BeagleBoard Documentation Repository

Use the GitHub repo: `https://github.com/beagleboard/docs.beagleboard.io` (branch `main`).

POST `/api/ingest-data`

Request body:
```json
{
    "collection_name": "beaglemind_col",
    "github_url": "https://github.com/beagleboard/docs.beagleboard.io",
    "branch": "main"
}
```

Successful ingestion response (fields will vary):
```json
{
    "success": true,
    "message": "Successfully ingested repository into collection 'beaglemind_col'",
    "stats": {
        "files_processed": 150,
        "chunks_generated": 1200,
        "files_with_code": 80,
        "avg_quality_score": 0.78,
        "total_time": 45.2
    }
}
```

Notes:
* Re‑ingesting the same repo appends new/changed content (no automatic dedupe by hash—add if needed).
* Progress is logged with tags like `[FETCH]`, `[PROCESS]`, `[EMBEDDINGS]`, `[STORAGE]`, `[SERVICE]`, `[ROUTER]`.
* Tail logs during ingestion:
    ```bash
    tail -f app.log
    ```

### 3. Check Ingestion Service Status

GET `/api/ingest-data/status`

Sample response:
```json
{
    "success": true,
    "message": "GitHub ingestion service is running",
    "active_collections": 1
}
```

### 4. Retrieve Documents (Semantic Search + Optional Rerank)

POST `/api/retrieve`

Request body:
```json
{
    "query": "Blink an LED on BeagleBone",
    "n_results": 5,
    "include_metadata": true,
    "rerank": true,
    "collection_name": "beaglemind_col"
}
```

Sample response (truncated):
```json
{
    "documents": [["...chunk text...", "...chunk text..." ]],
    "metadatas": [[
        {
            "score": 0.95,
            "distance": 0.05,
            "file_name": "getting-started.md",
            "file_path": "docs/getting-started.md",
            "chunk_index": 0,
            "language": "markdown",
            "has_code": false,
            "repo_name": "docs.beagleboard.io"
        }
    ]],
    "distances": [[0.05, 0.08]],
    "total_found": 120,
    "filtered_results": 2
}
```

### 5. Swagger UI

Browse interactive docs at: `http://localhost:8000/docs`

---

## Raw Endpoint Reference

| Method | Path                    | Description                                   |
|--------|-------------------------|-----------------------------------------------|
| GET    | /health                 | Health probe                                  |
| POST   | /api/ingest-data        | Ingest a GitHub repo into a Milvus collection |
| GET    | /api/ingest-data/status | Ingestion service status                      |
| POST   | /api/retrieve           | Semantic search + (optional) rerank           |

---

## Models Used

- Embeddings: BGE Base EN v1.5 (ONNX) — tokenizer loaded from `./onnx` (offline)
- Reranker: Cross-Encoder MS MARCO MiniLM-L-6-v2 (ONNX) — tokenizer loaded from `./onnx` (offline)

## Features

- Semantic document search using BAAI BGE embeddings
- CrossEncoder-based reranking for improved relevance
- Comprehensive metadata support
- Vector similarity search (L2/IVF_FLAT)
- Configurable result filtering and ranking

## API Docs

Swagger UI: `http://localhost:8000/docs`
