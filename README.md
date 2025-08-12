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

## API Endpoints

### 1) Retrieve Documents

POST `/api/retrieve`

Search for documents using semantic similarity.

**Request Body:**
```json
{
    "query": "Blinking LEDs",
    "n_results": 10,
    "include_metadata": true,
    "rerank": true,
    "collection_name":"beaglemind_col"
}
```

**Response:**
```json
{
    "documents": [["Document text 1", "Document text 2", ...]],
    "metadatas": [[
        {
            "score": 0.95,
            "distance": 0.05,
            "file_name": "example.txt",
            "file_path": "/path/to/file",
            "chunk_index": 0,
            "language": "python",
            "has_code": true
        }
    ]],
    "distances": [[0.05, 0.08, ...]],
    "total_found": 100,
    "filtered_results": 10
}
```

### 2) Ingest GitHub Repository into a Collection

POST `/api/ingest-data`

Body:
```json
{
    "collection_name": "beaglemind_col",
    "github_url": "https://github.com/beagleboard/docs.beagleboard.io",
    "branch": "main"
}
```

Response:
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
- If the collection exists, new content is appended.
- Progress logs show in the API console.

### 3) Ingestion Service Status

GET `/api/ingest-data/status`

Response:
```json
{
    "success": true,
    "message": "GitHub ingestion service is running",
    "active_collections": 1
}
```

## Models Used

- Embeddings: BGE Base EN v1.5 (ONNX) — tokenizer loaded from `./onnx` (offline)
- Reranker: Cross-Encoder MS MARCO MiniLM-L-6-v2 (ONNX) — tokenizer loaded from `./onnx` (offline)

## Features

- Semantic document search using BAAI BGE embeddings
- CrossEncoder-based reranking for improved relevance
- Comprehensive metadata support
- Vector similarity search (L2/IVF_FLAT)
- Configurable result filtering and ranking

## Troubleshooting

- "cannot create index on non-exist field": ensure indexes are created only on existing fields. Current schema fields are: id, document, embedding, file_name, file_path, file_type, source_link, chunk_index, language, has_code, repo_name, content_quality_score, semantic_density_score, information_value_score.
- SSL errors during Docker builds: pre-download wheels on host or build with host network. See Dockerfile comments and consider a local wheelhouse.
- No logs during ingestion: the service runs blocking work in a thread pool and logs to console and app.log. Tail logs with `tail -f app.log`.

## API Docs

Swagger UI: `http://localhost:8000/docs`

## Health Check

**GET** `/health`

Returns API health status:
```json
{
    "status": "healthy"
}
```

