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

Run Milvus locally with Docker Compose:

```yaml
# docker-compose.milvus.yml
version: "3.8"
services:
    etcd:
        image: quay.io/coreos/etcd:v3.5.18
        environment:
            - ETCD_AUTO_COMPACTION_MODE=revision
            - ETCD_AUTO_COMPACTION_RETENTION=1000
            - ETCD_QUOTA_BACKEND_BYTES=4294967296
            - ETCD_SNAPSHOT_COUNT=50000
        volumes:
            - ./volumes/etcd:/etcd
        command: ["etcd", "-advertise-client-urls", "http://etcd:2379", "-listen-client-urls", "http://0.0.0.0:2379", "-listen-peer-urls", "http://0.0.0.0:2380"]

    minio:
        image: minio/minio:RELEASE.2024-05-28T17-19-04Z
        environment:
            - MINIO_ACCESS_KEY=minioadmin
            - MINIO_SECRET_KEY=minioadmin
        volumes:
            - ./volumes/minio:/minio_data
        command: ["server", "/minio_data"]
        ports:
            - "9000:9000"
            - "9001:9001"

    milvus:
        image: milvusdb/milvus:v2.6.0
        depends_on: [etcd, minio]
        environment:
            - ETCD_ENDPOINTS=etcd:2379
            - MINIO_ADDRESS=minio:9000
        ports:
            - "19530:19530"
            - "9091:9091"
        volumes:
            - ./volumes/milvus:/var/lib/milvus
        command: ["milvus", "run", "standalone"]
```

Bring up Milvus:

```bash
docker compose -f docker-compose.milvus.yml up -d
```

Health check:

```bash
curl -s http://localhost:9091/healthz
```

## Installation (local)

### Run with Docker

Milvus must be started separately (see above). This container only runs the API.

```bash
docker build -t beaglemind-api .
docker run --rm -p 8000:8000 \
    -e MILVUS_HOST=host.docker.internal \
    -e MILVUS_PORT=19530 \
    beaglemind-api
```

If running Milvus on Linux host, use `--network host` or set `MILVUS_HOST=localhost` and add `--add-host=host.docker.internal:host-gateway` if needed.

## API Endpoints

### 1. Initialize Retrieval System

**POST** `/api/v1/initialize`

Initialize the Milvus collection and embedding models.

**Request Body:**
```json
{
    "collection_name": "beaglemind_col"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Retrieval system initialized successfully",
    "collection_name": "beaglemind_col"
}
```

### 1) Retrieve Documents

POST `/api/retrieve`

Search for documents using semantic similarity.

**Request Body:**
```json
{
    "query": "machine learning algorithms",
    "n_results": 10,
    "include_metadata": true,
    "rerank": true
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
    "github_url": "https://github.com/owner/repo",
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

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
