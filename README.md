# BeagleMind API (RAG)

FastAPI service for Retrieval‑Augmented Generation backed by Milvus vector store and offline ONNX models.

---
## Project Structure
```
rag_api/
├── app/
│   ├── models/                # Pydantic schemas
│   ├── routes/                # FastAPI routers (retrieval, ingestion)
│   ├── services/              # Core logic (retrieval, ingestion)
│   ├── scripts/               # Ingestion scripts
│   └── config.py              # Config helpers
├── onnx/                      # Offline embedding & reranker models/tokenizer
├── docker-compose.yml         # Full stack (Milvus + API)
├── Dockerfile                 # API-only image
├── requirements.txt
├── main.py                    # FastAPI app
└── README.md
```

---
## Prerequisites
Choose ONE of these paths:
1. Easiest: Docker + Docker Compose (recommended) — Everything (Milvus + etcd + MinIO + API) starts together.
2. Manual: Run Milvus separately (script or your own compose) then run the API (local Python or API container).

Offline models are already provided in `onnx/`; no internet required for embeddings / reranking.

---
## Quick Start (ONE COMMAND: Vector Store + API)
Bring up the full stack using the included `docker-compose.yml`:
```bash
docker compose up --build -d
```
Services exposed:
| Service | Port(s) | Description |
|---------|---------|-------------|
| API (`rag-api`) | 8000 | FastAPI endpoints |
| Milvus gRPC | 19530 | Vector DB client port |
| Milvus Metrics | 9091 | Health / metrics |
| MinIO S3 | 9000 | Object storage backend |
| MinIO Console | 9001 | Web UI for MinIO |

Persistent data: `./volumes/` (etcd, minio, milvus). ONNX models are mounted read‑only.

Health check:
```bash
docker compose ps
curl -s http://localhost:8000/health
```

Rebuild API after code changes:
```bash
docker compose build rag-api
docker compose up -d rag-api
```

Tear down (keep data):
```bash
docker compose down
```
---
## Manual Alternative
### Start Milvus (standalone script)
```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

### OR Build Only the API Image
```bash
docker build -t beaglemind-api .
docker run -d --name beaglemind-api \
  -p 8000:8000 \
  -v $(pwd)/onnx:/app/onnx:ro \
  beaglemind-api
```

---
## User Guide (End‑to‑End)
### 1. Health Check
GET `/health`
Expected:
```json
{ "status": "healthy" }
```

### 2. Ingest BeagleBoard Documentation Repository
Repo: `https://github.com/beagleboard/docs.beagleboard.io` (branch `main`).
POST `/api/ingest-data`
Request body:
```json
{
  "collection_name": "beaglemind_col",
  "github_url": "https://github.com/beagleboard/docs.beagleboard.io",
  "branch": "main"
}
```
Sample success:
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
* Re‑ingesting appends new/changed content (no hash dedupe yet).
* Progress log tags: `[FETCH]`, `[PROCESS]`, `[EMBEDDINGS]`, `[STORAGE]`, `[SERVICE]`, `[ROUTER]`.
* Tail logs: `tail -f app.log` or `docker compose logs -f rag-api`.

### 3. Check Ingestion Service Status
GET `/api/ingest-data/status`
```json
{
  "success": true,
  "message": "GitHub ingestion service is running",
  "active_collections": 1
}
```

### 4. Retrieve Documents
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
  "documents": [["...chunk text...", "...chunk text..."]],
  "metadatas": [[{ "score": 0.95, "distance": 0.05, "file_name": "getting-started.md" }]],
  "distances": [[0.05, 0.08]],
  "total_found": 120,
  "filtered_results": 2
}
```

### 5. Swagger UI
Navigate: `http://localhost:8000/docs`

---
## Raw Endpoint Reference
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health probe |
| POST | /api/ingest-data | Ingest a GitHub repo into a Milvus collection |
| GET | /api/ingest-data/status | Ingestion service status |
| POST | /api/retrieve | Semantic search with optional rerank |

---
## Models Used
- Embeddings: BGE Base EN v1.5 (ONNX) — tokenizer loaded from `./onnx` (offline)
- Reranker: Cross-Encoder MS MARCO MiniLM-L-6-v2 (ONNX) — tokenizer loaded from `./onnx` (offline)

## Features
- Semantic document search (BAAI BGE embeddings)
- Cross-encoder reranking for higher relevance
- Rich metadata (file path, type, language, code flag, repo name, quality scores)
- IVF_FLAT vector index (L2)
- Offline model operation (no HuggingFace network calls)

## API Docs
Swagger UI: `http://localhost:8000/docs`

## Troubleshooting
| Issue | Likely Cause | Resolution |
|-------|--------------|------------|
| API can't connect to Milvus | Milvus still initializing (first boot 60–90s) | Wait; `docker compose logs -f standalone` |
| Connection refused outside compose | Wrong MILVUS_HOST | `export MILVUS_HOST=localhost MILVUS_PORT=19530` |
| Empty retrieval results | Ingestion incomplete / wrong collection | Verify ingestion response & name |
| Slow first query | Segment / index warmup | Subsequent queries faster |
| No ingestion logs in terminal | Logs written to file | `tail -f app.log` |
| Offline model load fails | Missing `onnx/` dir or mount | Ensure directory present & mounted read‑only |
| Volume permission errors | Host FS perms | `chown -R $(id -u):$(id -g) volumes/` |
| Code changes not reflected | Old container image running | `docker compose build rag-api && docker compose up -d rag-api` |

Collect logs:
```bash
docker compose logs --tail=200 rag-api standalone
```

---
## Next Ideas
- Add hash-based deduplication on ingestion
- Add export/import endpoints for collection portability
- Add unit tests for chunk & embedding pipeline
