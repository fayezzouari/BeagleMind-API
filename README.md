# BeagleMind RAG API

A FastAPI-based system for document ingestion and semantic search using ONNX models.

## Features

- **GitHub Repository Ingestion**: Directly ingest GitHub repositories with metadata extraction
- **Forum Data Ingestion**: Process forum JSON data into searchable format
- **Semantic Search**: ONNX-powered semantic search with reranking
- **Background Processing**: Asynchronous ingestion with status tracking

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place ONNX models in `onnx/` directory:
   - `onnx/model.onnx` (BGE embedding model)
   - `onnx/cross_encoder.onnx` (Reranker model)

3. Configure environment variables:
```bash
export MILVUS_HOST=localhost
export MILVUS_PORT=19530
# Add other Milvus config as needed
```

4. Start the API:
```bash
python app/main.py
```

## API Endpoints

### Search
```bash
POST /api/v1/search
{
    "query": "How to configure BeagleBone GPIO?",
    "collection_name": "beaglemind_docs",
    "n_results": 10,
    "rerank": true
}
```

### GitHub Ingestion
```bash
POST /api/v1/ingest/github
{
    "repo_url": "https://github.com/beagleboard/docs.beagleboard.io",
    "branch": "main",
    "collection_name": "beaglemind_docs"
}
```

### Forum Ingestion
```bash
POST /api/v1/ingest/forum
{
    "json_path": "/path/to/forum_data.json",
    "collection_name": "beaglemind_docs"
}
```

### Check Status
```bash
GET /api/v1/ingest/status/{task_id}
```

## Command Line Usage

### GitHub Ingestion
```bash
python app/scripts/github_ingestor.py https://github.com/beagleboard/docs.beagleboard.io \
    --collection beaglemind_docs \
    --branch main
```

### Forum Ingestion
```bash
python app/scripts/forum_ingestor.py forum_data.json \
    --collection beaglemind_docs
```

## Architecture

- **FastAPI**: Web framework for the API
- **ONNX Runtime**: Efficient model inference
- **Milvus**: Vector database for semantic search
- **Background Tasks**: Asynchronous processing for large ingestions
