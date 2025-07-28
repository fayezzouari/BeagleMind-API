# Information Retrieval API

A FastAPI-based information retrieval system using Milvus vector database and sentence transformers for semantic search.

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

## Installation

1. Create and activate virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Set environment variables for Milvus connection:

```bash
export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
export MILVUS_USER=""
export MILVUS_PASSWORD=""
export MILVUS_TOKEN=""
export MILVUS_URI=""  # Use this for cloud instances
```

## Running the API

### Development Server
```bash
python run.py
```

### Production Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

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

### 2. Retrieve Documents

**POST** `/api/v1/retrieve`

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

## Models Used

- **Embedding Model**: `BAAI/bge-base-en-v1.5` (BGE base English v1.5)
- **Reranker Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (CrossEncoder for reranking)

## Features

- Semantic document search using BAAI BGE embeddings
- CrossEncoder-based reranking for improved relevance
- Comprehensive metadata support
- Vector similarity search with COSINE metric
- Configurable result filtering and ranking

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
