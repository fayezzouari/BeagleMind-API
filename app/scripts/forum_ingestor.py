# filepath: /home/fayez/gsoc/rag_poc/src/forum_ingest.py
import os
import json
import re
import uuid
import logging
from typing import List, Dict, Any
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
from datetime import datetime
import dotenv

dotenv.load_dotenv()
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _encode_text(text: str, tokenizer, session) -> List[float]:
    """Encode text using ONNX embedding model"""
    inputs = tokenizer(
        text, 
        return_tensors="np", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    if "token_type_ids" in inputs:
        onnx_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
    
    # Get outputs from the ONNX model
    outputs = session.run(None, onnx_inputs)
    
    # Use mean pooling over token embeddings
    embedding = outputs[0][0].mean(axis=0)
    
    # Normalize the embedding
    norm = np.linalg.norm(embedding)
    normalized_embedding = (embedding / norm) if norm != 0 else embedding
    
    return normalized_embedding.tolist()

def semantic_chunk_post(content: str, language: str = "text", chunk_size: int = 512) -> List[str]:
    """
    Chunk forum post content using RecursiveCharacterTextSplitter.
    More reliable than semantic chunking for forum posts.
    """
    # Use RecursiveCharacterTextSplitter for reliable chunking of forum posts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,  # Small overlap to maintain context
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_text(content)
    return [chunk for chunk in chunks if len(chunk.strip()) > 10]

def connect_milvus():
    connect_kwargs = {'alias': "default", 'timeout': 30}
    if MILVUS_URI:
        connect_kwargs['uri'] = MILVUS_URI
    else:
        connect_kwargs['host'] = MILVUS_HOST
        connect_kwargs['port'] = MILVUS_PORT
    if MILVUS_USER:
        connect_kwargs['user'] = MILVUS_USER
    if MILVUS_PASSWORD:
        connect_kwargs['password'] = MILVUS_PASSWORD
    if MILVUS_TOKEN:
        connect_kwargs['token'] = MILVUS_TOKEN
    connections.connect(**connect_kwargs)

def get_or_create_collection(collection_name: str, embedding_dim: int) -> Collection:
    # Use the same 14-field schema as the retrieval service
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="source_link", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="has_code", dtype=DataType.BOOL),
        FieldSchema(name="repo_name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="content_quality_score", dtype=DataType.FLOAT),
        FieldSchema(name="semantic_density_score", dtype=DataType.FLOAT),
        FieldSchema(name="information_value_score", dtype=DataType.FLOAT),
    ]
    
    schema = CollectionSchema(fields, "Forum content with semantic chunking")
    if utility.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists")
        col = Collection(collection_name)
    else:
        logger.info(f"Creating collection '{collection_name}'")
        col = Collection(collection_name, schema)
        index_params = {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        col.create_index("embedding", index_params)
    col.load()
    return col

def ingest_forum_json(json_path: str, collection_name: str = "beaglemind_docs", model_name: str = "BAAI/bge-base-en-v1.5"):
    connect_milvus()
    
    # Initialize ONNX embedding model
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
    session = ort.InferenceSession("onnx/model.onnx")
    
    # Get embedding dimension
    sample_embedding = _encode_text("test", tokenizer, session)
    embedding_dim = len(sample_embedding)
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    collection = get_or_create_collection(collection_name, embedding_dim)
    
    with open(json_path, 'r') as f:
        threads = json.load(f)
    
    # Prepare data for all 35 fields
    chunk_data = []
    for thread in threads:
        thread_link = thread.get("url", "")
        thread_name = thread.get("thread_name", "")
        content = thread.get("content", "")
        
        # Split by 'Post #' (robust for forum dumps)
        post_splits = [p for p in re.split(r'Post #\d+ by [^:]+:', content) if p.strip()]
        for post_idx, post_text in enumerate(post_splits):
            post_text = post_text.strip()
            if not post_text or len(post_text) < 20:
                continue
                
            # Semantic chunking
            chunks = semantic_chunk_post(post_text)
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 20:
                    continue
                    
                # Only create metadata for the 14 fields
                chunk_data.append({
                    'id': str(uuid.uuid4()),
                    'document': chunk[:65535],
                    'file_name': f"forum_post_{post_idx}",
                    'file_path': f"forum/{thread_name}",
                    'file_type': '.forum',
                    'source_link': thread_link[:2000],
                    'chunk_index': chunk_idx,
                    'language': 'text',
                    'has_code': False,
                    'repo_name': 'beagleboard_forum',
                    'content_quality_score': 0.7,
                    'semantic_density_score': 0.6,
                    'information_value_score': 0.8
                })
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(chunk_data)} chunks...")
    documents = [item['document'] for item in chunk_data]
    
    embeddings = []
    for i, doc in enumerate(documents):
        embedding = _encode_text(doc, tokenizer, session)
        embeddings.append(embedding)
        if i % 100 == 0:
            logger.info(f"Generated embeddings: {i}/{len(documents)}")
    
    embeddings = np.array(embeddings)
    
    # Insert in batches with 14 fields
    batch_size = 100
    for i in range(0, len(chunk_data), batch_size):
        batch_end = min(i + batch_size, len(chunk_data))
        batch_data = chunk_data[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]
        
        # Prepare entities for 14 fields in correct order
        entities = [
            [item['id'] for item in batch_data],
            [item['document'] for item in batch_data],
            batch_embeddings.tolist(),
            [item['file_name'] for item in batch_data],
            [item['file_path'] for item in batch_data],
            [item['file_type'] for item in batch_data],
            [item['source_link'] for item in batch_data],
            [item['chunk_index'] for item in batch_data],
            [item['language'] for item in batch_data],
            [item['has_code'] for item in batch_data],
            [item['repo_name'] for item in batch_data],
            [item['content_quality_score'] for item in batch_data],
            [item['semantic_density_score'] for item in batch_data],
            [item['information_value_score'] for item in batch_data]
        ]
        
        collection.insert(entities)
        collection.flush()
        logger.info(f"Inserted {batch_end}/{len(chunk_data)} chunks")
    
    logger.info(f"Forum ingestion complete: {len(chunk_data)} chunks stored in '{collection_name}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ingest forum JSON threads into Milvus collection with semantic chunking.")
    parser.add_argument("json_path", help="Path to scraped_threads_complete.json")
    parser.add_argument("--collection", default="beaglemind_docs", help="Milvus collection name")
    parser.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="Embedding model name")
    args = parser.parse_args()
    ingest_forum_json(args.json_path, args.collection, args.model)