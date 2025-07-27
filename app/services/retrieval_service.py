import json
import uuid
import logging
import re
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import os
from app.config import (
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_USER,
    MILVUS_PASSWORD,
    MILVUS_TOKEN,
    MILVUS_URI
)


logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)


class RetrievalService:
    def __init__(self):
        # Initialize embedding model with ONNX
        try:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
            self.embedding_session = ort.InferenceSession("onnx/model.onnx")
            self.has_embedding_model = True
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            self.embedding_tokenizer = None
            self.embedding_session = None
            self.has_embedding_model = False
        
        # Initialize reranker model with ONNX
        try:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.reranker_session = ort.InferenceSession("onnx/cross_encoder.onnx")
            self.has_reranker = True
        except Exception as e:
            logger.warning(f"Could not load reranker model: {e}")
            self.reranker_tokenizer = None
            self.reranker_session = None
            self.has_reranker = False
        
        self.collection = None
        
    def connect_to_milvus(self, collection_name: str):
        connect_kwargs = {
            'alias': "default",
            'timeout': 30
        }
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
        
    def _encode_text(self, text: str) -> List[float]:
        """Encode text using ONNX embedding model"""
        if not self.has_embedding_model:
            raise ValueError("Embedding model not loaded")
            
        inputs = self.embedding_tokenizer(
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
        outputs = self.embedding_session.run(None, onnx_inputs)
        
        # Use mean pooling over token embeddings
        embedding = outputs[0][0].mean(axis=0)
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        normalized_embedding = (embedding / norm) if norm != 0 else embedding
        
        return normalized_embedding.tolist()
        
    def create_collection(self, collection_name: str):
        # Get embedding dimension from a sample
        if not self.has_embedding_model:
            # Default dimension for BGE base ONNX model (768 for bge-base-en-v1.5)
            embedding_dim = 768
        else:
            try:
                sample_embedding = self._encode_text("test")
                embedding_dim = len(sample_embedding)
            except Exception as e:
                logger.warning(f"Error getting embedding dimension: {e}")
                # Fallback to common dimensions
                embedding_dim = 768
        
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
        
        schema = CollectionSchema(fields, "Repository content with semantic chunking")
        
        if utility.has_collection(collection_name):
            # Check if existing collection has matching dimension
            existing_collection = Collection(collection_name)
            existing_dim = None
            for field in existing_collection.schema.fields:
                if field.name == "embedding":
                    existing_dim = field.params.get('dim')
                    break
            
            if existing_dim != embedding_dim:
                logger.info(f"Dimension mismatch: existing collection has {existing_dim}, but model produces {embedding_dim}")
                logger.info("Dropping and recreating collection...")
                utility.drop_collection(collection_name)
                self.collection = Collection(collection_name, schema)
                
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.collection.create_index("embedding", index_params)
            else:
                self.collection = existing_collection
        else:
            self.collection = Collection(collection_name, schema)
            
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
        
        self.collection.load()
        
    def search(self, query: str, n_results: int = 10, include_metadata: bool = True, rerank: bool = True) -> Dict[str, Any]:
        if self.collection is None:
            raise ValueError("Collection not created.")
            
        self.collection.load()
        
        # Use ONNX embedding model
        if not self.has_embedding_model:
            raise ValueError("Embedding model not loaded")
            
        embedding = self._encode_text(query)
        
        # Convert to numpy array with correct shape (1, embedding_dim)
        query_embedding = np.array([embedding], dtype=np.float32)
        
        # Verify the shape is correct
        if query_embedding.ndim != 2:
            query_embedding = query_embedding.reshape(1, -1)
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        output_fields = ["document"]
        enhanced_fields = [
            "file_name", "file_path", "file_type", "source_link", "chunk_index", 
            "language", "has_code", "repo_name", "content_quality_score", 
            "semantic_density_score", "information_value_score"
        ]
        
        if include_metadata:
            try:
                collection_fields = [field.name for field in self.collection.schema.fields]
                output_fields.extend([field for field in enhanced_fields if field in collection_fields])
            except:
                pass
        
        search_limit = n_results * 3 if rerank else n_results
        
        try:
            results = self.collection.search(
                query_embedding, 
                "embedding", 
                search_params, 
                limit=search_limit,
                output_fields=output_fields,
                expr=None
            )
        except Exception as e:
            logger.warning(f"Search with enhanced fields failed: {e}")
            basic_fields = ["document"]
            results = self.collection.search(
                query_embedding, 
                "embedding", 
                search_params, 
                limit=search_limit,
                output_fields=basic_fields,
            )
        
        if not results or len(results[0]) == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "total_found": 0,
                "filtered_results": 0
            }
        
        hits = results[0]
        
        if rerank and len(hits) > n_results:
            hits = self._rerank_results(hits, query, n_results)
        else:
            hits = hits[:n_results]
        
        documents = []
        metadatas = []
        distances = []
        
        for hit in hits:
            doc_text = hit.entity.get("document", "")
            documents.append(doc_text)
            
            metadata = {
                "score": float(hit.score) if hasattr(hit, 'score') else (1 - hit.distance),
                "distance": float(hit.distance)
            }
            
            for field in output_fields:
                if field != "document":
                    value = hit.entity.get(field)
                    if value is not None:
                        metadata[field] = value
            
            metadatas.append(metadata)
            distances.append(float(hit.distance))
        
        return {
            "documents": [documents],
            "metadatas": [metadatas], 
            "distances": [distances],
            "total_found": len(results[0]),
            "filtered_results": len(hits)
        }
    
    def _rerank_results(self, hits: List[Any], query: str, n_results: int) -> List[Any]:
        documents = [hit.entity.get("document", "") for hit in hits]
        rerank_scores = None
        
        if self.has_reranker:
            try:
                # Prepare sentence pairs for reranking
                sentence_pairs = [(query, doc) for doc in documents]
                
                # Tokenize inputs
                inputs = self.reranker_tokenizer(
                    sentence_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='np',
                    max_length=512
                )
                
                # Run ONNX inference
                outputs = self.reranker_session.run(
                    None,
                    {
                        'input_ids': inputs['input_ids'].astype(np.int64),
                        'attention_mask': inputs['attention_mask'].astype(np.int64),
                        'token_type_ids': inputs['token_type_ids'].astype(np.int64)
                    }
                )
                
                rerank_scores = outputs[0].flatten()
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        scored_hits = []
        for i, hit in enumerate(hits):
            semantic_score = 1 - hit.distance
            
            quality_score = hit.entity.get("content_quality_score", 0.5)
            semantic_density = hit.entity.get("semantic_density_score", 0.5)
            info_value = hit.entity.get("information_value_score", 0.5)
            
            if rerank_scores is not None:
                rerank_score = rerank_scores[i]
                composite_score = (
                    rerank_score * 0.5 +
                    semantic_score * 0.2 +
                    quality_score * 0.1 +
                    semantic_density * 0.1 +
                    info_value * 0.1
                )
            else:
                query_terms = set(re.findall(r'\b\w+\b', query.lower()))
                doc_text = documents[i].lower()
                doc_terms = set(re.findall(r'\b\w+\b', doc_text))
                keyword_overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
                
                composite_score = (
                    semantic_score * 0.4 +
                    keyword_overlap * 0.3 +
                    quality_score * 0.1 +
                    semantic_density * 0.1 +
                    info_value * 0.1
                )
            
            scored_hits.append((composite_score, hit))
        
        scored_hits.sort(key=lambda x: x[0], reverse=True)
        return [hit for _, hit in scored_hits[:n_results]]
