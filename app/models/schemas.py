from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class InitializeRequest(BaseModel):
    collection_name: str = "beaglemind_col"


class InitializeResponse(BaseModel):
    success: bool
    message: str
    collection_name: str


class RetrieveRequest(BaseModel):
    query: str
    n_results: int = 10
    include_metadata: bool = True
    rerank: bool = True


class DocumentMetadata(BaseModel):
    score: float
    distance: float
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    source_link: Optional[str] = None
    chunk_index: Optional[int] = None
    language: Optional[str] = None
    has_code: Optional[bool] = None
    repo_name: Optional[str] = None


class RetrieveResponse(BaseModel):
    documents: List[List[str]]
    metadatas: List[List[DocumentMetadata]]
    distances: List[List[float]]
    total_found: int
    filtered_results: int
