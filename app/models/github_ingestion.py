"""
GitHub Ingestion Models

Pydantic models for GitHub repository ingestion API requests and responses.
"""

from pydantic import BaseModel, HttpUrl
from typing import Optional


class IngestionRequest(BaseModel):
    """Request model for GitHub repository ingestion."""
    collection_name: str
    github_url: HttpUrl
    branch: Optional[str] = "main"


class IngestionResponse(BaseModel):
    """Response model for GitHub repository ingestion."""
    success: bool
    message: str
    stats: Optional[dict] = None


class IngestionStatusResponse(BaseModel):
    """Response model for ingestion service status."""
    success: bool
    message: str
    active_collections: int