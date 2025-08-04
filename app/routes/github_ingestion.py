"""
GitHub Ingestion Router

API endpoints for ingesting GitHub repositories into Milvus collections.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import logging

from app.services.github_ingestion_service import github_ingestion_service
from app.models.github_ingestion import IngestionRequest, IngestionResponse, IngestionStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/ingest-data", response_model=IngestionResponse)
async def ingest_github_repository(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """
    Ingest a GitHub repository into a Milvus collection.
    
    If the collection exists, the new repository data will be appended.
    If the collection doesn't exist, it will be created.
    
    Args:
        request: IngestionRequest containing collection_name, github_url, and optional branch
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        IngestionResponse with success status and details
    """
    try:
        logger.info(f"Received ingestion request for {request.github_url} into {request.collection_name}")
        
        # Validate collection name
        if not request.collection_name or len(request.collection_name.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Collection name cannot be empty"
            )
        
        # Convert HttpUrl to string
        github_url_str = str(request.github_url)
        
        # Validate GitHub URL format
        if not github_url_str.startswith("https://github.com/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid GitHub URL. Must start with https://github.com/"
            )
        
        # Start ingestion (this runs synchronously for now)
        result = await github_ingestion_service.ingest_repository(
            collection_name=request.collection_name,
            github_url=github_url_str,
            branch=request.branch
        )
        
        if result["success"]:
            return IngestionResponse(
                success=True,
                message=result["message"],
                stats=result.get("stats")
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/ingest-data/status", response_model=IngestionStatusResponse)
async def get_ingestion_status():
    """
    Get the status of the ingestion service.
    
    Returns:
        Status information about the ingestion service
    """
    try:
        return IngestionStatusResponse(
            success=True,
            message="GitHub ingestion service is running",
            active_collections=len(github_ingestion_service.ingesters)
        )
    except Exception as e:
        logger.error(f"Error getting ingestion status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting status: {str(e)}"
        )