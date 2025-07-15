from fastapi import APIRouter, HTTPException
from app.models.schemas import InitializeRequest, InitializeResponse, RetrieveRequest, RetrieveResponse
from app.services.retrieval_service import RetrievalService

router = APIRouter()
retrieval_service = None


@router.post("/initialize", response_model=InitializeResponse)
async def initialize(request: InitializeRequest):
    global retrieval_service
    
    try:
        retrieval_service = RetrievalService()
        retrieval_service.connect_to_milvus(request.collection_name)
        retrieval_service.create_collection(request.collection_name)
        
        return InitializeResponse(
            success=True,
            message="Retrieval system initialized successfully",
            collection_name=request.collection_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    global retrieval_service
    
    if retrieval_service is None:
        raise HTTPException(status_code=400, detail="Retrieval system not initialized. Call /initialize first.")
    
    try:
        results = retrieval_service.search(
            query=request.query,
            n_results=request.n_results,
            include_metadata=request.include_metadata,
            rerank=request.rerank
        )
        
        formatted_metadatas = []
        for metadata_list in results["metadatas"]:
            formatted_list = []
            for metadata in metadata_list:
                formatted_list.append(metadata)
            formatted_metadatas.append(formatted_list)
        
        return RetrieveResponse(
            documents=results["documents"],
            metadatas=formatted_metadatas,
            distances=results["distances"],
            total_found=results["total_found"],
            filtered_results=results["filtered_results"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
