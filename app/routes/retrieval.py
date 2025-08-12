from fastapi import APIRouter, HTTPException
from app.models.schemas import RetrieveRequest, RetrieveResponse
from app.services.retrieval_service import RetrievalService
from app.services.persist_knowledge_service import export_collection, import_to_local
from pymilvus import connections, utility, Collection
import os

router = APIRouter()
retrieval_services = {}


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    global retrieval_services
    
    if request.collection_name not in retrieval_services:
        try:
            retrieval_service = RetrievalService()
            retrieval_service.connect_to_milvus()
            retrieval_service.create_collection(request.collection_name)
            retrieval_services[request.collection_name] = retrieval_service
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize collection {request.collection_name}: {str(e)}")
    
    retrieval_service = retrieval_services[request.collection_name]
    
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
