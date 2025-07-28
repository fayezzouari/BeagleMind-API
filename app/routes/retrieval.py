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


@router.post("/persist-knowledge")
async def persist_knowledge():
    """
    Check if beaglemind_col collection exists in local Milvus, if not, migrate it from remote.
    """
    try:
        # Connect to local Milvus
        milvus_host =  "standalone"
        milvus_port =  19530
        
        connections.connect(
            alias="local_check",
            host=milvus_host,
            port=milvus_port,
            timeout=30
        )
        
        collection_name = "beaglemind_col"
        print(f"Checking if collection '{collection_name}' exists in local Milvus...")
        # Check if collection exists
        if utility.has_collection(collection_name, using="local_check"):
            print(f"Collection '{collection_name}' exists. Dropping it before migration...")
            Collection(collection_name, using="local_check").drop()
        # Collection doesn't exist or has been dropped, perform migration
        print("Starting migration...")
        
        # Export from remote
        exported = export_collection(collection_name)
        
        if not exported:
            raise HTTPException(status_code=404, detail="No records found in remote collection to migrate")
        
        # Validate embedding structure
        emb = exported[0].get("embedding")
        if not isinstance(emb, list):
            raise HTTPException(status_code=400, detail="Embedding field not found or invalid in remote collection")
        
        embedding_dim = len(emb)
        
        # Import to local
        import_to_local(collection_name, exported, embedding_dim)
        
        return {
            "status": "success",
            "message": "Migration completed successfully",
            "collection": collection_name,
            "records_migrated": len(exported),
            "embedding_dimension": embedding_dim,
            "action": "migrated_from_remote"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge persistence failed: {str(e)}")
