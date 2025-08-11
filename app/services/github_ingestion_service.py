#!/usr/bin/env python3
"""
GitHub Repository Ingestion Service

This service handles the ingestion of GitHub repositories into Milvus collections.
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from app.scripts.github_ingestor import GitHubDirectIngester

logger = logging.getLogger(__name__)

class GitHubIngestionService:
    """Service for ingesting GitHub repositories into Milvus collections."""
    
    def __init__(self):
        self.ingesters = {}  # Cache ingesters by collection name
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent ingestions
    
    def get_or_create_ingester(self, collection_name: str) -> GitHubDirectIngester:
        """Get existing ingester or create new one for collection."""
        if collection_name not in self.ingesters:
            logger.info(f"[SERVICE] Creating new ingester for collection: {collection_name}")
            self.ingesters[collection_name] = GitHubDirectIngester(
                collection_name=collection_name,
                model_name="BAAI/bge-base-en-v1.5"
            )
        else:
            logger.info(f"[SERVICE] Using existing ingester for collection: {collection_name}")
        return self.ingesters[collection_name]
    
    def _sync_ingest_repository(self, collection_name: str, github_url: str, 
                               branch: str = "main") -> Dict[str, Any]:
        """
        Synchronous repository ingestion - runs in thread pool.
        """
        try:
            logger.info(f"[SERVICE] Thread started for ingesting {github_url}")
            
            # Get or create ingester for this collection
            ingester = self.get_or_create_ingester(collection_name)
            
            # Ingest repository (this is the blocking operation)
            result = ingester.ingest_repository(
                repo_url=github_url,
                branch=branch,
                max_workers=4  # Reduced to avoid overwhelming the system
            )
            
            if result['success']:
                logger.info(f"[SERVICE] Thread completed successfully for {github_url}")
                return {
                    "success": True,
                    "message": f"Successfully ingested repository into collection '{collection_name}'",
                    "stats": {
                        "files_processed": result['files_processed'],
                        "chunks_generated": result['chunks_generated'],
                        "files_with_code": result['files_with_code'],
                        "avg_quality_score": result['avg_quality_score'],
                        "total_time": result['total_time']
                    }
                }
            else:
                logger.error(f"[SERVICE] Thread failed for {github_url}: {result.get('message', 'Unknown error')}")
                return {
                    "success": False,
                    "message": f"Failed to ingest repository: {result.get('message', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"[SERVICE] Thread exception for {github_url}: {e}")
            return {
                "success": False,
                "message": f"Error during ingestion: {str(e)}"
            }
    
    async def ingest_repository(self, collection_name: str, github_url: str, 
                              branch: str = "main") -> Dict[str, Any]:
        """
        Ingest a GitHub repository into the specified collection.
        
        This method runs the blocking ingestion in a thread pool to avoid
        blocking the FastAPI event loop.
        
        Args:
            collection_name: Name of the Milvus collection
            github_url: GitHub repository URL
            branch: Repository branch to ingest
            
        Returns:
            Dictionary with success status and ingestion results
        """
        logger.info(f"[SERVICE START] Repository: {github_url}, Collection: {collection_name}, Branch: {branch}")
        
        try:
            # Run the blocking ingestion in a thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._sync_ingest_repository,
                collection_name,
                github_url,
                branch
            )
            
            if result["success"]:
                logger.info(f"[SERVICE SUCCESS] Repository {github_url} successfully processed")
                logger.info(f"[SERVICE STATS] Files processed: {result.get('stats', {}).get('files_processed', 0)}, "
                           f"Chunks generated: {result.get('stats', {}).get('chunks_generated', 0)}")
            else:
                logger.error(f"[SERVICE FAILED] Repository {github_url}: {result.get('message', 'Unknown error')}")
            
            return result
                
        except Exception as e:
            logger.error(f"[SERVICE ERROR] Critical error during repository ingestion: {e}")
            logger.error(f"[SERVICE ERROR] Repository: {github_url}, Collection: {collection_name}")
            return {
                "success": False,
                "message": f"Error during ingestion: {str(e)}"
            }

# Global service instance
github_ingestion_service = GitHubIngestionService()