#!/usr/bin/env python3
"""
GitHub Repository Ingestion Service

This service handles the ingestion of GitHub repositories into Milvus collections.
"""

import logging
from typing import Dict, Any
from app.scripts.github_ingestor import GitHubDirectIngester

logger = logging.getLogger(__name__)

class GitHubIngestionService:
    """Service for ingesting GitHub repositories into Milvus collections."""
    
    def __init__(self):
        self.ingesters = {}  # Cache ingesters by collection name
    
    def get_or_create_ingester(self, collection_name: str) -> GitHubDirectIngester:
        """Get existing ingester or create new one for collection."""
        if collection_name not in self.ingesters:
            self.ingesters[collection_name] = GitHubDirectIngester(
                collection_name=collection_name,
                model_name="BAAI/bge-base-en-v1.5"
            )
        return self.ingesters[collection_name]
    
    async def ingest_repository(self, collection_name: str, github_url: str, 
                              branch: str = "main") -> Dict[str, Any]:
        """
        Ingest a GitHub repository into the specified collection.
        
        Args:
            collection_name: Name of the Milvus collection
            github_url: GitHub repository URL
            branch: Repository branch to ingest
            
        Returns:
            Dictionary with success status and ingestion results
        """
        try:
            logger.info(f"Starting ingestion for {github_url} into collection {collection_name}")
            
            # Get or create ingester for this collection
            ingester = self.get_or_create_ingester(collection_name)
            
            # Ingest repository
            result = ingester.ingest_repository(
                repo_url=github_url,
                branch=branch,
                max_workers=8
            )
            
            if result['success']:
                logger.info(f"Successfully ingested {github_url} into {collection_name}")
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
                logger.error(f"Failed to ingest {github_url}: {result.get('message', 'Unknown error')}")
                return {
                    "success": False,
                    "message": f"Failed to ingest repository: {result.get('message', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"Error during repository ingestion: {e}")
            return {
                "success": False,
                "message": f"Error during ingestion: {str(e)}"
            }

# Global service instance
github_ingestion_service = GitHubIngestionService()