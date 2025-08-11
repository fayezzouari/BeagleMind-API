#!/usr/bin/env python3

import uvicorn
import os
import logging

# Setup basic logging for the runner
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"[RUNNER] Starting FastAPI server on {host}:{port}")
    logger.info(f"[RUNNER] Reload mode: True")
    logger.info(f"[RUNNER] GitHub Ingestion API will be available at: http://{host}:{port}/api/ingest-data")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
