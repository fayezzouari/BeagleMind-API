import logging
import sys
from fastapi import FastAPI
from app.routes.retrieval import router as retrieval_router
from app.routes.github_ingestion import router as github_ingestion_router

# Configure logging to ensure all logs are visible
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Output to console
        logging.FileHandler('app.log', mode='a')  # Also log to file
    ]
)

# Set specific loggers
logger = logging.getLogger(__name__)
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('uvicorn.access').setLevel(logging.INFO)

app = FastAPI(
    title="Information Retrieval API",
    description="API for semantic document retrieval using Milvus and ONNX embeddings",
    version="1.0.0"
)

app.include_router(retrieval_router, prefix="/api", tags=["retrieval"])
app.include_router(github_ingestion_router, prefix="/api", tags=["github_ingestion"])

@app.get("/")
async def root():
    logger.info("[MAIN] Root endpoint accessed")
    return {"message": "Information Retrieval API is running"}

@app.get("/health")
async def health_check():
    logger.info("[MAIN] Health check endpoint accessed")
    return {"status": "healthy"}

logger.info("[MAIN] FastAPI application initialized successfully")
