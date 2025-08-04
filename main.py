from fastapi import FastAPI
from app.routes.retrieval import router as retrieval_router
from app.routes.github_ingestion import router as github_ingestion_router

app = FastAPI(
    title="Information Retrieval API",
    description="API for semantic document retrieval using Milvus and ONNX embeddings",
    version="1.0.0"
)

app.include_router(retrieval_router, prefix="/api", tags=["retrieval"])
app.include_router(github_ingestion_router, prefix="/api", tags=["github_ingestion"])

@app.get("/")
async def root():
    return {"message": "Information Retrieval API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
