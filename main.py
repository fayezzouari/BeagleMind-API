from fastapi import FastAPI
from app.routes.retrieval import router as retrieval_router

app = FastAPI(
    title="Information Retrieval API",
    description="API for semantic document retrieval using Milvus and sentence transformers",
    version="1.0.0"
)

app.include_router(retrieval_router, prefix="/api", tags=["retrieval"])


@app.get("/")
async def root():
    return {"message": "Information Retrieval API is running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
