from fastapi import FastAPI
from app.core.config import settings

app = FastAPI(title="RAG Worker Service", version="1.0.0")

@app.get("/health")
async def health():
    return {"status": "worker running", "service": "RAG worker"}