from fastapi import FastAPI

from app.routers.pinecone import router as pinecone_router

app = FastAPI(title="Luxia Worker Service", version="1.0.0")
app.include_router(pinecone_router, prefix="/worker", tags=["Pinecone"])
