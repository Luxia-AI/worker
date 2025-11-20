from fastapi import FastAPI

from app.core.logger import get_logger
from app.routers.pinecone import router as pinecone_router

logger = get_logger(__name__)

app = FastAPI(title="Luxia Worker Service", version="1.0.0")
app.include_router(pinecone_router, prefix="/worker", tags=["Pinecone"])

logger.info("Luxia Worker Service initialized")
