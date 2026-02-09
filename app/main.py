import os

from fastapi import FastAPI
from prometheus_client import Counter
from shared.metrics import install_metrics

SERVICE_NAME = "worker"
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "1.0.0")
SERVICE_ENV = os.getenv("APP_ENV", "prod")

rag_jobs_completed_total = Counter(
    "rag_jobs_completed_total",
    "Total completed RAG jobs",
)

app = FastAPI(title="Luxia Worker", version=SERVICE_VERSION)
install_metrics(app, service_name=SERVICE_NAME, version=SERVICE_VERSION, env=SERVICE_ENV)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/worker/test")
async def worker_test() -> dict[str, object]:
    rag_jobs_completed_total.inc()
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "message": "worker completion counter incremented",
    }


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "status": "running"}
