from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel


class WorkerJob(BaseModel):
    job_id: str
    assigned_worker_group: str
    attempt: int
    post: Dict[str, Any]


class WorkerResult(BaseModel):
    job_id: str
    post_id: str
    truth_score: float
    confidence: float
    evidence: list
    sources: list
    status: str
    completed_at: str = datetime.utcnow().isoformat()
