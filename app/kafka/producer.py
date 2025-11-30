import json

from aiokafka import AIOKafkaProducer

from app.core.config import DLQ_TOPIC, PROGRESS_TOPIC, RESULTS_TOPIC
from app.core.schemas import WorkerResult


class ResultPublisher:
    def __init__(self, producer: AIOKafkaProducer):
        self.producer = producer

    async def publish_progress(self, job_id: str, stage: str):
        await self.producer.send_and_wait(
            PROGRESS_TOPIC, json.dumps({"job_id": job_id, "stage": stage}).encode("utf-8")
        )

    async def publish_result(self, result: WorkerResult):
        await self.producer.send_and_wait(RESULTS_TOPIC, result.model_dump_json().encode("utf-8"))

    async def publish_dlq(self, job_payload: dict, reason: str):
        await self.producer.send_and_wait(DLQ_TOPIC, json.dumps({"reason": reason, "job": job_payload}).encode("utf-8"))
