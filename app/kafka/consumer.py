import json
import logging

from aiokafka import AIOKafkaConsumer
from pydantic import ValidationError

from app.algorithms.trigger_pipeline import run_worker_pipeline
from app.core.config import GROUP_ID, JOBS_TOPIC, MAX_ATTEMPTS
from app.core.schemas import WorkerJob, WorkerResult
from app.kafka.producer import ResultPublisher

logger = logging.getLogger("worker")


class WorkerJobConsumer:
    """
    Main consumer for worker pipeline.
    Consumes job messages, runs verification pipeline, publishes results.
    """

    def __init__(self, consumer: AIOKafkaConsumer, publisher: ResultPublisher):
        self.consumer = consumer
        self.publisher = publisher

    async def start_loop(self):
        logger.info("Worker consumer started on topic %s", JOBS_TOPIC)

        async for msg in self.consumer:
            try:
                payload = json.loads(msg.value.decode("utf-8"))
                job = WorkerJob(**payload)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error("Invalid job received: %s", e)
                await self.publisher.publish_dlq(payload, str(e))
                continue

            # Check worker group routing
            if job.assigned_worker_group != GROUP_ID:
                logger.info(
                    "Skipping job %s: assigned group %s does not match this worker (%s)",
                    job.job_id,
                    job.assigned_worker_group,
                    GROUP_ID,
                )
                continue

            await self.publisher.publish_progress(job.job_id, "received")

            try:
                # Execute RAG Pipeline
                result_dict = await run_worker_pipeline(job.post)
                result = WorkerResult(job_id=job.job_id, **result_dict)

                await self.publisher.publish_progress(job.job_id, "completed")
                await self.publisher.publish_result(result)

                logger.info("Job %s completed for post %s", job.job_id, job.post["post_id"])

            except Exception as e:
                logger.exception("Worker error on job %s: %s", job.job_id, e)

                # retry logic
                if job.attempt + 1 < MAX_ATTEMPTS:
                    retry_job = job.model_copy(update={"attempt": job.attempt + 1})
                    await self.publisher.publish_dlq(
                        retry_job.model_dump(), f"Retrying attempt {retry_job.attempt} - {str(e)}"
                    )
                else:
                    await self.publisher.publish_dlq(job.model_dump(), f"Max attempts exceeded: {str(e)}")
