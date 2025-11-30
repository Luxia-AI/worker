#!/usr/bin/env python3
import asyncio
import json
import uuid
from datetime import datetime

from aiokafka import AIOKafkaProducer

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


async def submit_real_post(post_text, post_id=None, domain="social-media"):
    post_id = post_id or f"post-{uuid.uuid4().hex[:8]}"
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    worker_job = {
        "job_id": job_id,
        "assigned_worker_group": "default",
        "attempt": 1,
        "post": {
            "post_id": post_id,
            "text": post_text,
            "domain": domain,
            "submitted_at": datetime.utcnow().isoformat(),
        },
    }

    producer = AIOKafkaProducer(bootstrap_servers=settings.KAFKA_BOOTSTRAP)
    await producer.start()

    try:
        await producer.send_and_wait(settings.POSTS_TOPIC, json.dumps(worker_job).encode("utf-8"))
        logger.info(f"Published job {job_id}")
        return job_id
    finally:
        await producer.stop()


async def main():
    print("\n" + "=" * 80)
    print("LUXIA REAL WORKFLOW TEST - NO MOCKS")
    print("=" * 80 + "\n")

    test_post = "Drinking coffee causes heart disease and should be avoided"

    print("[STAGE 1] ORIGINAL POST")
    print("-" * 80)
    print(f'Post: "{test_post}"')
    print("Domain: health\n")

    print("[STAGE 2] KAFKA SUBMISSION (REAL)")
    print("-" * 80)

    job_id = await submit_real_post(post_text=test_post, post_id="test-coffee-real-workflow", domain="health")

    print("✓ Real post submitted to Kafka")
    print(f"  Job ID: {job_id}")
    print(f"  Topic: {settings.POSTS_TOPIC}")
    print(f"  Bootstrap: {settings.KAFKA_BOOTSTRAP}\n")

    print("[STAGE 3] LIVE PIPELINE EXECUTION")
    print("-" * 80)
    print("Processing stages (NO MOCKS):")
    print("  1. Kafka picks up message")
    print("  2. Search: Real Google CSE queries to trusted domains")
    print("  3. Scraping: Extract actual content from URLs")
    print("  4. Extraction: LLM processes facts/entities/relations")
    print("  5. Ingestion: Store to Pinecone + Neo4j")
    print("  6. Retrieval: Search stored knowledge")
    print("  7. Ranking: Score with trust grades")
    print("  8. RAG: Generate final verdict\n")

    print("[STAGE 4] WAITING FOR PROCESSING")
    print("-" * 80)
    print("Giving worker 15 seconds to start processing...\n")

    await asyncio.sleep(15)

    print("[STAGE 5] CHECKING DATABASES")
    print("-" * 80)

    try:
        from app.services.kg.neo4j_client import Neo4jClient
        from app.services.vdb.pinecone_client import get_pinecone_client

        client = Neo4jClient()
        driver = await client._ensure_driver()
        async with driver.session() as session:
            result = await session.run("MATCH (n) RETURN count(n) AS cnt")
            nodes = (await result.single())["cnt"]
            result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            rels = (await result.single())["cnt"]

        print(f"✓ Neo4j: {nodes} entities, {rels} relationships")

        pc = get_pinecone_client()
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        print(f"✓ Pinecone: {stats.total_vector_count} vectors")

    except Exception as e:
        print(f"Database check: {e}")

    print("\n" + "=" * 80)
    print("REAL WORKFLOW SUBMITTED")
    print("=" * 80)
    print(f"\nJob ID: {job_id}")
    print(f"Track with: docker-compose logs worker | grep '{job_id}'")
    print("Monitor worker logs for complete trace of pipeline execution\n")


if __name__ == "__main__":
    asyncio.run(main())
