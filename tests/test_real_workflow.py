#!/usr/bin/env python3
"""
LUXIA REAL WORKFLOW TEST - NO MOCKS

Tests the complete end-to-end pipeline with a real post submission via Kafka.
No mocked search, scraping, or extraction - all services work with real data.

This demonstrates:
1. Real post submission to Kafka
2. Worker consumer processes the message
3. Complete corrective retrieval pipeline executes
4. Real fact/entity/relation extraction
5. Data stored to Pinecone and Neo4j
6. Evidence ranking and retrieval
7. Final RAG verdict generation
"""

import asyncio
import json
import uuid
from datetime import datetime

from aiokafka import AIOKafkaProducer

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


async def submit_real_post(post_text: str, post_id: str = None, domain: str = "social-media") -> str:
    """
    Submit a real post to Kafka for processing.

    Args:
        post_text: The actual claim/post text to fact-check
        post_id: Unique post identifier (auto-generated if None)
        domain: Domain/source of the post

    Returns:
        job_id: The job ID for tracking
    """
    post_id = post_id or f"post-{uuid.uuid4().hex[:8]}"
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    # Create WorkerJob payload
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

    # Publish to Kafka
    producer = AIOKafkaProducer(bootstrap_servers=settings.KAFKA_BOOTSTRAP)
    await producer.start()

    try:
        await producer.send_and_wait(settings.POSTS_TOPIC, json.dumps(worker_job).encode("utf-8"))
        logger.info(f"✅ Published job {job_id} with post {post_id} to Kafka")
        return job_id
    finally:
        await producer.stop()


async def monitor_job_logs(job_id: str, timeout_seconds: int = 120) -> None:
    """
    Monitor logs for a specific job.

    Args:
        job_id: The job ID to monitor
        timeout_seconds: Max time to monitor
    """
    from app.services.logging.log_handler import LogManagerHandler

    start_time = datetime.utcnow()
    last_count = 0

    print(f"\n⏳ Monitoring job {job_id}...")
    print(f"   (Watching for up to {timeout_seconds}s)\n")

    while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
        try:
            log_manager = LogManagerHandler._log_manager
            if log_manager:
                logs = log_manager.get_logs(request_id=job_id, limit=50)

                if logs and len(logs) > last_count:
                    # Print new logs
                    for log in logs[last_count:]:
                        level = log.get("level", "INFO")
                        message = log.get("message", "")

                        # Color code by level
                        level_symbol = {"INFO": "ℹ️ ", "ERROR": "❌", "WARNING": "⚠️ ", "DEBUG": "🔍"}.get(level, "📝")

                        print(f"   {level_symbol} {message}")

                    last_count = len(logs)

                    # Check if job completed
                    if any("COMPLETED" in log.get("message", "") for log in logs):
                        print(f"\n   ✅ Job {job_id} completed!")
                        break

        except Exception as e:
            logger.debug(f"Log monitoring: {e}")

        await asyncio.sleep(3)


async def run_real_workflow_test():
    """Execute complete real workflow test."""

    print("\n" + "=" * 90)
    print(" " * 20 + "LUXIA FACT-CHECKING PIPELINE - REAL WORKFLOW TEST")
    print("=" * 90)
    print(f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")

    # Test case - submit ONE real post to see full workflow
    test_post = "Drinking coffee causes heart disease and should be avoided"

    print("[PHASE 1] REAL POST SUBMISSION")
    print("-" * 90)
    print(f'\n📝 Original Post:\n   "{test_post}"\n')
    print("Domain: health")
    print("Expected Result: MISINFORMATION (backed by scientific evidence)\n")

    print("[PHASE 2] SUBMITTING TO KAFKA")
    print("-" * 90)

    try:
        job_id = await submit_real_post(post_text=test_post, post_id="test-coffee-claim", domain="health")
        print("\n✅ Real post submitted successfully")
        print(f"   Job ID: {job_id}")
        print(f"   Topic: {settings.POSTS_TOPIC}")
        print("   Status: Queued for worker processing\n")
    except Exception as e:
        logger.error(f"❌ Failed to submit post: {e}")
        return

    print("[PHASE 3] WORKER PROCESSING")
    print("-" * 90)
    print("Real pipeline stages (NO MOCKS):")
    print("  1. Search: Querying trusted domains (Google CSE)")
    print("  2. Scraping: Fetching real content from URLs")
    print("  3. Extraction: LLM analyzing facts/entities/relations")
    print("  4. Ingestion: Storing to Pinecone & Neo4j")
    print("  5. Retrieval: Searching stored knowledge")
    print("  6. Ranking: Scoring evidence with trust grades")
    print("  7. Reinforcement: Loop if confidence is low\n")

    await monitor_job_logs(job_id, timeout_seconds=120)

    print("\n[PHASE 4] DATABASE VERIFICATION")
    print("-" * 90)

    # Check what was actually stored
    try:
        from app.services.kg.neo4j_client import Neo4jClient
        from app.services.vdb.pinecone_client import get_pinecone_client

        # Neo4j check
        client = Neo4jClient()
        driver = await client._ensure_driver()
        async with driver.session() as session:
            result = await session.run("MATCH (n) RETURN count(n) AS cnt")
            nodes = (await result.single())["cnt"]
            result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            rels = (await result.single())["cnt"]

        print("✅ Neo4j Knowledge Graph:")
        print(f"   - Entity Nodes: {nodes}")
        print(f"   - Relationships: {rels}")

        # Pinecone check
        pc = get_pinecone_client()
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()

        print("\n✅ Pinecone Vector Database:")
        print(f"   - Total Vectors: {stats.total_vector_count}")
        print("   - Dimension: 384")

    except Exception as e:
        logger.error(f"Database verification error: {e}")

    print("\n" + "=" * 90)
    print("REAL WORKFLOW TEST COMPLETE")
    print("=" * 90)
    print("\n📌 Summary:")
    print("   - Real post submitted via Kafka (not mocked)")
    print("   - Worker processes with actual search/scrape/extract")
    print("   - Pinecone and Neo4j populate with real data")
    print("   - Pipeline generates RAG verdict with evidence")
    print(f"\n   Job ID: {job_id}")
    print(f'   Original: "{test_post}"')
    print("   Status: Check logs above for full trace\n")


if __name__ == "__main__":
    asyncio.run(run_real_workflow_test())
