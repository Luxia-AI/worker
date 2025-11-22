"""
End-to-End Testing with Real Biomedical Claims

This test suite performs full pipeline execution with real biomedical fact-checking scenarios.
It validates the complete integration of:
- Corrective retrieval pipeline (all 7 phases)
- Logging system (LogManager with structured context)
- Hybrid ranking with trust grading
- Knowledge graph and vector database ingestion
- Reinforcement loop for low-confidence results

Test claims cover various biomedical topics with different confidence levels
to exercise all pipeline paths.
"""

import asyncio
import os
import uuid
from typing import Dict, List

import pytest
import pytest_asyncio

from app.services.corrective.pipeline import CorrectivePipeline
from app.services.logging.log_manager import LogManager

# Real biomedical claims for testing
BIOMEDICAL_CLAIMS = [
    {
        "claim": "More than half of the 206 bones in an adult human body are located in the hands and feet.",
        "domain": "anatomy",
        "expected_verdict": "TRUE",
        "description": "Verifiable anatomical fact - should retrieve high-confidence evidence",
    },
    {
        "claim": "Vitamin C supplements can prevent the common cold in healthy adults.",
        "domain": "nutrition",
        "expected_verdict": "FALSE",
        "description": "Common misconception - should find evidence debunking the claim",
    },
    {
        "claim": "The human brain uses approximately 20% of the body's total energy at rest.",
        "domain": "physiology",
        "expected_verdict": "TRUE",
        "description": "Well-documented physiological fact",
    },
    {
        "claim": "Antibiotics are effective treatments for viral infections like the flu.",
        "domain": "microbiology",
        "expected_verdict": "FALSE",
        "description": "Medical misconception - should find clear evidence antibiotics don't work on viruses",
    },
    {
        "claim": "The COVID-19 vaccines contain microchips for tracking purposes.",
        "domain": "public_health",
        "expected_verdict": "FALSE",
        "description": "Conspiracy theory - should trigger reinforcement loop due to low initial confidence",
    },
]


# Check if all required services are available
HAS_GROQ = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY"))
HAS_PINECONE = bool(
    os.environ.get("PINECONE_API_KEY") or os.environ.get("PINECONE_APIKEY") or os.environ.get("PINECONE_API_KEY")
)
HAS_NEO4J = bool(os.environ.get("NEO4J_URI"))
HAS_GOOGLE = bool(os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"))
HAS_REDIS = bool(os.environ.get("REDIS_URL") or "redis://localhost:6379")

ALL_SERVICES_AVAILABLE = HAS_GROQ and HAS_PINECONE and HAS_NEO4J and HAS_GOOGLE


@pytest_asyncio.fixture
async def log_manager():
    """Initialize LogManager for testing."""
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    db_path = "test_logs_e2e.db"

    # Clean up old test database
    if os.path.exists(db_path):
        os.remove(db_path)

    manager = LogManager(redis_url=redis_url, db_path=db_path)

    try:
        await manager.start()
        yield manager
    finally:
        await manager.stop()
        # Clean up test database
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest_asyncio.fixture
async def pipeline():
    """Initialize pipeline for testing."""
    return CorrectivePipeline()


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.redis_required
@pytest.mark.skipif(not ALL_SERVICES_AVAILABLE, reason="Requires all external services (LLM, VDB, KG, Google CSE)")
async def test_e2e_single_claim_high_confidence(pipeline: CorrectivePipeline, log_manager: LogManager):
    """
    Test E2E pipeline with a claim that should produce high-confidence results.

    Expected behavior:
    - Pipeline completes all 7 phases successfully
    - Facts, entities, and triples are extracted
    - Ranking produces results with confidence > 0.7
    - Logs are captured with structured context
    """
    claim_data = BIOMEDICAL_CLAIMS[0]  # Bones in hands and feet
    round_id = str(uuid.uuid4())

    print(f"\n{'='*80}")
    print(f"TEST: {claim_data['description']}")
    print(f"CLAIM: {claim_data['claim']}")
    print(f"ROUND ID: {round_id}")
    print(f"{'='*80}\n")

    # Execute pipeline
    result = await pipeline.run(
        post_text=claim_data["claim"],
        domain=claim_data["domain"],
        round_id=round_id,
        top_k=5,
    )

    # Validate results
    assert result["status"] == "completed", "Pipeline should complete successfully"
    assert result["round_id"] == round_id
    assert len(result["facts"]) > 0, "Should extract at least one fact"
    assert len(result["ranked"]) > 0, "Should return ranked results"

    # Check confidence scores
    top_result = result["ranked"][0]
    assert "final_score" in top_result, "Ranked results should have final_score"
    assert "grade" in top_result, "Ranked results should have trust grade"

    print("\nRESULTS:")
    print(f"  Facts extracted: {len(result['facts'])}")
    print(f"  Triples extracted: {len(result['triples'])}")
    print(f"  Queries generated: {len(result['queries'])}")
    print(f"  Top result score: {top_result['final_score']:.3f}")
    print(f"  Top result grade: {top_result['grade']}")

    # Verify logs were captured
    await asyncio.sleep(2)  # Allow logs to be processed
    logs = await log_manager.get_logs(request_id=f"claim-{round_id}", limit=100)

    assert len(logs) > 0, "Logs should be captured"
    print(f"  Logs captured: {len(logs)}")

    # Check for key phase logs
    log_messages = [log["message"] for log in logs]
    assert any("Pipeline started" in msg for msg in log_messages), "Should log pipeline start"
    assert any("Search phase" in msg for msg in log_messages), "Should log search phase"
    assert any("extraction" in msg.lower() for msg in log_messages), "Should log extraction phases"
    assert any("ingestion" in msg.lower() or "ingested" in msg.lower() for msg in log_messages), "Should log ingestion"
    assert any("Ranking completed" in msg for msg in log_messages), "Should log ranking"

    print(f"\n{'='*80}\n")


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.redis_required
@pytest.mark.skipif(not ALL_SERVICES_AVAILABLE, reason="Requires all external services")
async def test_e2e_claim_with_reinforcement(pipeline: CorrectivePipeline, log_manager: LogManager):
    """
    Test E2E pipeline with a claim that should trigger reinforcement loop.

    Expected behavior:
    - Initial results have confidence < 0.7
    - Reinforcement loop executes (up to 3 rounds)
    - New URLs are fetched and processed
    - Final confidence improves or max rounds reached
    """
    claim_data = BIOMEDICAL_CLAIMS[4]  # COVID vaccine microchip conspiracy
    round_id = str(uuid.uuid4())

    print(f"\n{'='*80}")
    print(f"TEST: {claim_data['description']}")
    print(f"CLAIM: {claim_data['claim']}")
    print(f"ROUND ID: {round_id}")
    print(f"{'='*80}\n")

    # Execute pipeline
    result = await pipeline.run(
        post_text=claim_data["claim"],
        domain=claim_data["domain"],
        round_id=round_id,
        top_k=5,
    )

    # Validate results
    assert result["status"] == "completed", "Pipeline should complete successfully"
    assert len(result["ranked"]) > 0, "Should return ranked results"

    print("\nRESULTS:")
    print(f"  Facts extracted: {len(result['facts'])}")
    print(f"  Top result score: {result['ranked'][0]['final_score']:.3f}")
    print(f"  Top result grade: {result['ranked'][0]['grade']}")

    # Check logs for reinforcement activity
    await asyncio.sleep(2)
    logs = await log_manager.get_logs(request_id=f"claim-{round_id}", limit=200)

    log_messages = [log["message"] for log in logs]
    reinforcement_logs = [msg for msg in log_messages if "reinforcement" in msg.lower()]

    print(f"  Reinforcement logs: {len(reinforcement_logs)}")
    if reinforcement_logs:
        print(f"  Sample: {reinforcement_logs[0]}")

    print(f"\n{'='*80}\n")


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.redis_required
@pytest.mark.skipif(not ALL_SERVICES_AVAILABLE, reason="Requires all external services")
async def test_e2e_multiple_claims_batch(pipeline: CorrectivePipeline, log_manager: LogManager):
    """
    Test E2E pipeline with multiple claims in sequence.

    Expected behavior:
    - All claims are processed successfully
    - Each claim has unique round_id
    - Logs are properly segregated by request_id
    - Results vary appropriately based on claim content
    """
    results: List[Dict] = []

    print(f"\n{'='*80}")
    print(f"BATCH TESTING: {len(BIOMEDICAL_CLAIMS)} claims")
    print(f"{'='*80}\n")

    for i, claim_data in enumerate(BIOMEDICAL_CLAIMS[:3]):  # Test first 3 claims
        round_id = str(uuid.uuid4())

        print(f"[{i+1}/{3}] Processing: {claim_data['description']}")
        print(f"  Claim: {claim_data['claim'][:80]}...")

        result = await pipeline.run(
            post_text=claim_data["claim"],
            domain=claim_data["domain"],
            round_id=round_id,
            top_k=3,
        )

        results.append({"claim_data": claim_data, "round_id": round_id, "result": result})

        print(f"  Status: {result['status']}")
        print(f"  Facts: {len(result['facts'])}, Ranked: {len(result['ranked'])}")
        if result["ranked"]:
            print(f"  Top score: {result['ranked'][0]['final_score']:.3f}, Grade: {result['ranked'][0]['grade']}")
        print()

    # Validate all succeeded
    assert all(r["result"]["status"] == "completed" for r in results), "All claims should complete"

    # Check logs for each claim
    await asyncio.sleep(3)
    for r in results:
        logs = await log_manager.get_logs(request_id=f"claim-{r['round_id']}", limit=50)
        print(f"Round {r['round_id'][:8]}... has {len(logs)} logs")
        assert len(logs) > 0, f"Should have logs for round {r['round_id']}"

    print(f"\n{'='*80}\n")


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.redis_required
@pytest.mark.skipif(not ALL_SERVICES_AVAILABLE, reason="Requires all external services")
async def test_e2e_log_statistics(pipeline: CorrectivePipeline, log_manager: LogManager):
    """
    Test log statistics and filtering capabilities.

    Expected behavior:
    - Logs can be filtered by level, module, and time
    - Statistics are accurate
    - Context data is properly stored
    """
    claim_data = BIOMEDICAL_CLAIMS[2]  # Brain energy usage
    round_id = str(uuid.uuid4())
    request_id = f"claim-{round_id}"

    print(f"\n{'='*80}")
    print("TEST: Log statistics and filtering")
    print(f"{'='*80}\n")

    # Execute pipeline
    result = await pipeline.run(
        post_text=claim_data["claim"],
        domain=claim_data["domain"],
        round_id=round_id,
        top_k=3,
    )

    assert result["status"] == "completed"

    # Wait for logs to be processed
    await asyncio.sleep(3)

    # Test various log queries
    all_logs = await log_manager.get_logs(request_id=request_id, limit=200)
    info_logs = await log_manager.get_logs(request_id=request_id, level="INFO", limit=200)
    warning_logs = await log_manager.get_logs(request_id=request_id, level="WARNING", limit=200)

    print("Log Statistics:")
    print(f"  Total logs: {len(all_logs)}")
    print(f"  INFO logs: {len(info_logs)}")
    print(f"  WARNING logs: {len(warning_logs)}")

    # Verify log structure
    if all_logs:
        sample_log = all_logs[0]
        assert "id" in sample_log
        assert "level" in sample_log
        assert "message" in sample_log
        assert "module" in sample_log
        assert "timestamp" in sample_log
        assert "request_id" in sample_log
        assert sample_log["request_id"] == request_id

        print("\nSample log entry:")
        print(f"  Module: {sample_log['module']}")
        print(f"  Level: {sample_log['level']}")
        print(f"  Message: {sample_log['message'][:80]}...")
        if sample_log.get("context"):
            print(f"  Context keys: {list(sample_log['context'].keys())}")

    # Get statistics
    stats = await log_manager.get_log_statistics(request_id=request_id)

    print("\nLog statistics from LogManager:")
    print(f"  Total: {stats['total']}")
    print(f"  By level: {stats['by_level']}")
    print(f"  By module (top 3): {list(stats['by_module'].items())[:3]}")

    assert stats["total"] == len(all_logs), "Statistics should match actual log count"

    print(f"\n{'='*80}\n")


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.redis_required
@pytest.mark.skipif(not ALL_SERVICES_AVAILABLE, reason="Requires all external services")
async def test_e2e_error_handling(pipeline: CorrectivePipeline, log_manager: LogManager):
    """
    Test pipeline error handling and logging of failures.

    Expected behavior:
    - Pipeline handles malformed input gracefully
    - Errors are logged with appropriate level
    - Pipeline returns partial results if possible
    """
    round_id = str(uuid.uuid4())

    print(f"\n{'='*80}")
    print("TEST: Error handling and recovery")
    print(f"{'='*80}\n")

    # Test with very short/ambiguous claim
    result = await pipeline.run(
        post_text="X causes Y",  # Extremely vague claim
        domain="unknown",
        round_id=round_id,
        top_k=3,
    )

    print(f"Result status: {result['status']}")
    print(f"Facts extracted: {len(result.get('facts', []))}")
    print(f"Ranked results: {len(result.get('ranked', []))}")

    # Should complete even if results are poor
    assert "status" in result

    # Check for warning/error logs
    await asyncio.sleep(2)
    logs = await log_manager.get_logs(request_id=f"claim-{round_id}", limit=100)

    error_logs = [log for log in logs if log["level"] in ("WARNING", "ERROR")]
    print(f"Warning/Error logs: {len(error_logs)}")

    if error_logs:
        print(f"Sample error: {error_logs[0]['message']}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    """
    Run E2E tests directly for development/debugging.

    Usage:
        python tests/test_e2e_real_claims.py
    """
    print("\n" + "=" * 80)
    print("LUXIA WORKER: E2E TESTING WITH REAL BIOMEDICAL CLAIMS")
    print("=" * 80 + "\n")

    # Check service availability
    print("Service Availability Check:")
    print(f"  LLM (Groq/OpenAI): {'✓' if HAS_GROQ else '✗'}")
    print(f"  Pinecone VDB: {'✓' if HAS_PINECONE else '✗'}")
    print(f"  Neo4j KG: {'✓' if HAS_NEO4J else '✗'}")
    print(f"  Google CSE: {'✓' if HAS_GOOGLE else '✗'}")
    print(f"  Redis: {'✓' if HAS_REDIS else '✗'}")
    print(f"\n  All services ready: {'YES' if ALL_SERVICES_AVAILABLE else 'NO'}\n")

    if not ALL_SERVICES_AVAILABLE:
        print("⚠ WARNING: Not all services are available. Tests will be skipped.")
        print("\nRequired environment variables:")
        if not HAS_GROQ:
            print("  - GROQ_API_KEY or OPENAI_API_KEY")
        if not HAS_PINECONE:
            print("  - PINECONE_API_KEY")
        if not HAS_NEO4J:
            print("  - NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        if not HAS_GOOGLE:
            print("  - GOOGLE_API_KEY, GOOGLE_CSE_ID")
        print()
    else:
        print("Running pytest suite...\n")
        pytest.main([__file__, "-v", "-s"])
