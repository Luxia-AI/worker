import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.corrective.pipeline.retrieval_phase import retrieve_candidates
from app.services.kg.kg_ingest import KGIngest
from app.services.kg.kg_retrieval import KGRetrieval
from app.services.vdb.vdb_ingest import VDBIngest
from app.services.vdb.vdb_retrieval import VDBRetrieval

logger = get_logger(__name__)


@dataclass
class TestData:
    run_id: str
    namespace: str
    source_url: str
    facts: List[Dict[str, Any]]
    triples: List[Dict[str, Any]]
    entities: List[str]
    queries: List[str]


def build_test_data() -> TestData:
    run_id = "luxia_test_20260206"
    namespace = "luxia_test"
    source_url = f"https://nih.gov/{run_id}/bones"

    # Synthetic facts imitating LLM extraction output
    facts = [
        {
            "fact_id": f"{run_id}_fact_1",
            "statement": "The adult human skeleton has 206 bones.",
            "confidence": 0.96,
            "source_url": source_url,
            "source": "nih.gov",
            "published_at": "2020-01-01",
            "entities": ["adult human skeleton", "bones"],
        },
        {
            "fact_id": f"{run_id}_fact_2",
            "statement": "More than half of the 206 bones are in the hands and feet.",
            "confidence": 0.78,
            "source_url": source_url,
            "source": "nih.gov",
            "published_at": "2020-01-01",
            "entities": ["hands", "feet", "bones"],
        },
    ]

    # Synthetic triples imitating relation extraction output
    triples = [
        {
            "id": f"{run_id}_t1",
            "subject": "luxia_test_hands",
            "relation": "contain",
            "object": "luxia_test_bones",
            "confidence": 0.9,
            "source_url": source_url,
            "fact_id": f"{run_id}_fact_2",
        },
        {
            "id": f"{run_id}_t2",
            "subject": "luxia_test_feet",
            "relation": "contain",
            "object": "luxia_test_bones",
            "confidence": 0.9,
            "source_url": source_url,
            "fact_id": f"{run_id}_fact_2",
        },
    ]

    entities = ["luxia_test_hands", "luxia_test_feet", "luxia_test_bones"]
    queries = [
        "The adult human skeleton has 206 bones.",
        "More than half of the 206 bones are in the hands and feet.",
    ]

    return TestData(
        run_id=run_id,
        namespace=namespace,
        source_url=source_url,
        facts=facts,
        triples=triples,
        entities=entities,
        queries=queries,
    )


def _ensure_env_aliases() -> None:
    # Map legacy env var name for Neo4j if needed
    if not os.environ.get("NEO4J_USER") and os.environ.get("NEO4J_USERNAME"):
        os.environ["NEO4J_USER"] = os.environ.get("NEO4J_USERNAME", "")


def _load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _missing_required_env() -> List[str]:
    required = [
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
    ]
    missing = [key for key in required if not os.environ.get(key)]
    return missing


async def _cleanup_neo4j(kg_ingest: KGIngest, test: TestData) -> None:
    # Delete relations by rid and entities by name, plus source by url
    rids = []
    for t in test.triples:
        subj_id = kg_ingest._generate_entity_id(t["subject"])
        obj_id = kg_ingest._generate_entity_id(t["object"])
        rid = kg_ingest._generate_relation_rid(subj_id, t["relation"], obj_id)
        rids.append(rid)

    # Use direct client to execute cleanup queries
    client = kg_ingest.client
    # Delete Relation nodes
    if rids:
        await client.execute(
            "MATCH (r:Relation) WHERE r.rid IN $rids DETACH DELETE r",
            {"rids": rids},
        )

    # Delete Entity nodes by name
    await client.execute(
        "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e",
        {"names": test.entities},
    )

    # Delete Source nodes by url
    await client.execute(
        "MATCH (s:Source) WHERE s.url = $url DETACH DELETE s",
        {"url": test.source_url},
    )


async def _cleanup_pinecone(vdb_retriever: VDBRetrieval, test: TestData) -> None:
    try:
        vdb_retriever.index.delete(ids=[f["fact_id"] for f in test.facts], namespace=test.namespace)
    except Exception as e:
        logger.warning(f"[Cleanup] Pinecone delete failed: {e}")


async def main() -> None:
    _load_dotenv()
    _ensure_env_aliases()
    missing = _missing_required_env()
    if missing:
        print(
            json.dumps(
                {
                    "status": "missing_config",
                    "missing_env": missing,
                    "note": "Set the missing env vars in worker/.env and rerun this script.",
                },
                indent=2,
            )
        )
        return

    test = build_test_data()

    # Initialize services with test namespace
    vdb_ingest = VDBIngest(namespace=test.namespace)
    vdb_retriever = VDBRetrieval(namespace=test.namespace)
    kg_ingest = KGIngest()
    kg_retriever = KGRetrieval()

    # Pre-cleanup to avoid collisions
    try:
        await _cleanup_neo4j(kg_ingest, test)
    except Exception as e:
        logger.warning(f"[Cleanup] Neo4j pre-cleanup failed: {e}")

    try:
        await _cleanup_pinecone(vdb_retriever, test)
    except Exception as e:
        logger.warning(f"[Cleanup] Pinecone pre-cleanup failed: {e}")

    # Ingest mock facts and triples (imitating LLM output)
    ingested_ids = await vdb_ingest.embed_and_ingest(test.facts)
    kg_result = await kg_ingest.ingest_triples(test.triples)

    # Fetch topics from stored metadata to use in retrieval filters
    fetched = vdb_retriever.fetch_by_ids(ingested_ids)
    topics = list({(item.get("topic") or "other") for item in fetched})

    # Run retrieval using pipeline retrieval phase
    dedup_sem, kg_candidates = await retrieve_candidates(
        vdb_retriever=vdb_retriever,
        kg_retriever=kg_retriever,
        queries=test.queries,
        all_entities=test.entities,
        top_k=5,
        round_id=test.run_id,
        topics=topics,
        lexical_index=None,
        log_manager=None,
    )

    # Validate expected results
    expected_statements = {f["statement"] for f in test.facts}
    retrieved_statements = {r.get("statement") for r in dedup_sem}
    missing_vdb = expected_statements - retrieved_statements

    expected_kg = {f"{t['subject']} {t['relation']} {t['object']}" for t in test.triples}
    retrieved_kg = {r.get("statement") for r in kg_candidates}
    missing_kg = expected_kg - retrieved_kg

    result = {
        "namespace": test.namespace,
        "topics_used": topics,
        "pinecone_ingested_ids": ingested_ids,
        "neo4j_ingest": kg_result,
        "vdb_retrieval_count": len(dedup_sem),
        "kg_retrieval_count": len(kg_candidates),
        "missing_vdb_statements": sorted([m for m in missing_vdb if m]),
        "missing_kg_statements": sorted([m for m in missing_kg if m]),
        "vdb_results": dedup_sem,
        "kg_results": kg_candidates,
    }

    print(json.dumps(result, indent=2))

    # Cleanup after test
    try:
        await _cleanup_neo4j(kg_ingest, test)
    except Exception as e:
        logger.warning(f"[Cleanup] Neo4j post-cleanup failed: {e}")

    try:
        await _cleanup_pinecone(vdb_retriever, test)
    except Exception as e:
        logger.warning(f"[Cleanup] Pinecone post-cleanup failed: {e}")

    # Hard assertions for 100% confirmation
    assert not missing_vdb, f"Missing VDB statements: {missing_vdb}"
    assert not missing_kg, f"Missing KG statements: {missing_kg}"


if __name__ == "__main__":
    asyncio.run(main())
