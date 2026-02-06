import argparse
import asyncio
from typing import Any, Dict, List

from app.core.logger import get_logger
from app.services.retrieval.lexical_index import LexicalIndex
from app.services.retrieval.metadata_enricher import MetadataEnricher
from app.services.vdb.pinecone_client import get_pinecone_index

logger = get_logger(__name__)


async def _enrich_and_index(index, lexical: LexicalIndex, enricher: MetadataEnricher, ids: List[str], namespace: str):
    if not ids:
        return
    try:
        response = index.fetch(ids=ids, namespace=namespace)
    except Exception as e:
        logger.warning(f"[Backfill] Fetch failed for batch: {e}")
        return

    vectors = response.get("vectors") or {}
    facts_to_index: List[Dict[str, Any]] = []

    for vec_id, vec_data in vectors.items():
        metadata = vec_data.get("metadata") or {}
        statement = metadata.get("statement") or ""
        if not statement:
            continue
        fact: Dict[str, Any] = dict(metadata)
        fact["fact_id"] = vec_id
        enriched = await enricher.enrich_fact(fact)
        facts_to_index.append(enriched)

        # Update Pinecone metadata if new fields are missing
        set_meta = {
            "domain": enriched.get("domain"),
            "topic": enriched.get("topic"),
            "source": enriched.get("source"),
            "doc_type": enriched.get("doc_type"),
            "fact_type": enriched.get("fact_type"),
            "count_value": enriched.get("count_value"),
        }
        set_meta = {k: v for k, v in set_meta.items() if v is not None}
        try:
            index.update(id=vec_id, namespace=namespace, set_metadata=set_meta)
        except Exception:
            # Some clients use different argument name
            try:
                index.update(id=vec_id, namespace=namespace, metadata=set_meta)
            except Exception as e:
                logger.warning(f"[Backfill] Metadata update failed for {vec_id}: {e}")

    lexical.upsert_facts(facts_to_index)


async def _run(namespace: str):
    index = get_pinecone_index()
    lexical = LexicalIndex()
    enricher = MetadataEnricher()

    total = 0
    for ids_batch in index.list(namespace=namespace):
        if not ids_batch:
            continue
        await _enrich_and_index(index, lexical, enricher, ids_batch, namespace)
        total += len(ids_batch)
        logger.info(f"[Backfill] Processed {total} facts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill FTS5 index and metadata from Pinecone.")
    parser.add_argument("--namespace", default="health", help="Pinecone namespace")
    args = parser.parse_args()

    asyncio.run(_run(args.namespace))


if __name__ == "__main__":
    main()
