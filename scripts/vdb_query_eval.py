import argparse
import asyncio
from typing import List

from app.services.retrieval.metadata_enricher import TopicClassifier
from app.services.vdb.vdb_retrieval import VDBRetrieval


def _load_queries(path: str) -> List[str]:
    queries: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q and not q.startswith("#"):
                queries.append(q)
    return queries


async def _run(queries: List[str], top_k: int, namespace: str, language: str | None) -> None:
    retriever = VDBRetrieval(namespace=namespace, language=language)
    topic_classifier = TopicClassifier()
    for q in queries:
        print("=" * 80)
        print(f"QUERY: {q}")
        topics, _ = await topic_classifier.classify(q, [], None)
        if not topics:
            print("No topics inferred; skipping VDB retrieval.")
            continue
        results = await retriever.search(q, top_k=top_k, topics=topics)
        if not results:
            print("No results.")
            continue
        for i, r in enumerate(results, start=1):
            score = r.get("score", 0.0)
            stmt = (r.get("statement") or "").strip()
            src = r.get("source_url") or r.get("source") or ""
            print(f"{i}. score={score:.3f} source={src}")
            if stmt:
                print(f"   {stmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VDB retrieval for existing queries.")
    parser.add_argument("--query", action="append", default=[], help="Query string (repeatable).")
    parser.add_argument("--file", help="Path to file with one query per line.")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results to return.")
    parser.add_argument("--namespace", default="health", help="Pinecone namespace.")
    parser.add_argument("--language", default="en", help="Language filter (set to '' for none).")
    args = parser.parse_args()

    queries = list(args.query or [])
    if args.file:
        queries.extend(_load_queries(args.file))
    if not queries:
        queries = [
            "how many bones does an adult human have",
            "how many bones are in adult human hands and feet",
            "how many times does the human heart beat per day",
            "how many new cells does the human body produce per second",
            "are tongue prints unique",
        ]

    language = args.language if args.language else None
    asyncio.run(_run(queries, top_k=args.top_k, namespace=args.namespace, language=language))


if __name__ == "__main__":
    main()
