# Worker Service

`worker` is the evidence retrieval, ranking, and verdict engine.

## Responsibilities

- Execute claim verification pipeline
- Retrieve evidence from VDB and KG paths
- Apply ranking and adaptive trust metrics
- Produce final verdict payload for downstream consumers

## Entry Points

- API: `worker/app/main.py`
- Pipeline orchestrator: `worker/app/services/corrective/pipeline/__init__.py`
- Verdict logic: `worker/app/services/verdict/verdict_generator.py`

## Dependencies

- LLM provider (Groq/OpenAI pathing through service layer)
- Pinecone
- Neo4j
- Redis (logging/state integrations)

## Canonical Docs

- `docs/worker-pipeline.md`
- `docs/interfaces-and-contracts.md`
- `docs/deployment-and-operations.md`
- `docs/testing-and-validation.md`

Last verified against code: February 13, 2026
