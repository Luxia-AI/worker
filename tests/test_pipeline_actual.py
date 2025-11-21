import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Use your pipeline implementation
from app.services.corrective.pipeline import CorrectivePipeline

# local uploaded file path (as requested)
UPLOADED_PDF_PATH = "/mnt/data/Research Project Proposal.pdf"

CLAIM = "More than half of the 206 bones in an adult human body are located in the hands and feet."


@pytest.mark.asyncio
async def _try_load_pdf_text(path: str) -> str | None:
    """
    Try to extract plain text from a PDF located at `path`.
    Returns combined text or None if extraction failed / library not installed.
    """
    p = Path(path)
    if not p.exists():
        return None

    # Try pypdf or PyPDF2 depending on what is installed
    try:
        # pypdf
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(p))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(pages).strip() or None
    except Exception:
        pass

    try:
        # PyPDF2 (older package)
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(p))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(pages).strip() or None
    except Exception:
        pass

    return None


@pytest.mark.asyncio
async def test_pipeline_actual():
    pipeline = CorrectivePipeline()

    # Determine which external services appear configured
    env = os.environ
    has_groq = bool(env.get("GROQ_API_KEY") or env.get("OPENAI_API_KEY"))
    has_pinecone = bool(env.get("PINECONE_API_KEY") or env.get("PINECONE_APIKEY") or env.get("PINECONE_API_KEY"))
    has_neo4j = bool(env.get("NEO4J_URI") or env.get("NEO4J_URI"))
    has_google = bool(env.get("GOOGLE_API_KEY") and env.get("GOOGLE_CSE_ID"))

    # Attempt to read the uploaded PDF to use as a scraped page if available
    pdf_text = await _try_load_pdf_text(UPLOADED_PDF_PATH)

    # Build a fallback scraped page using the PDF or the claim text
    scraped_page = {
        "url": UPLOADED_PDF_PATH if pdf_text else "https://example.com/claim-page",
        "content": pdf_text or ("Claim page placeholder. " "Using the claim text as page content: " + CLAIM),
        "source": Path(UPLOADED_PDF_PATH).name if pdf_text else "example.com",
        "published_at": None,
    }

    # If critical services are missing, we'll patch parts of pipeline to use mocks,
    # otherwise we run unmocked (real) calls.
    need_mock = not (has_groq and has_pinecone and has_neo4j and has_google)

    print("ENV DETECTION:")
    print(f"  GROQ/LLM available: {has_groq}")
    print(f"  Pinecone available: {has_pinecone}")
    print(f"  Neo4j available: {has_neo4j}")
    print(f"  Google CSE available: {has_google}")
    print(f"  Will use mocks for missing services: {need_mock}")
    print()

    if need_mock:
        # Mock TrustedSearch.run to return either the uploaded file path or an example URL
        mocked_search_results = [UPLOADED_PDF_PATH] if pdf_text else ["https://example.com/claim-page"]

        # Patch multiple components with async mocks that return deterministic values
        pipeline.search_agent.run = AsyncMock(return_value=mocked_search_results)

        # Patch Scraper.scrape_all to return our scraped_page
        pipeline.scraper.scrape_all = AsyncMock(return_value=[scraped_page])

        # Patch FactExtractor.extract to return a small set of facts (including the claim)
        async def fake_extract(pages):
            return [
                {
                    "fact_id": "f-claim-1",
                    "statement": CLAIM,
                    "confidence": 0.6,
                    "source_url": pages[0]["url"],
                    "source": pages[0]["source"],
                    "published_at": pages[0].get("published_at"),
                    "entities": ["hands", "feet", "bones"],
                },
                {
                    "fact_id": "f-ref-1",
                    "statement": "The adult human skeleton has 206 bones.",
                    "confidence": 0.95,
                    "source_url": pages[0]["url"],
                    "source": pages[0]["source"],
                    "published_at": pages[0].get("published_at"),
                    "entities": ["skeleton", "bones"],
                },
            ]

        pipeline.fact_extractor.extract = AsyncMock(side_effect=fake_extract)

        # Patch EntityExtractor.annotate_entities to simply pass entities through (already on facts)
        pipeline.entity_extractor.annotate_entities = AsyncMock(side_effect=lambda facts: facts)

        # Patch RelationExtractor.extract_relations to return a simple triple list
        pipeline.relation_extractor.extract_relations = AsyncMock(
            return_value=[
                {
                    "id": "t1",
                    "subject": "hands",
                    "relation": "contain",
                    "object": "bones",
                    "confidence": 0.9,
                    "source_url": scraped_page["url"],
                    "fact_id": "f-claim-1",
                },
            ]
        )

        # Patch VDB ingest so we don't actually call Pinecone
        pipeline.vdb_ingest.embed_and_ingest = AsyncMock(return_value=["f-claim-1", "f-ref-1"])

        # Patch vdb_retriever.search to return semantic candidates (simulate vector hits)
        pipeline.vdb_retriever.search = AsyncMock(
            return_value=[
                {
                    "statement": "The adult human skeleton has 206 bones.",
                    "score": 0.95,
                    "entities": ["skeleton", "bones"],
                    "source_url": "https://nih.gov/article",
                    "published_at": "2020-01-01T00:00:00+00:00",
                    "credibility": 0.95,
                },
                {
                    "statement": "More than half of the 206 bones are in the hands and feet.",
                    "score": 0.55,
                    "entities": ["hands", "feet", "bones"],
                    "source_url": scraped_page["url"],
                    "published_at": scraped_page.get("published_at"),
                    "credibility": 0.6,
                },
            ]
        )

        # Patch KG retrieval to return the uploaded file as provenance
        # If KGRetrieval exists, patch its retrieve; otherwise patch pipeline's kg_retrieve
        try:
            pipeline.kg_retriever.retrieve = AsyncMock(
                return_value=[
                    {
                        "statement": "hands contain many small bones (metacarpals & phalanges)",
                        "score": 0.85,
                        "entities": ["hands", "bones"],
                        "source_url": UPLOADED_PDF_PATH,
                        "published_at": None,
                        "credibility": 0.8,
                    }
                ]
            )
        except Exception:
            pipeline.kg_retrieve = AsyncMock(
                return_value=[
                    {
                        "statement": "hands contain many small bones (metacarpals & phalanges)",
                        "score": 0.85,
                        "entities": ["hands", "bones"],
                        "source_url": UPLOADED_PDF_PATH,
                        "published_at": None,
                        "credibility": 0.8,
                    }
                ]
            )

        # Patch KG ingest so we don't call Neo4j
        pipeline.kg_ingest.ingest_triples = AsyncMock(return_value=1)

    # Run the pipeline
    print("Running CorrectivePipeline with claim:")
    print("  ", CLAIM)
    print()
    out = await pipeline.run(CLAIM, domain="health", failed_entities=["hands", "feet", "bones"], top_k=5)

    # Print structured output
    print("\n=== PIPELINE OUTPUT ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))

    # Validate output structure
    assert "status" in out, "Output should have status field"
    assert "facts" in out, "Output should have facts field"
    assert isinstance(out["facts"], list), "facts should be a list"
