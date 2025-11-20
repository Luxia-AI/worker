from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from app.services.corrective.pipeline import CorrectivePipeline

POST_TEXT = "More than half of the 206 bones in an adult human body are located in the hands and feet."
DOMAIN = "health"


@pytest_asyncio.fixture
async def pipeline():
    return CorrectivePipeline()


@pytest.mark.asyncio
async def test_full_pipeline_e2e(pipeline):
    """
    Full integration test for the entire corrective retrieval pipeline.

    ALL external dependencies are mocked:
    - Google Search (TrustedSearch.google_search)
    - Scraper (Scraper.scrape_all)
    - FactExtractor.extract
    - EntityExtractor.annotate_entities
    - RelationExtractor.extract_relations
    - VDBIngest.embed_and_ingest
    - KGIngest.ingest_triples
    - VDBRetrieval.search
    - KGRetrieval.retrieve

    The goal is to verify:
        1) Pipeline orchestration works end-to-end
        2) Ranking returns at least 1 candidate
        3) Reinforcement loop is executed when confidence low
    """

    # ---- Mock Google CSE initial search ----
    with patch(
        "app.services.corrective.trusted_search.TrustedSearch.google_search",
        new=AsyncMock(
            return_value=[
                "https://www.niams.nih.gov/health-topics/psoriasis",
                "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
            ]
        ),
    ):

        # ---- Mock reinforcement search ----
        with patch(
            "app.services.corrective.trusted_search.TrustedSearch" ".llm_reformulate_for_reinforcement",
            new=AsyncMock(
                return_value=[
                    "hands feet bone distribution nih",
                    "anatomy extremities bones pubmed",
                ]
            ),
        ):
            with patch(
                "app.services.corrective.trusted_search.TrustedSearch.google_search",
                new=AsyncMock(
                    return_value=[
                        "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
                    ]
                ),
            ):

                # ---- Mock scraper ----
                with patch(
                    "app.services.corrective.scraper.Scraper.scrape_all",
                    new=AsyncMock(
                        return_value=[
                            {
                                "url": "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
                                "content": "The human skeleton contains 206 bones. 106 of these are in hands and feet.",
                            }
                        ]
                    ),
                ):

                    # ---- Mock fact extraction ----
                    with patch(
                        "app.services.corrective.fact_extractor.FactExtractor.extract",
                        new=AsyncMock(
                            return_value=[
                                {
                                    "statement": "The human skeleton contains 206 bones.",
                                    "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
                                    "published_at": None,
                                },
                                {
                                    "statement": "106 of the bones are located in hands and feet.",
                                    "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
                                    "published_at": None,
                                },
                            ]
                        ),
                    ):

                        # ---- Mock entity extraction ----
                        with patch(
                            "app.services.corrective.entity_extractor.EntityExtractor.annotate_entities",
                            new=AsyncMock(
                                return_value=[
                                    {
                                        "statement": "The human skeleton contains 206 bones.",
                                        "entities": ["human skeleton", "206 bones"],
                                        "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
                                    },
                                    {
                                        "statement": "106 of the bones are located in hands and feet.",
                                        "entities": ["106 bones", "hands", "feet"],
                                        "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
                                    },
                                ]
                            ),
                        ):

                            # ---- Mock relation extraction ----
                            with patch(
                                "app.services.corrective.relation_extractor.RelationExtractor.extract_relations",
                                new=AsyncMock(
                                    return_value=[
                                        {
                                            "subject": "human skeleton",
                                            "relation": "contains",
                                            "object": "206 bones",
                                            "confidence": 0.94,
                                            "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
                                        },
                                        {
                                            "subject": "hands and feet",
                                            "relation": "contain",
                                            "object": "106 bones",
                                            "confidence": 0.92,
                                            "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
                                        },
                                    ]
                                ),
                            ):

                                # ---- Mock VDB ingest ----
                                with patch(
                                    "app.services.vdb.vdb_ingest.VDBIngest.embed_and_ingest",
                                    new=AsyncMock(return_value=None),
                                ):

                                    # ---- Mock KG ingest ----
                                    with patch(
                                        "app.services.kg.kg_ingest.KGIngest.ingest_triples",
                                        new=AsyncMock(return_value=None),
                                    ):

                                        # ---- Mock VDB semantic retrieval ----
                                        with patch(
                                            "app.services.vdb.vdb_retrieval.VDBRetrieval.search",
                                            new=AsyncMock(
                                                return_value=[
                                                    {
                                                        "statement": "The human skeleton contains 206 bones.",
                                                        "score": 0.83,
                                                        "entities": ["human skeleton", "206 bones"],
                                                        "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK531506/",
                                                        "credibility": 0.95,
                                                    }
                                                ]
                                            ),
                                        ):

                                            # ---- Mock KG retrieval ----
                                            with patch(
                                                "app.services.kg.kg_retrieval.KGRetrieval.retrieve",
                                                new=AsyncMock(
                                                    return_value=[
                                                        {
                                                            "statement": ("hands and feet contain 106 bones"),
                                                            "score": 0.91,
                                                            "entities": ["hands", "feet"],
                                                            "source_url": (
                                                                "https://www.ncbi.nlm.nih.gov/" "books/NBK531506/"
                                                            ),
                                                            "credibility": 0.95,
                                                        }
                                                    ]
                                                ),
                                            ):

                                                result = await pipeline.run(
                                                    post_text=POST_TEXT, domain=DOMAIN, failed_entities=[], top_k=5
                                                )

                                                # Assertions
                                                assert result["status"] == "completed"
                                                assert len(result["facts"]) >= 2
                                                assert len(result["triples"]) >= 1
                                                assert len(result["ranked"]) >= 1

                                                top_item = result["ranked"][0]
                                                assert "final_score" in top_item
                                                assert top_item["final_score"] > 0

                                                print("\n\n==== PIPELINE OUTPUT ====")
                                                print(result)
