"""
Corrective Retrieval Pipeline: Main orchestrator.

QUOTA-OPTIMIZED ARCHITECTURE (One Query at a Time):
    1. Embedding: Embed claim and extract entities
    2. Retrieval: Check VDB + KG for existing evidence FIRST
    3. Ranking: If trust score >= threshold, DONE (no web search needed)
    4. Query Generation: Generate search queries (1 LLM call)
    5. Incremental Search: For each query (one at a time):
       a. Execute single search API call
       b. Scrape URLs from that query
       c. Extract facts (LLM call)
       d. Ingest to VDB + KG
       e. Re-rank with new evidence
       f. If threshold met → STOP (save remaining quota)
       g. If not → continue to next query
    6. Return results

This approach MAXIMIZES quota efficiency by:
    - Using cached evidence when available (no API calls)
    - Running ONE search query at a time
    - Processing each query's results completely before deciding to continue
    - Stopping immediately when threshold is reached
    - Never wasting API calls on unused queries
"""

import uuid
from typing import Any, Dict, List, Optional

from app.constants.config import (
    PIPELINE_CONF_THRESHOLD,
    PIPELINE_MAX_ROUNDS,
    PIPELINE_MAX_SEARCH_QUERIES,
    PIPELINE_MAX_URLS_PER_QUERY,
    PIPELINE_MIN_NEW_URLS,
    PIPELINE_RETRIEVAL_TOP_K,
)
from app.core.logger import get_logger
from app.services.corrective.entity_extractor import EntityExtractor
from app.services.corrective.fact_extractor import FactExtractor
from app.services.corrective.pipeline.extraction_phase import extract_all, scrape_pages
from app.services.corrective.pipeline.ingestion_phase import ingest_facts_and_triples
from app.services.corrective.pipeline.ranking_phase import rank_candidates
from app.services.corrective.pipeline.retrieval_phase import retrieve_candidates
from app.services.corrective.relation_extractor import RelationExtractor
from app.services.corrective.scraper import Scraper
from app.services.corrective.trusted_search import TrustedSearch
from app.services.kg.kg_ingest import KGIngest
from app.services.kg.kg_retrieval import KGRetrieval
from app.services.kg.neo4j_client import Neo4jClient
from app.services.llms.hybrid_service import reset_groq_counter
from app.services.logging.log_handler import LogManagerHandler
from app.services.ranking.trust_ranker import EvidenceItem, TrustRankingModule
from app.services.retrieval.lexical_index import LexicalIndex
from app.services.retrieval.metadata_enricher import TopicClassifier
from app.services.vdb.vdb_ingest import VDBIngest
from app.services.vdb.vdb_retrieval import VDBRetrieval
from app.services.verdict.verdict_generator import VerdictGenerator

logger = get_logger(__name__)


class CorrectivePipeline:
    """
    Corrective Retrieval Pipeline with integrated hybrid ranking.

    Orchestrates:
        1. Query reformulation (TrustedSearch -> LLM)
        2. Trusted Search on whitelisted domains
        3. Page scraping and content extraction
        4. Fact/Entity/Relation extraction (LLM-powered)
        5. Ingestion to VDB (Pinecone) and KG (Neo4j)
        6. Retrieval: semantic (VDB) + structural (KG)
        7. Hybrid ranking combining 5 scoring signals
        8. Reinforcement loop for low-confidence results
    """

    MAX_ROUNDS = PIPELINE_MAX_ROUNDS
    CONF_THRESHOLD = PIPELINE_CONF_THRESHOLD
    MIN_NEW_URLS = PIPELINE_MIN_NEW_URLS
    MAX_SEARCH_QUERIES = PIPELINE_MAX_SEARCH_QUERIES
    MAX_URLS_PER_QUERY = PIPELINE_MAX_URLS_PER_QUERY

    def __init__(self) -> None:
        # Search and scraping
        self.search_agent = TrustedSearch()
        self.scraper = Scraper()

        # Extraction
        self.fact_extractor = FactExtractor()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()

        # Storage and retrieval
        self.vdb_ingest = VDBIngest()  # Pinecone
        self.vdb_retriever = VDBRetrieval()
        self.kg_client = Neo4jClient()
        self.kg_ingest = KGIngest()
        self.kg_retriever = KGRetrieval()
        self.lexical_index = LexicalIndex()
        self.topic_classifier = TopicClassifier()

        # Verdict generation (RAG phase)
        self.verdict_generator = VerdictGenerator()

        # Trust ranking
        self.trust_ranker = TrustRankingModule()

        # Logging system
        self.log_manager = LogManagerHandler._log_manager

    async def run(
        self,
        post_text: str,
        domain: str,
        failed_entities: Optional[List[str]] = None,
        round_id: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Execute the OPTIMIZED corrective retrieval pipeline.

        RETRIEVAL-FIRST APPROACH:
        1. Extract entities from claim (single LLM call)
        2. Retrieve existing evidence from VDB + KG
        3. Rank existing evidence
        4. IF trust score >= threshold: DONE (no web search, no extra LLM calls!)
        5. ELSE: Search incrementally, process in batches until threshold reached

        This minimizes Groq API calls by using cached knowledge when available.
        """
        round_id = round_id or str(uuid.uuid4())
        failed_entities = failed_entities or []

        # Reset Groq call counter for this new request
        reset_groq_counter()

        logger.info(f"[CorrectivePipeline:{round_id}] Start for post: {post_text}")

        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Pipeline started (retrieval-first): {post_text[:100]}...",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={"domain": domain},
            )

        # ====================================================================
        # PHASE 1: Extract entities from claim (1 LLM call)
        # ====================================================================
        claim_entities = await self._extract_claim_entities(post_text, round_id)
        logger.info(f"[CorrectivePipeline:{round_id}] Claim entities: {claim_entities}")

        # ====================================================================
        # PHASE 2: Retrieve existing evidence from VDB + KG (NO LLM calls)
        # ====================================================================
        claim_topics, topic_conf = await self._infer_claim_topics(post_text, claim_entities)
        logger.info(f"[CorrectivePipeline:{round_id}] Claim topics: {claim_topics} (conf={topic_conf:.2f})")

        retrieval_queries = self._build_retrieval_queries(post_text)
        raw_retrieval_top_k = max(PIPELINE_RETRIEVAL_TOP_K, top_k * 3)
        dedup_sem, kg_candidates = await retrieve_candidates(
            self.vdb_retriever,
            self.kg_retriever,
            retrieval_queries,
            claim_entities,
            raw_retrieval_top_k,
            round_id,
            claim_topics,
            self.lexical_index,
            self.log_manager,
            post_text,
        )

        logger.info(
            f"[CorrectivePipeline:{round_id}] Retrieved {len(dedup_sem)} VDB + {len(kg_candidates)} KG candidates"
        )

        # ====================================================================
        # PHASE 3: Rank existing evidence (NO LLM calls)
        # ====================================================================
        top_ranked = []
        if dedup_sem or kg_candidates:
            top_ranked = await rank_candidates(
                dedup_sem,
                kg_candidates,
                claim_entities,
                post_text,
                top_k,
                round_id,
                self.log_manager,
            )

        # Compute trust scores for evidence

        top_ranked_evidence = [
            EvidenceItem(
                statement=item.get("statement", item.get("fact", "")),
                semantic_score=item.get("sem_score", item.get("final_score", item.get("score", 0.0))),
                source_url=item.get("source_url", item.get("source", "")),
                published_at=item.get("published_at", item.get("publish_date")),
                trust=item.get("final_score", item.get("score", 0.0)),  # Use final_score as trust
                stance="neutral",  # Default stance
                score_components={
                    "semantic": item.get("sem_score", item.get("final_score", item.get("score", 0.0))),
                    "source": 0.5,  # Default source credibility
                    "recency": 0.5,  # Default recency
                    "stance_raw": 0.0,  # Neutral stance
                    "stance_mapped": 0.5,  # Neutral mapped to 0.5
                    "trust": item.get("final_score", item.get("score", 0.0)),
                },
            )
            for item in top_ranked
        ]

        # Use adaptive trust policy for multi-part claims
        adaptive_trust = self.trust_ranker.compute_adaptive_post_trust(post_text, top_ranked_evidence, top_k)
        is_sufficient = adaptive_trust["is_sufficient"]

        logger.info(
            f"[CorrectivePipeline:{round_id}] Adaptive trust: sufficient={is_sufficient}, "
            f"coverage={adaptive_trust['coverage']:.2f}, diversity={adaptive_trust['diversity']:.2f}"
        )

        if is_sufficient:
            logger.info(
                f"[CorrectivePipeline:{round_id}] Adaptive trust sufficient. "
                "Skipping web search - using cached evidence!"
            )

            verdict_result = await self.verdict_generator.generate_verdict(
                claim=post_text,
                ranked_evidence=top_ranked,
                top_k=top_k,
                used_web_search=False,
                cache_sufficient=True,
            )

            logger.info(
                f"[CorrectivePipeline:{round_id}] Verdict (cache-sufficient): {verdict_result['verdict']} "
                f"(confidence: {verdict_result['confidence']:.2f})"
            )
            return {
                "round_id": round_id,
                "status": "completed_from_cache",
                "facts": [],  # No new facts extracted
                "triples": [],
                "queries": [post_text],
                "semantic_candidates_count": len(dedup_sem),
                "kg_candidates_count": len(kg_candidates),
                "ranked": top_ranked,
                "used_web_search": False,
                "data_source": "CACHE",
                "trust_threshold": "adaptive",  # Now using adaptive policy
                "trust_threshold_met": True,
                "initial_top_score": adaptive_trust["trust_post"],
                "ranking_top_score": (top_ranked[0].get("final_score", 0.0) if top_ranked else 0.0),
                "ranking_avg_score": (
                    sum(float(r.get("final_score", 0.0) or 0.0) for r in top_ranked[:5]) / max(1, len(top_ranked[:5]))
                    if top_ranked
                    else 0.0
                ),
                "trust_post": adaptive_trust["trust_post"],
                "trust_grade": "adaptive",  # Adaptive grading
                "agreement_ratio": adaptive_trust["agreement"],
                "coverage": adaptive_trust["coverage"],
                "diversity": adaptive_trust["diversity"],
                "num_subclaims": adaptive_trust["num_subclaims"],
                "verdict": verdict_result,
                "cache_sufficient": True,
            }

        # ====================================================================
        # PHASE 4: Quota-Optimized Incremental Search (ONE QUERY AT A TIME)
        # ====================================================================
        logger.info(
            f"[CorrectivePipeline:{round_id}] Evidence insufficient (coverage={adaptive_trust['coverage']:.2f}, "
            f"diversity={adaptive_trust['diversity']:.2f}), starting quota-optimized search..."
        )

        # Generate all search queries upfront (1 LLM call only)
        raw_subclaims = self.trust_ranker.adaptive_policy.decompose_claim(post_text)
        merged_subclaims = self.search_agent.merge_subclaims(raw_subclaims)
        queries = await self.search_agent.generate_search_queries(
            post_text,
            failed_entities,
            max_queries=self.MAX_SEARCH_QUERIES,
            subclaims=merged_subclaims,
            entities=claim_entities,
        )

        if not queries:
            logger.warning(f"[CorrectivePipeline:{round_id}] No search queries generated")

            # Generate verdict with whatever evidence we have
            verdict_result = await self.verdict_generator.generate_verdict(
                claim=post_text,
                ranked_evidence=top_ranked,
                top_k=top_k,
                used_web_search=False,
            )

            return {
                "round_id": round_id,
                "status": "no_queries_generated",
                "facts": [],
                "triples": [],
                "queries": [],
                "semantic_candidates_count": len(dedup_sem),
                "kg_candidates_count": len(kg_candidates),
                "ranked": top_ranked,
                "used_web_search": False,
                "data_source": "CACHE",
                "search_api_calls": 0,
                "trust_threshold": "adaptive",
                "trust_threshold_met": False,
                "initial_top_score": adaptive_trust["trust_post"],
                "ranking_top_score": (top_ranked[0].get("final_score", 0.0) if top_ranked else 0.0),
                "ranking_avg_score": (
                    sum(float(r.get("final_score", 0.0) or 0.0) for r in top_ranked[:5]) / max(1, len(top_ranked[:5]))
                    if top_ranked
                    else 0.0
                ),
                "coverage": adaptive_trust["coverage"],
                "diversity": adaptive_trust["diversity"],
                "num_subclaims": adaptive_trust["num_subclaims"],
                "verdict": verdict_result,
            }

        # Get already-processed URLs from VDB to avoid re-scraping
        already_processed_urls = self.vdb_ingest.get_processed_urls()
        logger.info(
            f"[CorrectivePipeline:{round_id}] Found {len(already_processed_urls)} " "already-processed URLs in VDB"
        )

        # Process ONE QUERY AT A TIME for maximum quota efficiency
        all_facts: List[Dict[str, Any]] = []
        all_triples: List[Dict[str, Any]] = []
        all_entities = list(claim_entities)
        processed_urls: List[str] = []
        skipped_urls: List[str] = []
        search_api_calls = 0
        queries_executed: List[str] = []

        for query_idx, query in enumerate(queries):
            # OPTIMIZATION: Hard limit on search queries to prevent runaway searches
            if search_api_calls >= self.MAX_SEARCH_QUERIES:
                logger.info(
                    f"[CorrectivePipeline:{round_id}] Reached MAX_SEARCH_QUERIES ({self.MAX_SEARCH_QUERIES}), "
                    f"stopping to conserve quota. {len(queries) - query_idx} queries unused."
                )
                break

            logger.info(f"[CorrectivePipeline:{round_id}] === Query {query_idx + 1}/{len(queries)} ===")
            logger.info(f"[CorrectivePipeline:{round_id}] Executing: '{query}'")

            # Step 1: Execute SINGLE search API call
            search_api_calls += 1
            query_urls = await self.search_agent.execute_single_query(query)
            queries_executed.append(query)

            if not query_urls:
                logger.info(
                    f"[CorrectivePipeline:{round_id}] Query {query_idx + 1} returned no URLs, " "trying next query..."
                )
                continue

            logger.info(f"[CorrectivePipeline:{round_id}] Query {query_idx + 1} returned " f"{len(query_urls)} URLs")

            # Filter out already-processed URLs to avoid re-scraping
            new_urls = [url for url in query_urls if url not in already_processed_urls and url not in processed_urls]
            skipped_this_query = len(query_urls) - len(new_urls)

            if skipped_this_query > 0:
                skipped_urls.extend([url for url in query_urls if url in already_processed_urls])
                logger.info(
                    f"[CorrectivePipeline:{round_id}] Filtered {skipped_this_query} already-processed URLs, "
                    f"{len(new_urls)} new URLs to scrape"
                )

            if not new_urls:
                logger.info(
                    f"[CorrectivePipeline:{round_id}] All URLs from query {query_idx + 1} already processed, "
                    "trying next query..."
                )
                continue

            # Step 2: Scrape only NEW URLs from this query
            if len(new_urls) > self.MAX_URLS_PER_QUERY:
                logger.info(
                    f"[CorrectivePipeline:{round_id}] Capping URLs to {self.MAX_URLS_PER_QUERY} "
                    f"from {len(new_urls)} candidates for this query"
                )
                new_urls = new_urls[: self.MAX_URLS_PER_QUERY]
            scraped_pages = await scrape_pages(self.scraper, new_urls, round_id, self.log_manager)
            processed_urls.extend(new_urls)
            # Add to already_processed set to avoid re-scraping in subsequent queries
            already_processed_urls.update(new_urls)

            if not scraped_pages:
                logger.info(f"[CorrectivePipeline:{round_id}] No content scraped from query {query_idx + 1}")
                continue

            # Step 3: Extract facts, entities, relations
            query_facts, query_entities, query_triples = await extract_all(
                self.fact_extractor,
                self.entity_extractor,
                self.relation_extractor,
                scraped_pages,
                round_id,
                self.log_manager,
            )

            if not query_facts:
                logger.info(f"[CorrectivePipeline:{round_id}] No facts extracted from query {query_idx + 1}")
                continue

            # Accumulate results
            all_facts.extend(query_facts)
            all_entities.extend(query_entities)
            all_triples.extend(query_triples)

            logger.info(
                f"[CorrectivePipeline:{round_id}] Query {query_idx + 1} extracted "
                f"{len(query_facts)} facts, {len(query_triples)} triples"
            )

            # Step 4: Ingest to VDB and KG immediately
            await ingest_facts_and_triples(
                self.vdb_ingest,
                self.kg_ingest,
                query_facts,
                query_triples,
                round_id,
                self.log_manager,
            )

            # Step 5: Re-retrieve and re-rank with new evidence
            # Keep retrieval anchored to claim entities to avoid topic drift from scraped-page artifacts.
            retrieval_entities = list(set(claim_entities + failed_entities))
            dedup_sem, kg_candidates = await retrieve_candidates(
                self.vdb_retriever,
                self.kg_retriever,
                queries_executed,
                retrieval_entities,
                raw_retrieval_top_k,
                round_id,
                claim_topics,
                self.lexical_index,
                self.log_manager,
                post_text,
            )

            top_ranked = await rank_candidates(
                dedup_sem,
                kg_candidates,
                retrieval_entities,
                post_text,
                top_k,
                round_id,
                self.log_manager,
            )

            # Compute trust scores
            top_ranked_evidence = [
                EvidenceItem(
                    statement=item.get("statement", item.get("fact", "")),
                    semantic_score=item.get("sem_score", item.get("final_score", item.get("score", 0.0))),
                    source_url=item.get("source_url", item.get("source", "")),
                    published_at=item.get("published_at", item.get("publish_date")),
                    trust=item.get("final_score", item.get("score", 0.0)),  # Use final_score as trust
                    stance="neutral",  # Default stance
                    score_components={
                        "semantic": item.get("sem_score", item.get("final_score", item.get("score", 0.0))),
                        "source": 0.5,  # Default source credibility
                        "recency": 0.5,  # Default recency
                        "stance_raw": 0.0,  # Neutral stance
                        "stance_mapped": 0.5,  # Neutral mapped to 0.5
                        "trust": item.get("final_score", item.get("score", 0.0)),
                    },
                )
                for item in top_ranked
            ]

            adaptive_trust = self.trust_ranker.compute_adaptive_post_trust(post_text, top_ranked_evidence, top_k)
            is_sufficient = adaptive_trust["is_sufficient"]

            logger.info(
                f"[CorrectivePipeline:{round_id}] After query {query_idx + 1}: "
                f"adaptive_sufficient={is_sufficient}, "
                f"trust_post={adaptive_trust['trust_post']:.3f}, "
                f"total_facts={len(all_facts)}, "
                f"search_calls={search_api_calls}"
            )

            # Step 6: Check if adaptive trust sufficient - STOP to save quota!
            if is_sufficient:
                remaining_queries = len(queries) - query_idx - 1
                logger.info(
                    f"[CorrectivePipeline:{round_id}] ✅ ADAPTIVE THRESHOLD MET after query {query_idx + 1}! "
                    f"Saved {remaining_queries} search API calls!"
                )
                break

        # Log final statistics
        logger.info(
            f"[CorrectivePipeline:{round_id}] Completed: {len(all_facts)} facts, "
            f"{len(top_ranked)} ranked, {search_api_calls}/{len(queries)} queries used, "
            f"{len(processed_urls)} URLs processed, {len(skipped_urls)} URLs skipped (already processed)"
        )

        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Pipeline completed: {len(all_facts)} facts, {search_api_calls} API calls",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={
                    "facts_count": len(all_facts),
                    "triples_count": len(all_triples),
                    "ranked_count": len(top_ranked),
                    "top_score": top_ranked[0]["final_score"] if top_ranked else 0.0,
                    "urls_processed": len(processed_urls),
                    "urls_skipped": len(skipped_urls),
                    "search_api_calls": search_api_calls,
                    "queries_total": len(queries),
                    "queries_saved": len(queries) - search_api_calls,
                },
            )

        # Determine final status
        final_status = "completed"
        if not all_facts and search_api_calls > 0:
            final_status = "no_facts_extracted"
        elif search_api_calls == 0:
            final_status = "no_search_results"

        # ====================================================================
        # PHASE 7: RAG Generation - Generate Final Verdict
        # ====================================================================
        logger.info(f"[CorrectivePipeline:{round_id}] Generating verdict with RAG...")

        # Compute final trust scores
        final_top_ranked_evidence = [
            EvidenceItem(
                statement=item.get("statement", item.get("fact", "")),
                semantic_score=item.get("sem_score", item.get("final_score", item.get("score", 0.0))),
                source_url=item.get("source_url", item.get("source", "")),
                published_at=item.get("published_at", item.get("publish_date")),
                trust=item.get("final_score", item.get("score", 0.0)),  # Use final_score as trust
                stance="neutral",  # Default stance
                score_components={
                    "semantic": item.get("sem_score", item.get("final_score", item.get("score", 0.0))),
                    "source": 0.5,  # Default source credibility
                    "recency": 0.5,  # Default recency
                    "stance_raw": 0.0,  # Neutral stance
                    "stance_mapped": 0.5,  # Neutral mapped to 0.5
                    "trust": item.get("final_score", item.get("score", 0.0)),
                },
            )
            for item in top_ranked
        ]

        final_trust_post = self.trust_ranker.compute_post_trust(final_top_ranked_evidence, top_k)
        final_trust_score = final_trust_post["trust_post"]

        verdict_result = await self.verdict_generator.generate_verdict(
            claim=post_text,
            ranked_evidence=top_ranked,
            top_k=top_k,
            used_web_search=search_api_calls > 0,
        )

        logger.info(
            f"[CorrectivePipeline:{round_id}] Final verdict: {verdict_result['verdict']} "
            f"(confidence: {verdict_result['confidence']:.2f})"
        )

        if self.log_manager:
            await self.log_manager.add_log(
                level="INFO",
                message=f"Verdict generated: {verdict_result['verdict']}",
                module=__name__,
                request_id=f"claim-{round_id}",
                round_id=round_id,
                context={
                    "verdict": verdict_result["verdict"],
                    "confidence": verdict_result["confidence"],
                    "rationale": verdict_result.get("rationale", ""),
                },
            )

        return {
            "round_id": round_id,
            "status": final_status,
            "facts": all_facts,
            "triples": all_triples,
            "queries": queries_executed,
            "queries_total": len(queries),
            "semantic_candidates_count": len(dedup_sem),
            "kg_candidates_count": len(kg_candidates),
            "ranked": top_ranked,
            "used_web_search": search_api_calls > 0,
            "data_source": "WEB_SEARCH" if search_api_calls > 0 else "CACHE",
            "search_api_calls": search_api_calls,
            "search_api_calls_saved": len(queries) - search_api_calls,
            "urls_processed": len(processed_urls),
            "urls_skipped_already_processed": len(skipped_urls),
            "trust_threshold": self.CONF_THRESHOLD,
            "trust_threshold_met": final_trust_score >= self.CONF_THRESHOLD,
            "initial_top_score": final_trust_score,
            "ranking_top_score": (top_ranked[0].get("final_score", 0.0) if top_ranked else 0.0),
            "ranking_avg_score": (
                sum(float(r.get("final_score", 0.0) or 0.0) for r in top_ranked[:5]) / max(1, len(top_ranked[:5]))
                if top_ranked
                else 0.0
            ),
            "trust_post": final_trust_score,
            "trust_grade": final_trust_post.get("grade", "D"),
            "agreement_ratio": final_trust_post.get("agreement_ratio", 0.0),
            "verdict": verdict_result,
        }

    async def _extract_claim_entities(self, claim: str, round_id: str) -> List[str]:
        """Extract entities from the claim text using LLM (1 call) with a deterministic fallback."""
        import re

        def _fallback_entities(text: str) -> List[str]:
            stop = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "to",
                "for",
                "of",
                "in",
                "on",
                "with",
                "by",
                "at",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "over",
                "under",
                "about",
                "around",
                "average",
                "per",
                "times",
            }
            junk = {
                "according",
                "significantly",
                "significant",
                "every",
                "morning",
                "drinking",
                "improves",
                "improve",
                "research",
                "medical",
            }
            tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{2,}\b", text)]
            tokens = [t for t in tokens if t not in stop]
            tokens = [t for t in tokens if t not in junk]
            freq: dict[str, int] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            bigrams: List[str] = []
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                if w1 in stop or w2 in stop:
                    continue
                bigrams.append(f"{w1} {w2}")
            ranked_unigrams = sorted(freq.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
            unigram_entities = [w for w, _ in ranked_unigrams[:6]]
            bigram_entities: List[str] = []
            for bg in bigrams:
                if any(part in junk for part in bg.split()):
                    continue
                if bg.split()[0] in unigram_entities and bg.split()[1] in unigram_entities:
                    bigram_entities.append(bg)
                if len(bigram_entities) >= 4:
                    break
            out: List[str] = []
            for item in bigram_entities + unigram_entities:
                if item not in out:
                    out.append(item)
            return out[:10]

        llm_entities: List[str] = []
        try:
            # Use entity extractor on a synthetic fact
            synthetic_facts = [{"statement": claim, "source_url": "claim", "fact_id": "claim_0"}]
            annotated = await self.entity_extractor.annotate_entities(synthetic_facts)
            if annotated:
                entities = annotated[0].get("entities", []) or []
                # Filter generic/placeholder labels
                generic = {
                    "anatomical terms",
                    "medical terms",
                    "health terms",
                    "general",
                    "unknown",
                    "misc",
                }
                cleaned = [e for e in entities if isinstance(e, str) and e.strip()]
                cleaned = [e for e in cleaned if e.strip().lower() not in generic]
                llm_entities = cleaned
        except Exception as e:
            logger.warning(f"[CorrectivePipeline:{round_id}] Entity extraction from claim failed: {e}")

        fallback = _fallback_entities(claim)
        merged = []
        for e in llm_entities + fallback:
            if e and e not in merged:
                merged.append(e)

        if fallback and fallback != llm_entities:
            logger.info(f"[CorrectivePipeline:{round_id}] Fallback entities from claim: {fallback}")
            if self.log_manager:
                await self.log_manager.add_log(
                    level="INFO",
                    message=f"Fallback entities from claim: {fallback}",
                    module=__name__,
                    request_id=f"claim-{round_id}",
                    round_id=round_id,
                    context={"fallback_entities": fallback},
                )
        return merged or fallback

    def _build_retrieval_queries(self, claim: str) -> List[str]:
        """
        Build retrieval queries using the full claim plus a few decomposed subclaims.
        This increases recall for multi-part claims while keeping VDB calls bounded.
        """
        queries = [claim.strip()] if claim and claim.strip() else []
        subclaims = self.trust_ranker.adaptive_policy.decompose_claim(claim)
        # Limit to top 3 subclaims to avoid query explosion
        for sub in subclaims[:3]:
            s = sub.strip()
            if s and s not in queries:
                queries.append(s)
        return queries

    async def _infer_claim_topics(self, claim: str, entities: List[str]) -> tuple[list[str], float]:
        subclaims = self.trust_ranker.adaptive_policy.decompose_claim(claim)
        topics: set[str] = set()
        best_conf = 0.0
        for sub in subclaims:
            t, conf = await self.topic_classifier.classify(sub, entities, None)
            for item in t:
                topics.add(item)
            if conf > best_conf:
                best_conf = conf
        return list(topics), best_conf


__all__ = ["CorrectivePipeline"]
