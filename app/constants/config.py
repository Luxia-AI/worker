"""
Application configuration constants.
Centralized settings for models, domains, API endpoints, and pipeline thresholds.
"""

import os
from enum import Enum

from app.config.trusted_domains import TRUSTED_ROOT_DOMAINS

# ============================================================================
# VALIDATION & VERDICT STATE ENUMS
# ============================================================================


class ValidationState(Enum):
    """Evidence validation state for domain trust resolution."""

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    PENDING_DOMAIN_TRUST = "pending_domain_trust"


class VerdictState(Enum):
    """Verdict state: confirms confidence level and domain trust timing."""

    CONFIRMED = "confirmed"  # Domain was trusted at verdict time
    PROVISIONAL = "provisional"  # Domain approval pending; verdict may change
    REVOKED = "revoked"  # Domain trust was removed after verdict
    EVIDENCE_INSUFFICIENCY = "evidence_insufficiency"  # Insufficient evidence coverage for multi-part claims


# ============================================================================
# PIPELINE THRESHOLDS & SETTINGS
# ============================================================================

# Maximum reinforcement rounds in corrective pipeline
PIPELINE_MAX_ROUNDS = 3

# Confidence threshold for stopping search: if top evidence >= this, stop searching
# OPTIMIZED: Raised from 0.65 to 0.70 to require better evidence but stop sooner
# when good evidence is found (reduces unnecessary queries)
PIPELINE_CONF_THRESHOLD = 0.70

# Minimum number of new URLs required to continue reinforcement
PIPELINE_MIN_NEW_URLS = 2

# Maximum search queries to execute before stopping (even if threshold not met)
# Increased to improve subclaim coverage (keep queries direct to claim points)
PIPELINE_MAX_SEARCH_QUERIES = 6

# Maximum URLs to scrape per search query (limits extraction cost)
PIPELINE_MAX_URLS_PER_QUERY = 3

# Raw retrieval breadth before ranking (per query, VDB/KG retrieval stage)
PIPELINE_RETRIEVAL_TOP_K = 40

# Minimum candidate target entering ranking (retrieval permissive, ranking strict)
PIPELINE_MIN_RANK_CANDIDATES = 5

# ============================================================================
# RETRIEVAL FILTERS
# ============================================================================

# Minimum acceptable VDB similarity score (cosine) for evidence retention
# Lowered slightly to improve recall on longer, compositional health claims.
VDB_MIN_SCORE = 0.28

# Backfill floor used when strict score threshold yields too few candidates
VDB_BACKFILL_MIN_SCORE = 0.20

# Lexical (FTS5) database path for BM25 retrieval
LEXICAL_DB_PATH = "worker/data/lexical.db"

# Maximum BM25 candidates to retrieve per query
LEXICAL_BM25_LIMIT = 50

# ============================================================================
# LLM MODEL SETTINGS
# ============================================================================

# Default LLM model for Groq service (Llama 3.1 Instant)
LLM_MODEL_NAME = "llama-3.1-8b-instant"

# Temperature for LLM calls (lower = more deterministic)
LLM_TEMPERATURE = 0.2

# Temperature for verdict generation (lower for consistent claim breakdown)
# Set to 0.0 for maximum determinism in claim segmentation
LLM_TEMPERATURE_VERDICT = 0.0

# Per-call output caps to protect Groq TPM and preserve verdict generation headroom.
LLM_MAX_TOKENS_DEFAULT = int(os.getenv("LLM_MAX_TOKENS_DEFAULT", "384"))
LLM_MAX_TOKENS_ENTITY_EXTRACTION = int(os.getenv("LLM_MAX_TOKENS_ENTITY_EXTRACTION", "320"))
LLM_MAX_TOKENS_RELATION_EXTRACTION = int(os.getenv("LLM_MAX_TOKENS_RELATION_EXTRACTION", "320"))
LLM_MAX_TOKENS_FACT_EXTRACTION = int(os.getenv("LLM_MAX_TOKENS_FACT_EXTRACTION", "512"))
LLM_MAX_TOKENS_QUERY_REFORMULATION = int(os.getenv("LLM_MAX_TOKENS_QUERY_REFORMULATION", "256"))
LLM_MAX_TOKENS_VERDICT_GENERATION = int(os.getenv("LLM_MAX_TOKENS_VERDICT_GENERATION", "512"))

# Max tokens for reinforcement query generation
LLM_MAX_TOKENS_REINFORCEMENT = 300

# ============================================================================
# EMBEDDING MODEL SETTINGS
# ============================================================================

# Production embedding model: MUST match Pinecone index dimension
# Current Pinecone index: 1536 dimensions
# Option 1: text-embedding-3-small (OpenAI) - 1536 dims - requires API
# Option 2: Recreate Pinecone index with new dimension
# For now, using all-MiniLM-L6-v2 (384 dims) - REQUIRES PINECONE INDEX RECREATION
EMBEDDING_MODEL_NAME_PROD = "sentence-transformers/all-MiniLM-L6-v2"

# Test embedding model: lightweight, fast downloads for testing
EMBEDDING_MODEL_NAME_TEST = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# TRUSTED DOMAINS FOR MEDICAL/SCIENTIFIC SOURCES
# ============================================================================

# Canonical trusted domains. Keep aliases for backward compatibility.
TRUSTED_DOMAINS_AUTHORITY = set(TRUSTED_ROOT_DOMAINS)
TRUSTED_DOMAINS_EDU_GOV: set[str] = set()
TRUSTED_DOMAINS_NEWS: set[str] = set()
TRUSTED_DOMAINS = set(TRUSTED_ROOT_DOMAINS)

# ============================================================================
# GOOGLE CUSTOM SEARCH ENGINE SETTINGS
# ============================================================================

# Google CSE API endpoint
GOOGLE_CSE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1?" "key={key}&cx={cse}&q={query}"

# Request timeout for Google CSE calls (seconds)
GOOGLE_CSE_TIMEOUT = 12

# ============================================================================
# CREDIBILITY SCORING THRESHOLDS
# ============================================================================

# Credibility score for very high-authority sources
CREDIBILITY_AUTHORITY = 0.95

# Credibility score for government/educational sources
CREDIBILITY_EDU_GOV = 0.75

# Credibility score for news/press/blog sources
CREDIBILITY_NEWS = 0.40

# Default credibility for unknown sources
CREDIBILITY_DEFAULT = 0.5

# ============================================================================
# RECENCY DECAY SETTINGS
# ============================================================================

# Half-life for recency decay in days
RECENCY_HALF_LIFE_DAYS = 365.0

# ============================================================================
# RANKING WEIGHTS
# ============================================================================

# Default weights for hybrid ranking (must sum to 1.0 for interpretability)
# Tuned for maximum accuracy with balanced KG contribution
RANKING_WEIGHTS = {
    "w_semantic": 0.31,  # prioritize direct semantic evidence from VDB/web facts
    "w_kg": 0.12,  # increase KG signal so structured evidence can surface in top-k
    "w_entity": 0.20,  # entity overlap with query (reduced but still significant)
    "w_claim_overlap": 0.15,  # lexical overlap between claim text and evidence statement
    "w_recency": 0.05,  # publication recency (slight boost for fresh evidence)
    "w_credibility": 0.17,  # source credibility (balanced with semantic)
}

# Minimum final score floor when both semantic and KG are zero but credibility high
RANKING_MIN_SCORE_FLOOR = 0.2

# Minimum credibility threshold to trigger min score floor
RANKING_MIN_CREDIBILITY_THRESHOLD = 0.9

# Minimum lexical overlap between claim text and evidence statement to keep candidate
RANKING_MIN_CLAIM_OVERLAP = 0.15

# ============================================================================
# TRUST-RANKING GRADE THRESHOLDS
# ============================================================================

# Grade boundaries (final_score ranges for letter grades A+ through F)
# Based on: semantic similarity + credibility + entity overlap + recency
GRADE_THRESHOLDS = {
    "A_PLUS": 0.90,  # Excellent: High similarity, high credibility, entity match
    "A": 0.80,  # Very Good
    "B": 0.70,  # Good
    "C": 0.60,  # Fair
    "D": 0.50,  # Poor
    "F": 0.0,  # Fail (below 0.50)
}

# Semantic similarity thresholds for confidence filtering
SEMANTIC_THRESHOLD_HIGH = 0.90  # A+ candidate
SEMANTIC_THRESHOLD_GOOD = 0.75  # A/B candidate
SEMANTIC_THRESHOLD_FAIR = 0.60  # C/D candidate
SEMANTIC_THRESHOLD_MIN = 0.40  # Minimum acceptable (below = no grade)
