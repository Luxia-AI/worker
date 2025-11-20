"""
Application configuration constants.
Centralized settings for models, domains, API endpoints, and pipeline thresholds.
"""

# ============================================================================
# PIPELINE THRESHOLDS & SETTINGS
# ============================================================================

# Maximum reinforcement rounds in corrective pipeline
PIPELINE_MAX_ROUNDS = 3

# Confidence threshold for reinforcement: if top evidence < this, trigger reinforcement
PIPELINE_CONF_THRESHOLD = 0.70

# Minimum number of new URLs required to continue reinforcement
PIPELINE_MIN_NEW_URLS = 2

# ============================================================================
# LLM MODEL SETTINGS
# ============================================================================

# Default LLM model for Groq service (MoonshotAI's Kimi K2)
LLM_MODEL_NAME = "moonshotai/kimi-k2-instruct"

# Temperature for LLM calls (lower = more deterministic)
LLM_TEMPERATURE = 0.2

# Max tokens for reinforcement query generation
LLM_MAX_TOKENS_REINFORCEMENT = 300

# ============================================================================
# EMBEDDING MODEL SETTINGS
# ============================================================================

# Production embedding model: recommended for RAG search -- strong performance
EMBEDDING_MODEL_NAME_PROD = "sentence-transformers/multilingual-e5-large"

# Test embedding model: lightweight, fast downloads for testing
EMBEDDING_MODEL_NAME_TEST = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# TRUSTED DOMAINS FOR MEDICAL/SCIENTIFIC SOURCES
# ============================================================================

# High-authority medical and government domains
TRUSTED_DOMAINS_AUTHORITY = {
    "who.int",
    "cdc.gov",
    "nih.gov",
    "fda.gov",
    "mayoclinic.org",
    "harvard.edu",
    "nhs.uk",
}

# Government and educational domains
TRUSTED_DOMAINS_EDU_GOV = {".gov", ".edu"}

# News, press, and blog domains (lower credibility tier)
TRUSTED_DOMAINS_NEWS = {"news", "press", "blog", "medium.com"}

# Complete set of trusted domains for CSE filtering
TRUSTED_DOMAINS = {
    "who.int",
    "cdc.gov",
    "nih.gov",
    "fda.gov",
    "nhs.uk",
    "mayoclinic.org",
    "health.harvard.edu",
    "medlineplus.gov",
    "livescience.com",
    "medicalnewstoday.com",
}

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
RANKING_WEIGHTS = {
    "w_semantic": 0.08,  # semantic similarity score
    "w_kg": 0.01,  # knowledge graph score
    "w_entity": 0.59,  # entity overlap with query
    "w_recency": 0.02,  # publication recency
    "w_credibility": 0.30,  # source credibility
}

# Minimum final score floor when both semantic and KG are zero but credibility high
RANKING_MIN_SCORE_FLOOR = 0.2

# Minimum credibility threshold to trigger min score floor
RANKING_MIN_CREDIBILITY_THRESHOLD = 0.9
