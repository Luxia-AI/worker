# Luxia Worker - Biomedical Fact Verification Pipeline

A sophisticated AI-powered fact-checking system that verifies medical and scientific claims against authoritative sources using a **7-phase Corrective Retrieval Pipeline** combined with hybrid ranking algorithms and knowledge graph integration.

## 🎯 Overview

**Luxia Worker** implements an advanced information retrieval and verification system designed for biomedical fact-checking (extensible to other domains). The system:

- 🔍 Searches trusted medical/scientific domains (WHO, CDC, NIH, PubMed, etc.)
- 📄 Extracts facts, entities, and relationships using LLM-powered NLP
- 📊 Stores findings in dual storage systems (Pinecone vector DB + Neo4j knowledge graph)
- ⭐ Ranks evidence using 5-signal hybrid scoring (recency, credibility, semantic similarity, entity match, KG score)
- 🔄 Reinforces low-confidence results through iterative search loops with failed entity targeting

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Endpoints](#-api-endpoints)
- [Pipeline Phases](#-pipeline-phases)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)

## 🚀 Quick Start

### Local Development

```bash
# Clone and setup
git clone https://github.com/Luxia-AI/worker.git
cd worker

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# or
source .venv/bin/activate      # macOS/Linux

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run quality checks
./run.sh all

# Start development server
python main.py
# or
uvicorn app.main:app --reload --port 9000
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# View logs
docker-compose logs -f worker

# Stop services
docker-compose down
```

## 🏗️ Architecture

### 7-Phase Pipeline

```
Input (claim)
    ↓
[1] SEARCH PHASE
    • Query reformulation (LLM)
    • Trusted domain filtering
    • Google CSE search
    ↓
[2] SCRAPING PHASE
    • HTML → Text extraction (Trafilatura)
    • Content deduplication
    ↓
[3] EXTRACTION PHASE
    • Fact extraction (LLM)
    • Entity extraction (LLM)
    • Relation extraction (LLM)
    ↓
[4] INGESTION PHASE
    • VDB ingestion (Pinecone)
    • KG ingestion (Neo4j)
    ↓
[5] RETRIEVAL PHASE
    • Semantic search (VDB)
    • Structural queries (KG)
    ↓
[6] RANKING PHASE
    • 5-signal hybrid scoring:
      1. Credibility (domain authority)
      2. Recency (exponential decay)
      3. Semantic similarity (embedding)
      4. Entity match (extracted entities)
      5. KG structural score
    ↓
[7] REINFORCEMENT PHASE
    • If confidence < THRESHOLD:
      → Collect failed entities
      → Re-search with entity queries
      → Loop up to MAX_ROUNDS
    ↓
Output (ranked evidence with confidence)
```

### Data Flow

```
Claim Input
    ↓
TrustedSearch (LLM query reformulation)
    ↓
Google CSE → Search URLs
    ↓
Trafilatura → Scrape content
    ↓
3x LLM Extraction (Facts, Entities, Relations)
    ↓
┌─────────────────────┐
│   Parallel Ingest   │
├──────────┬──────────┤
│ Pinecone │  Neo4j   │
│  (VDB)   │   (KG)   │
└──────────┴──────────┘
    ↓
┌─────────────────────┐
│ Parallel Retrieval  │
├──────────┬──────────┤
│ Semantic │Structural│
│ Search   │ Queries  │
└──────────┴──────────┘
    ↓
Hybrid Ranking (5 signals)
    ↓
Confidence >= Threshold?
    ├─ YES → Return top-k evidence
    └─ NO → Reinforcement Loop
```

### Services Layer Architecture

```
app/
├── main.py                           # FastAPI app entry point
├── constants/
│   ├── config.py                    # All configuration constants
│   └── llm_prompts.py               # LLM prompt templates
├── core/
│   ├── config.py                    # Pydantic settings loader
│   ├── logger.py                    # Structured logging
│   ├── rate_limit.py                # Rate limiter decorator
│   └── utils.py                     # Helper utilities
├── routers/
│   ├── pinecone.py                  # /worker/search endpoint
│   └── admin.py                     # /admin/logs endpoint
└── services/
    ├── corrective/                  # Core pipeline
    │   ├── pipeline/
    │   │   ├── __init__.py          # CorrectivePipeline orchestrator
    │   │   ├── search_phase.py
    │   │   ├── extraction_phase.py
    │   │   ├── ingestion_phase.py
    │   │   ├── retrieval_phase.py
    │   │   ├── ranking_phase.py
    │   │   └── reinforcement_phase.py
    │   ├── trusted_search.py        # Trusted domain search + LLM reformulation
    │   ├── scraper.py               # Trafilatura wrapper
    │   ├── fact_extractor.py        # LLM fact extraction
    │   ├── entity_extractor.py      # LLM entity extraction
    │   └── relation_extractor.py    # LLM relation extraction
    ├── embedding/
    │   └── model.py                 # Embedding model management
    ├── vdb/
    │   ├── pinecone_client.py       # Pinecone API wrapper
    │   ├── vdb_ingest.py            # Vector DB ingestion
    │   └── vdb_retrieval.py         # Semantic search
    ├── kg/
    │   ├── neo4j_client.py          # Neo4j API wrapper
    │   ├── kg_ingest.py             # Knowledge graph ingestion
    │   ├── kg_retrieval.py          # Structural queries
    │   └── schema_init.py           # KG schema creation
    ├── ranking/
    │   ├── hybrid_ranker.py         # 5-signal ranking
    │   └── trust_ranker.py          # Domain trust scoring
    ├── llms/
    │   └── groq_service.py          # LLM integration (Groq/OpenAI)
    ├── logging/
    │   ├── log_manager.py           # Redis + SQLite logging
    │   ├── log_handler.py           # Logging integration
    │   ├── log_store.py             # SQLite persistence
    │   └── redis_broadcaster.py     # Redis pub/sub
    └── common/
        ├── url_helpers.py           # URL utilities
        ├── text_cleaner.py          # Text normalization
        ├── dedup.py                 # Deduplication
        └── list_ops.py              # List operations
```

## 📦 Installation

### Requirements

- **Python**: 3.13+
- **External Services**:
  - Pinecone (vector database)
  - Neo4j (knowledge graph)
  - Redis (logging & caching)
  - Google Custom Search Engine (CSE)
  - Groq or OpenAI (LLM API)

### Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/Luxia-AI/worker.git
   cd worker
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate  # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

4. **Configure Environment** (see [Configuration](#-configuration))

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# FastAPI
LOG_LEVEL=INFO
PORT=9000

# LLM Configuration
LLM_MODEL_NAME=grok-2-1212
LLM_TEMPERATURE=0.7
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Fallback

# Embedding Model
EMBEDDING_MODEL_NAME_PROD=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_MODEL_NAME_TEST=sentence-transformers/all-MiniLM-L6-v2

# Pinecone (Vector Database)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=worker-index
PINECONE_ENVIRONMENT=us-east-1

# Neo4j (Knowledge Graph)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Google Custom Search
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_cse_id

# Redis (Logging & Caching)
REDIS_URL=redis://localhost:6379
LOG_DB_PATH=./logs.db

# Database
DATABASE_URL=sqlite:///./logs.db

# Features
RATE_LIMIT_ENABLED=true
RATE_LIMIT_CALLS=5
RATE_LIMIT_PERIOD=1  # seconds
```

### Configuration Constants

All tuneable parameters are in `app/constants/config.py`:

```python
# Pipeline configuration
PIPELINE_MAX_ROUNDS = 3                    # Max reinforcement loops
PIPELINE_CONF_THRESHOLD = 0.70            # Confidence threshold for reinforcement
PIPELINE_MIN_NEW_URLS = 2                 # Min new URLs per reinforcement round

# Ranking weights (5-signal hybrid scoring)
RANKING_WEIGHTS = {
    'credibility': 0.25,
    'recency': 0.25,
    'semantic_similarity': 0.25,
    'entity_match': 0.15,
    'kg_score': 0.10
}

# Trusted domains
TRUSTED_DOMAINS_AUTHORITY = {'who.int', 'cdc.gov', 'nih.gov', ...}
TRUSTED_DOMAINS_EDU_GOV = {'*.edu', '*.gov'}
```

## 🔌 API Endpoints

### 1. Fact Verification Search

**Endpoint**: `GET /worker/search`

**Query Parameters**:
- `query` (string, required): Medical claim or question to verify

**Response**:
```json
{
  "query": "Does vitamin C prevent colds?",
  "results": [
    {
      "statement": "Vitamin C does not prevent common cold infections...",
      "confidence": 0.85,
      "source_url": "https://example.com/article",
      "source": "NIH",
      "published_at": "2023-06-15",
      "entities": ["vitamin C", "common cold"],
      "evidence_score": 0.87
    }
  ]
}
```

**Example**:
```bash
curl "http://localhost:9000/worker/search?query=WHO+guidelines+for+COVID-19+vaccination"
```

### 2. Admin - Logging

**Endpoint**: `GET /admin/logs`

**Query Parameters**:
- `skip` (int): Number of logs to skip (pagination)
- `limit` (int): Number of logs to return
- `request_id` (string, optional): Filter by request ID
- `level` (string, optional): Filter by log level (DEBUG, INFO, WARNING, ERROR)
- `module` (string, optional): Filter by module name

**Response**:
```json
{
  "logs": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2025-11-22T10:30:00Z",
      "level": "INFO",
      "message": "[SearchPhase:uuid] Found 5 trusted sources",
      "module": "search_phase",
      "request_id": "req-123"
    }
  ],
  "total": 1250,
  "page": 1,
  "per_page": 10
}
```

### 3. Admin - Log Statistics

**Endpoint**: `GET /admin/logs/stats`

**Query Parameters**:
- `request_id` (string, optional): Filter by request ID
- `level` (string, optional): Filter by log level
- `module` (string, optional): Filter by module

**Response**:
```json
{
  "total": 1250,
  "by_level": {
    "DEBUG": 50,
    "INFO": 1000,
    "WARNING": 150,
    "ERROR": 50
  },
  "by_module": {
    "search_phase": 300,
    "extraction_phase": 280,
    "ranking_phase": 220,
    ...
  }
}
```

## 🔄 Pipeline Phases

### Phase 1: Search

**File**: `app/services/corrective/pipeline/search_phase.py`

**Process**:
1. Reformulate input query using LLM (improves search results)
2. Filter Google CSE results to trusted domains only
3. Return top-N URLs from authoritative sources

**Key Function**: `do_search(claim: str) -> List[str]`

**Outputs**:
- `search_urls`: List of URLs from trusted domains

### Phase 2: Scraping

**File**: `app/services/corrective/pipeline/extraction_phase.py:scrape_pages()`

**Process**:
1. Fetch HTML from URLs using HTTP client
2. Convert HTML to plain text using Trafilatura
3. Clean and normalize text
4. Deduplicate content

**Key Function**: `scrape_pages(search_urls: List[str]) -> List[str]`

**Outputs**:
- `scraped_content`: List of plain-text webpage content

### Phase 3: Extraction

**File**: `app/services/corrective/pipeline/extraction_phase.py:extract_all()`

**Process**:
1. **Fact Extraction**: LLM extracts claims/statements from content
2. **Entity Extraction**: LLM identifies medical/scientific entities
3. **Relation Extraction**: LLM identifies relationships between entities

**Key Functions**:
- `extract_all(content: List[str]) -> Tuple[List[Dict], List[str], List[Dict]]`

**Outputs**:
- `facts`: List of fact dicts with statement, confidence, source, published_at
- `entities`: List of extracted entities
- `triples`: List of relationship triples (subject-relation-object)

### Phase 4: Ingestion

**File**: `app/services/corrective/pipeline/ingestion_phase.py`

**Process**:
1. Embed facts using sentence transformer
2. Store embeddings in Pinecone (VDB)
3. Store facts as nodes/relationships in Neo4j (KG)

**Key Function**: `ingest_facts_and_triples(facts, triples) -> Tuple[int, int]`

**Outputs**:
- Facts stored in Pinecone
- Relationships stored in Neo4j

### Phase 5: Retrieval

**File**: `app/services/corrective/pipeline/retrieval_phase.py`

**Process**:
1. **Semantic Search**: Query Pinecone for similar facts
2. **Structural Search**: Query Neo4j for related entities/relationships
3. Combine and deduplicate candidates

**Key Function**: `retrieve_candidates(claim: str) -> List[Dict]`

**Outputs**:
- `candidates`: List of retrieved fact/relationship candidates

### Phase 6: Ranking

**File**: `app/services/corrective/pipeline/ranking_phase.py`

**Process**:
Compute 5-signal hybrid score:

1. **Credibility** (0.25 weight)
   - Domain authority mapping (WHO > CDC > Academic > News)
   - Normalized to [0, 1]

2. **Recency** (0.25 weight)
   - Exponential decay: exp(-age_days / HALF_LIFE)
   - Recent sources weighted higher

3. **Semantic Similarity** (0.25 weight)
   - Cosine similarity between claim and fact embeddings
   - VDB embedding match score

4. **Entity Match** (0.15 weight)
   - % of extracted entities found in candidate
   - Bonus for exact entity matches

5. **KG Structural Score** (0.10 weight)
   - Confidence of relationships in knowledge graph
   - Path strength in entity networks

**Final Score**: Weighted sum of normalized signals

**Key Function**: `rank_candidates(candidates: List[Dict]) -> List[Dict]`

**Outputs**:
- `ranked_candidates`: Sorted by final_score (descending)

### Phase 7: Reinforcement

**File**: `app/services/corrective/pipeline/reinforcement_phase.py`

**Process**:
1. Check if max(ranked_candidates).final_score < CONF_THRESHOLD
2. If yes and round < MAX_ROUNDS:
   - Collect entities from low-scoring candidates
   - Re-search with entity-focused queries (Phase 1)
   - Repeat phases 2-6
3. Return final ranked results

**Key Function**: `reinforcement_loop(candidate_results, round) -> List[Dict]`

**Logic**:
```
while round < MAX_ROUNDS and max_confidence < THRESHOLD:
    failed_entities = collect_low_confidence_entities(candidates)
    new_urls = search_with_entities(failed_entities)
    if len(new_urls) < MIN_NEW_URLS:
        break  # Not enough new evidence
    candidates = pipeline_phases_2_to_6(new_urls)
    round += 1
```

## 🧪 Testing

### Test Structure

```
tests/
├── test_pipeline_full.py              # E2E tests (mocked external services)
├── test_pipeline_integration.py       # Integration tests
├── test_pipeline_with_real_storage.py # Real storage tests
├── test_pipeline_actual.py            # Actual pipeline tests
├── test_entity_extractor.py           # Service unit tests
├── test_fact_extracting.py
├── test_relation_extractor.py
├── test_scraper.py
├── test_trusted_search.py
├── test_hybrid_rank.py
├── test_trust_ranker.py
├── test_neo4j_client.py
├── test_pinecone_client.py
├── test_vdb_ingest.py
├── test_vdb_retrieval.py
├── test_kg_ingest.py
├── test_logging_system.py
└── conftest.py                        # Pytest configuration with fixtures
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pipeline_full.py

# Run tests matching pattern
pytest tests/ -k "extraction"

# Run with coverage
pytest --cov=app tests/

# Run only local tests (skip external service tests)
pytest -m "not redis_required and not e2e"

# Run verbose output
pytest -v

# Run in parallel (faster)
pytest -n auto
```

### Test Markers

```python
@pytest.mark.integration       # Requires external services
@pytest.mark.slow             # Long-running tests
@pytest.mark.redis_required    # Requires Redis (auto-skipped in CI)
@pytest.mark.e2e              # End-to-end tests
```

### CI/CD in GitHub Actions

```yaml
# .github/workflows/ci.yml
- name: Run Tests
  run: pytest -m "not redis_required and not e2e" -q
```

Tests with Redis/E2E markers are auto-skipped in CI environment.

## 🚀 Development

### Code Organization

**Constants** (never hardcode):
```python
# app/constants/config.py
PIPELINE_MAX_ROUNDS = 3
PIPELINE_CONF_THRESHOLD = 0.70
RANKING_WEIGHTS = {...}
TRUSTED_DOMAINS_AUTHORITY = {...}
```

**Logging** (structured with round_id):
```python
from app.core.logger import get_logger
logger = get_logger(__name__)
logger.info(f"[PhaseX:round_id] Message", extra={"round_id": round_id})
```

**Async** (async-first design):
```python
async def extract_facts(content: List[str]) -> List[Dict]:
    tasks = [extractor.extract(c) for c in content]
    return await asyncio.gather(*tasks)
```

### Running Quality Checks

```bash
# Run all checks
./run.sh all

# Run specific check
./run.sh "black mypy ruff"

# Available checks: pytest, ruff, black, isort, flake8, bandit, mypy
```

### Code Style

- **Black**: Line length = 120
- **isort**: Black profile
- **Type hints**: mypy (lenient, some ignores for framework code)
- **Linting**: Ruff, Flake8
- **Security**: Bandit

## 🐳 Deployment

### Docker

**Build**:
```bash
docker build -t luxia-worker:latest .
```

**Run**:
```bash
docker run -p 9000:9000 \
  -e PINECONE_API_KEY=xxx \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e GROQ_API_KEY=xxx \
  -e GOOGLE_API_KEY=xxx \
  luxia-worker:latest
```

**Docker Compose** (recommended):
```bash
docker-compose up --build
```

### Environment: Local vs Docker vs Production

```yaml
# .env.local
REDIS_URL: redis://localhost:6379
NEO4J_URI: bolt://localhost:7687

# .env.docker (docker-compose.yml sets these)
REDIS_URL: redis://redis:6379
NEO4J_URI: bolt://neo4j:7687

# .env.prod (K8s secrets, etc.)
# All secrets from environment/vault
```

### Health Checks

```bash
# Check API health
curl http://localhost:9000/worker/search?query=test

# Check logs
curl http://localhost:9000/admin/logs?limit=10

# Check Docker container
docker ps
docker logs worker
```

## 🔧 Troubleshooting

### Common Issues

**Redis Connection Failed**
```
Error: Connection refused (redis://localhost:6379)
```
- Ensure Redis is running: `docker run -p 6379:6379 redis`
- Check REDIS_URL in .env

**Pinecone Not Found**
```
Error: Index not found: worker-index
```
- Create index in Pinecone dashboard
- Verify PINECONE_INDEX_NAME and PINECONE_API_KEY

**Neo4j Connection Issues**
```
Error: Could not connect to bolt://localhost:7687
```
- Ensure Neo4j is running: `docker run -p 7687:7687 neo4j`
- Check credentials: NEO4J_USER, NEO4J_PASSWORD

**LLM API Errors**
```
Error: Groq API rate limit exceeded
```
- Check GROQ_API_KEY
- Reduce rate limits or use fallback OpenAI
- Check OPENAI_API_KEY

**Tests Failing with "Redis not available"**
```bash
# Expected in CI (tests auto-skip)
# For local testing, ensure Redis is running or skip:
pytest -m "not redis_required"
```

### Debug Logging

Enable detailed logging:
```python
# .env
LOG_LEVEL=DEBUG
```

View logs:
```bash
curl http://localhost:9000/admin/logs?level=DEBUG&limit=50
```

### Performance Optimization

1. **Batch Operations**: Process multiple claims in parallel
2. **Caching**: Results cached in Redis (configurable TTL)
3. **Rate Limiting**: Respect external API limits (configured in config.py)
4. **Embedding Model**: Use lightweight model for production (all-MiniLM-L6-v2)

## 📊 Key Metrics

### Pipeline Performance

- **Search Phase**: ~2-3 seconds per claim
- **Scraping**: ~5-10 seconds (5-20 URLs)
- **Extraction**: ~8-12 seconds (3x LLM calls)
- **Retrieval**: ~1-2 seconds (VDB + KG queries)
- **Ranking**: ~0.5 seconds (hybrid scoring)
- **Total (single round)**: ~20-30 seconds
- **With Reinforcement**: 20-90 seconds (up to 3 rounds)

### Test Coverage

- Unit Tests: 81+ passing
- Integration Tests: 12+ tests
- E2E Tests: 5+ real claim scenarios
- Code Quality: 100% (Black, isort, mypy, ruff, flake8, bandit)

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes following code style
3. Run tests: `./run.sh all`
4. Commit: `git commit -m "feat: description"`
5. Push and create PR

## 📄 License

See LICENSE file in repository.

## 📞 Support

- **Issues**: GitHub Issues
- **Questions**: GitHub Discussions

---

**Version**: 1.0.0  
**Last Updated**: November 22, 2025  
**Status**: Production Ready ✅
