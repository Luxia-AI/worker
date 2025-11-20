"""
LLM Prompts for biomedical fact extraction, entity recognition, and knowledge graph construction.
Centralized prompt definitions used across the corrective and ranking pipelines.
"""

# ============================================================================
# ENTITY EXTRACTION PROMPTS
# ============================================================================

BIOMED_NER_PROMPT = """You are a biomedical Named Entity Recognition (NER) model.
Extract ALL medically relevant entities from the following fact.
Entities: diseases, conditions, symptoms, chemicals, nutrients, organs, viruses,
medication names, biological processes.
Return ONLY valid JSON (no markdown, no extra text):
{"entities": ["entity1", "entity2", ...]}

Fact: {statement}"""

# ============================================================================
# FACT EXTRACTION PROMPTS
# ============================================================================

FACT_EXTRACTION_PROMPT = """Extract key factual statements from this content.
Return ONLY valid JSON with this exact structure (no extra text, no markdown):
{"facts": [{"statement": "...", "confidence": 0.85}, {"statement": "...", "confidence": 0.90}]}

Content:
{content}"""

# ============================================================================
# RELATION EXTRACTION PROMPTS
# ============================================================================

TRIPLE_EXTRACTION_PROMPT = """You are a relation extraction agent specialized in biomedical/health facts.
Given a factual statement and detected entities, return ALL valid entity-relation-entity triples.

Requirements:
- Return ONLY valid JSON (no markdown, no explanation):
- Subject/object must be entity strings from provided list
- Relation should be concise (e.g., "causes", "reduces risk of", "is treatment for")
- Confidence: float 0-1 indicating support strength
- If no triples found: {"triples": []}

Example:
Statement: "COVID-19 vaccines reduce hospitalization."
Entities: ["covid-19", "vaccines", "hospitalization"]
Output: {"triples": [{"subject":"vaccines", "relation":"reduce",
"object":"hospitalization", "confidence":0.92}]}

Statement: {statement}
Entities: {entities}"""

# ============================================================================
# QUERY REFORMULATION PROMPTS
# ============================================================================

QUERY_REFORMULATION_PROMPT = """You are a query reformulation agent for retrieving medical
evidence from trusted sources (CDC, NIH, WHO, Mayo Clinic, Harvard Health).

Given a social media post, extract medically relevant keywords and generate
5-8 optimized Google search queries. Each query must be: short (3-7 words),
keyword dense, objective, medically oriented, suitable for evidence gathering.

Return ONLY valid JSON (no markdown, no explanation):
{"queries": ["query1", "query2", "query3"]}

Post: {post}"""

REINFORCEMENT_QUERY_PROMPT = """You are a search-query optimization model for misinformation
detection. Generate 8-12 highly effective Web search queries for authoritative,
peer-reviewed, scientific, or government-backed evidence.

Low-confidence statements:
{statements}

Failed/uncertain entities:
{entities}

Requirements: Queries must be highly targeted, focused on scientific/medical/
factual verification, prefer NIH/WHO/CDC/Mayo Clinic/PubMed.

Return ONLY valid JSON (no markdown):
{{"queries": ["query1", "query2", "query3"]}}"""
