"""
LLM Prompts for biomedical fact extraction, entity recognition, and knowledge graph construction.
Centralized prompt definitions used across the corrective and ranking pipelines.
"""

# ============================================================================
# ENTITY EXTRACTION PROMPTS
# ============================================================================

BIOMED_NER_PROMPT = """
You are a biomedical Named Entity Recognition (NER) model.

Extract ALL medically relevant entities from the following fact.
Entities should include:
- diseases
- conditions
- symptoms
- chemicals
- nutrients
- organs
- viruses
- medication names
- biological processes

Return ONLY this JSON format:
{
  "entities": ["entity1", "entity2", ...]
}
"""

# ============================================================================
# FACT EXTRACTION PROMPTS
# ============================================================================

FACT_EXTRACTION_PROMPT = """Extract key factual statements from this content.
Return ONLY valid JSON with this structure:
{
    "facts": [
        {"statement": "...", "confidence": 0.85},
        {"statement": "...", "confidence": 0.90}
    ]
}

Content:
{content}"""

# ============================================================================
# RELATION EXTRACTION PROMPTS
# ============================================================================

TRIPLE_EXTRACTION_PROMPT = """
You are a relation extraction agent specialized in biomedical / health facts.
Given a short factual statement and a list of entities detected in that statement,
return ALL valid entity-relation-entity triples implied by the statement.

Requirements:
- Only return JSON in this exact format (no extra text):
{
  "triples": [
    {
      "subject": "string",
      "relation": "string",
      "object": "string",
      "confidence": 0.0-1.0
    },
    ...
  ]
}
- Subject and object should be entity strings (prefer values from the provided entities list).
- Relation should be a concise verb or phrase (e.g., "causes", "reduces risk of", "is a treatment for").
- Confidence should be a float between 0 and 1 indicating how strongly the triple is supported by the statement.
- If no triples can be extracted, return {"triples": []}.
- Do not output any explanation or any fields other than the JSON above.

Example:
STATEMENT:
"COVID-19 vaccines reduce hospitalization."

ENTITIES:
["covid-19", "vaccines", "hospitalization"]

OUTPUT:
{
  "triples": [
    {"subject":"vaccines", "relation":"reduce risk of", "object":"hospitalization", "confidence":0.92}
  ]
}
"""

# ============================================================================
# QUERY REFORMULATION PROMPTS
# ============================================================================

QUERY_REFORMULATION_PROMPT = """
You are a query reformulation agent specialized in retrieving
high-quality medical evidence from trusted sources (CDC, NIH, WHO, Mayo Clinic, Harvard Health).

Given a social media post and optional failed biomedical entities:

1. Extract the medically relevant keywords.
2. Generate 5-8 optimized Google search queries.
3. Each query must be:
   - short (3-7 words)
   - keyword dense
   - objective
   - medically oriented
   - suitable for evidence gathering
4. Avoid question-like queries. Focus on **search-efficient** queries.

Return ONLY this JSON structure:
{
  "queries": ["...", "...", "..."]
}
"""

REINFORCEMENT_QUERY_PROMPT = """
You are a search-query optimization model for a misinformation detection system.
Generate 8-12 highly effective **Web search queries** that retrieve authoritative,
peer-reviewed, scientific, or government-backed evidence.

Inputs:
- Low-confidence extracted statements:
  {statements}

- Failed or uncertain entities:
  {entities}

Requirements for queries:
- MUST be highly targeted.
- MUST focus on scientific, medical, anatomical, or factual verification.
- Prefer sources like NIH, WHO, CDC, Mayo Clinic, PubMed, academic journals.
- DO NOT include filler words like 'is it true'.

Output ONLY a JSON list of search queries. Example:
["query1", "query2", ...]
"""
