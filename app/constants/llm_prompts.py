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
{{"facts": [{{"statement": "...", "confidence": 0.85}}, {{"statement": "...", "confidence": 0.90}}]}}

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
- If no triples found: {{"triples": []}}

Example:
Statement: "COVID-19 vaccines reduce hospitalization."
Entities: ["covid-19", "vaccines", "hospitalization"]
Output: {{"triples": [{{"subject":"vaccines", "relation":"reduce",
"object":"hospitalization", "confidence":0.92}}]}}

Statement: {statement}
Entities: {entities}"""

# ============================================================================
# QUERY REFORMULATION PROMPTS
# ============================================================================

QUERY_REFORMULATION_PROMPT = """You are a search query optimizer for claim verification (medical/health).
Your job: generate EXACTLY 3–4 web search queries that help verify THIS specific claim.

ABSOLUTE RULES (must follow):
1) Claim-anchored: Every query MUST include at least 1–2 key terms copied verbatim from the claim \
   (entity names, condition, drug, mechanism, population, outcome, numbers).
2) No topic drift: Do NOT introduce new entities/topics not present in the claim \
   (e.g., vitamins, nutrients, collagen, selenium) unless explicitly mentioned in the claim.
3) Query length: 3–7 words each, short and specific.
4) Evidence-oriented: Prefer terms that retrieve authoritative sources \
   (guideline, systematic review, RCT, cohort, meta-analysis, CDC/WHO/NIH).
5) Coverage: Queries should collectively cover:
   - definition/claim core (what is asserted)
   - mechanism or causality (if asserted)
   - effect/outcome magnitude or safety (if asserted)
6) Output MUST be ONLY valid JSON.

FORMAT EXAMPLES (structure only — DO NOT reuse these words):
Example claim: "<CLAIM_TEXT>"
Good query shapes:
- "<KEY_TERM_1> <KEY_TERM_2> systematic review"
- "<KEY_TERM_1> <OUTCOME_TERM> randomized trial"
- "<KEY_TERM_1> mechanism evidence"
Bad query shapes:
- "benefits of <KEY_TERM_1>" (too vague/promotional)
- "<unrelated nutrient> mechanism" (introduces new topic)

Return ONLY valid JSON:
{"queries": ["...", "...", "..."]}

Claim: {post}"""

REINFORCEMENT_QUERY_PROMPT = """You are a search-query optimization model for claim verification.
Generate 8–12 highly effective web search queries for authoritative evidence.

Inputs:
Low-confidence statements:
{statements}

Entities (use these; do NOT add new ones):
{entities}

STRICT RULES:
1) Every query must include at least one entity term from the provided Entities list (verbatim).
2) Do NOT introduce new entities/topics not present in Entities or Statements.
3) Prefer authoritative targets: WHO, CDC, NIH, NICE, Cochrane, PubMed, major journals.
4) Keep queries 4–9 words, specific and evidence-oriented.
5) Output ONLY valid JSON.

Return ONLY valid JSON:
{"queries": ["query1", "query2", "query3"]}"""
