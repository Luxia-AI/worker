"""
LLM Prompts for biomedical fact extraction, entity recognition, and knowledge graph construction.
Centralized prompt definitions used across the corrective and ranking pipelines.
"""

# ============================================================================
# ENTITY EXTRACTION PROMPTS
# ============================================================================

BIOMED_NER_PROMPT = """You are a biomedical Named Entity Recognition (NER) model.
Extract only medically relevant entities that are explicitly asserted in the fact.
Do not infer entities not directly present in the statement.
Do not return entities from speculative/hedged parts (may, might, could, possible, hypothesis).
Entities: diseases, conditions, symptoms, chemicals, nutrients, organs, viruses,
medication names, biological processes.
Return ONLY valid JSON (no markdown, no extra text):
{"entities": ["entity1", "entity2", ...]}

Fact: {statement}"""

# ============================================================================
# FACT EXTRACTION PROMPTS
# ============================================================================

FACT_EXTRACTION_PROMPT = """Extract key factual statements from this content.
IMPORTANT: Return only truth-grounded statements explicitly supported by the provided content.
IMPORTANT: Return only atomic, single-claim facts (no conjunctions, no multi-part statements).
Exclude:
- speculation/hedging (may, might, could, possible, potentially, suggests, appears)
- opinion/normative language
- rumors, claims-about-claims, rhetorical questions, anecdotal statements
- generic background not tied to a concrete assertion in the text
Return ONLY valid JSON with this exact structure (no extra text, no markdown):
{{"facts": [{{"statement": "...", "confidence": 0.85}}, {{"statement": "...", "confidence": 0.90}}]}}

Content:
{content}"""

FACT_EXTRACTION_PREDICATE_FORCING_PROMPT = """Extract only statements that explicitly address whether:
subject: {subject}
predicate: {predicate}
object: {object}

Strict extraction rules:
1) Keep only statements that explicitly negate or confirm the predicate mechanism for this subject/object.
2) A statement qualifies as refuting only when it contains explicit negation,
   biological impossibility, or mechanism preventing the predicate.
   Refutation cues include patterns such as:
   - does not
   - cannot
   - no evidence
   - does not enter
   - does not integrate
   - cannot alter
   - cannot modify
   - does not affect
3) Generic background information must be excluded.
4) If the predicate phrase (or a clear semantic equivalent) is not present, skip it.
5) Return atomic single-claim statements only.

Return ONLY valid JSON:
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
- For negated statements, encode negation in relation label (e.g., "does_not_cause", "not_associated_with")
- Do not infer relations not explicitly stated in the sentence
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

QUERY_REFORMULATION_PROMPT = """You are a medical claim verification query planner.
Generate exactly 4 search queries in two logical tracks:
- Track A (support-check): queries that could support the claim.
- Track B (refute-check): queries that could refute the claim.

Hard constraints:
1) Keep subject, predicate, and object aligned to the claim text. Do not add new entities.
2) Preserve negation logic exactly (e.g., "does not", "cannot"). No malformed grammar.
3) Include at least one high-evidence formulation:
   - systematic review OR meta-analysis OR randomized trial OR guideline.
4) If claim includes numbers/dose/population, include them verbatim in at least one query.
5) Keep each query concise (4-10 words) and evidence-oriented.
6) No promotional language, no broad/vague phrases.
7) Output valid JSON only. No markdown. No prose.

Expected JSON schema:
{"queries":["q1","q2","q3","q4"]}

Few-shot guidance:
- Claim: "Vitamin C does not support immune health"
  Good refute-check query: "vitamin c supports immune health systematic review"
  Good support-check query: "no evidence vitamin c supports immunity trial"
- Claim: "X reduces blood pressure"
  Good support-check query: "x reduces blood pressure randomized trial"
  Good refute-check query: "x does not reduce blood pressure meta-analysis"

Return ONLY:
{"queries":["...","...","...","..."]}"""
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
