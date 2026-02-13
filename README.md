# Worker Service: Microscopic Technical Guide

This document explains how the Worker actually works, from input request to final verdict, including the core formulas for ranking, trust, truthfulness, and confidence.

Audience:

- New engineers with zero system context
- Researchers evaluating scoring behavior
- Developers debugging verdict quality issues

Primary code references:

- `worker/app/main.py`
- `worker/app/services/corrective/pipeline/__init__.py`
- `worker/app/services/ranking/hybrid_ranker.py`
- `worker/app/services/ranking/adaptive_trust_policy.py`
- `worker/app/services/verdict/verdict_generator.py`
- `worker/app/constants/config.py`

## 1) Worker Mission

The Worker is the claim-verification core. It does four jobs:

1. Retrieve evidence (Vector DB + Knowledge Graph + optional web expansion)
2. Rank evidence with deterministic scoring
3. Compute adaptive trust sufficiency for the claim
4. Generate a verdict package with rationale and analysis metadata

Output is structured, not just text:

- `verdict`
- `truthfulness_percent`
- `confidence`
- `evidence`, `evidence_map`, `claim_breakdown`
- trust and ranking diagnostics

## 2) API Surface

Main endpoint:

- `POST /worker/verify`

Key request fields:

- `job_id` (required)
- `claim` (required)
- `room_id`, `source` (optional)
- `domain` (default `general`)
- `top_k` (default `5`, bounded `1..20`)

Key response signals:

- `verdict`, `verdict_confidence`, `truthfulness_percent`
- ranking: `top_ranking_score`, `avg_ranking_score`
- trust: `trust_policy_mode`, `trust_metric_value`, `trust_threshold_met`, `coverage`, `diversity`
- observability: `data_source`, `used_web_search`, signal counts, and `analysis_counts`

## 3) Execution Flow in Micro-Steps

Runtime path is implemented in `CorrectivePipeline.run(...)`.

1. Parse claim and run deterministic/LLM-assisted entity extraction.
2. Build retrieval queries from full claim plus decomposed subclaims.
3. Retrieve initial candidates from VDB and KG (`retrieve_candidates`).
4. Rank candidates (`rank_candidates` -> `hybrid_rank`).
5. Convert ranked rows to trust evidence objects and compute adaptive trust.
6. If adaptive trust is sufficient, run a cache verdict precheck.
7. If precheck passes strict gates, exit as `completed_from_cache`.
8. If insufficient, generate search queries for incremental web expansion.
9. For each query:
10. Search URLs, filter already-processed URLs, and cap per-query URLs.
11. Scrape pages and extract facts/entities/relations.
12. Ingest extracted facts/triples into VDB and KG.
13. Re-retrieve and re-rank with updated evidence.
14. Recompute adaptive trust and stop early when sufficient.
15. After search loop ends, run final verdict generation.
16. Return final payload with full scoring diagnostics.

## 4) Evidence Ranking Formula (Hybrid Ranker)

Ranking constants come from `worker/app/constants/config.py`:

- `w_semantic = 0.31`
- `w_kg = 0.12`
- `w_entity = 0.20`
- `w_claim_overlap = 0.15`
- `w_recency = 0.05`
- `w_credibility = 0.17`

### 4.1 Base score

For each candidate:

`final_score_base = (w_sem * sem_norm) + (w_kg * kg_norm) + (w_entity * entity_overlap) + (w_claim_overlap * claim_overlap) + (w_recency * recency_boost) + (w_cred * credibility)`

Where:

- `sem_norm`, `kg_norm` are min-max normalized across current candidate set
- `recency_boost = 0.5^(days_since_publish / RECENCY_HALF_LIFE_DAYS)` with half-life `365`
- `entity_overlap` and `claim_overlap` are lexical/entity recall-like overlaps

### 4.2 Additive boosts

After base score:

- `+ 0.05 * min(1, kg_raw)` if `kg_raw > 0` and `claim_overlap >= 0.08`
- `+ 0.10` if `kg_norm > 0`, `entity_overlap >= 0.34`, `claim_overlap >= 0.10`
- `+ 0.08` if candidate is KG, `kg_raw >= 0.40`, and anchors pass

### 4.3 Multiplicative penalties

Important score reductions:

- Anchor weak:
  - KG candidate: `* 0.30` when `anchor_match_score < 0.20`
  - non-KG: `* 0.70` when `anchor_match_score < 0.20`
- Backfill weak item: `* 0.85`
- Object mismatch: `* 0.55`
- Claim focus mismatch: `* 0.75`
- Subject mismatch: `* 0.60`
- Relation object mismatch: `* 0.70`
- Claim-mention/reporting style in non-belief claims: `* 0.65`
- Uncertainty penalty: subtract `0.12` for assertive claims with uncertain wording

### 4.4 Floors and hard filters

- Floor: if `sem_norm == 0`, `kg_norm == 0`, and `credibility >= 0.9`, then `final_score >= 0.2`
- Final clamp: `[0, 1]`
- Strict candidate filters remove low-overlap/low-signal items before final ranking

## 5) Adaptive Trust Policy (Coverage + Diversity + Agreement)

Adaptive trust is computed in `AdaptiveTrustPolicy.compute_adaptive_trust(...)`.

### 5.1 Core metrics

- `coverage`: weighted subclaim coverage from `compute_subclaim_coverage`
- `diversity`: source-domain diversity score
- `agreement`: fraction of evidence not marked `contradicts`

Coverage calculation:

`coverage = sum(detail.weight) / number_of_subclaims`

### 5.2 Gating rules (sufficiency decision)

Sufficiency can pass by multiple routes:

- Low-count override when diversity and relevance are very high
- Confidence-mode relaxed gate
- Adaptive override when no contradicted subclaims and one of:
  - `coverage >= 0.25 and diversity >= 0.75`
  - `strong_covered >= 1`
  - `avg_relevance >= 0.55`
- Strict gates:
  - high coverage + high agreement
  - medium coverage + high diversity + high agreement

Default thresholds (normal mode):

- `COVERAGE_THRESHOLD_HIGH = 0.60`
- `COVERAGE_THRESHOLD_MEDIUM = 0.40`
- `DIVERSITY_THRESHOLD_HIGH = 0.70`
- `AGREEMENT_THRESHOLD_HIGH = 0.80`
- `MIN_EVIDENCE_COUNT = 3`

Confidence mode lowers several thresholds and minimum evidence count.

### 5.3 Trust post formula

If evidence exists:

1. Build `subclaim_trusts = best_evidence_trust * subclaim_weight` for covered subclaims.
2. If non-empty:

`mean_subclaim_trust = average(subclaim_trusts)`

`trust_post = mean_subclaim_trust * (0.5 + 0.5 * coverage)`

3. Else fallback:

`trust_post = top_trust * coverage`

4. Additional consistency floor may re-apply `top_trust * coverage` when needed.

Verdict state from trust:

- `evidence_insufficiency` if not sufficient
- `confirmed` if `trust_post >= 0.8`
- `provisional` if `trust_post >= 0.6`
- else `revoked`

Weak-relevance guard can flip a pass back to insufficiency when trust/semantic/top-trust are weak.

## 6) Truthfulness: How It Is Actually Computed

There are two truth-related outputs inside verdict generation:

1. `truth_score_percent` (status-driven, used as output `truthfulness_percent`)
2. `evidence_quality_percent` (evidence-driven quality signal)

### 6.1 Status-driven truth score (contract-based)

Status weights:

- `STRONGLY_VALID = 1.0`
- `VALID = 1.0`
- `PARTIALLY_VALID = 0.75`
- `UNKNOWN = 0.45`
- `PARTIALLY_INVALID = 0.25`
- `INVALID = 0.0`

Base score:

`base = average(status_weight(segment_i))`

Then penalties:

- `- 0.30 * contradict_ratio`
- `- 0.10 * unresolved_ratio`

Then guards:

- unresolved support-only claims capped below full certainty
- fully resolved strong TRUE can be lifted near certainty
- pure contradiction FALSE can be forced low

Final:

`truth_score_percent = clamp(score, 0, 1) * 100`

Then additional pipeline-level caps/overrides apply:

- diversity ceiling: `ceiling_percent = (0.85 + 0.10 * diversity) * 100`
- numeric override paths
- UNVERIFIABLE and absolute-quantifier guardrails

Important:

- Final response field `truthfulness_percent` is this status-driven truth score.
- `evidence_quality_percent` is returned separately for diagnostics.

### 6.2 Evidence quality percent (diagnostic truthfulness)

`_calculate_truthfulness_from_evidence(...)` computes segment-level support with:

- semantic+kg relevance: `0.7*sem + 0.3*kg`
- anchor overlap
- lexical overlap
- credibility
- contradiction penalties
- uncertainty/reporting penalties

Segment support core:

`support = 0.70*rel + 0.15*anchor_overlap + 0.10*overlap_ratio + 0.05*cred`

Global evidence quality:

`avg_segment_support * (0.85 + 0.15*diversity)`

with a ceiling:

`<= 0.85 + 0.10*diversity`

Converted to percent and returned as `evidence_quality_percent`.

## 7) Confidence Formula (Final Path)

Primary final confidence in active path:

`confidence = 0.30*truthfulness + 0.25*coverage + 0.20*agreement + 0.15*diversity + 0.10*adaptive_trust_post`

Then:

- cap at `0.95`
- if `coverage >= 0.8`, enforce `confidence >= truthfulness * 0.75`
- apply contract/policy cap function (`_cap_confidence_with_contract`)
- additional hard overrides for strong contradiction votes, UNVERIFIABLE locks, and numeric gates

Contract cap behavior includes:

- unresolved segments lower max confidence
- policy insufficiency lowers cap further
- UNVERIFIABLE hard cap (`0.35` or `0.45` for comparative claims)
- contradiction-only fully resolved FALSE gets a minimum floor (not underconfident)

Note:

- There is also `_calculate_confidence(...)` helper in the file with a detailed formula, but the current primary output path uses the composite confidence formula above in the verdict generation flow.

## 8) Verdict Reconciliation and Polarity Guards

Worker does not blindly trust raw LLM verdict text. It reconciles verdict with:

- claim breakdown statuses
- admissible evidence stance
- evidence-map support/contradiction signal strengths
- rationale polarity hints

Hard guards:

- Strong contradiction stance can force FALSE
- Strong entailment can force TRUE (with constraints)
- Weak/mixed signal bands can force UNVERIFIABLE
- Binary verdicts require stronger quality gates, else downgraded to UNVERIFIABLE
- Absolute-quantifier claims (`always`, `never`, `all`, etc.) are guarded from overconfident TRUE

## 9) Why Two Trust Views Exist

Worker tracks:

- Adaptive trust (`trust_post_adaptive`): subclaim/coverage-aware sufficiency logic
- Fixed trust (`trust_post_fixed`): traditional post-trust score for comparability

Output includes both plus gate metadata so downstream systems can audit decisions.

## 10) Practical Debug Recipe

If verdict looks wrong, inspect in this order:

1. `analysis_counts.admissible_evidence_ratio`
2. `claim_breakdown` statuses and unresolved segments
3. `coverage`, `diversity`, `agreement`, `trust_post`
4. top evidence anchors and claim overlap
5. binary gate and UNVERIFIABLE lock conditions

This sequence usually identifies whether the problem is retrieval quality, ranking bias, trust gating, or verdict reconciliation.

## 11) Related Docs

- `docs/README.md`
- `docs/worker-pipeline.md`
- `docs/interfaces-and-contracts.md`
- `docs/testing-and-validation.md`

Last verified against code: February 13, 2026
