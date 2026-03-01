from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Segment:
    text: str
    status: str = "UNKNOWN"
    supporting_fact: str = ""
    source_url: str = ""


@dataclass
class EvidenceItem:
    evidence_id: int
    statement: str
    source_url: str = ""
    relevance: str = "NEUTRAL"
    relevance_score: float = 0.0
    credibility: float = 0.5
    blocked_content: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSignal:
    support_prob: float
    contradiction_prob: float
    trust: float
    admissible: bool
    evidence_id: int


@dataclass
class SegmentDecision:
    segment: str
    status: str
    support_score: float
    contradiction_score: float
    evidence_ids: List[int]


@dataclass
class VerdictDecision:
    verdict: str
    required_segments_count: int
    resolved_segments_count: int
    required_segments_resolved: bool
    unresolved_segments: int
    matched_statuses: List[str]
    weighted_truth: float
    truthfulness_cap: float
    resolved_ratio: float
    has_support: bool
    has_invalid: bool
    debug: Optional[Dict[str, Any]] = None


@dataclass
class EvidenceScoreV2:
    support_score: float
    contradict_score: float
    nli_entail_prob: float
    nli_contradict_prob: float
    admissible: bool
    evidence_id: int = -1
    neutral_score: float = 0.0
    nli_neutral_prob: float = 0.0
    weight: float = 0.0
    source_domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustSnapshotV2:
    trust_support: float
    trust_contradict: float
    trust_uncertain: float
    admissibility_rate: float = 0.0
    sufficiency_reason: str = ""
    info_gain: float = 0.0


@dataclass
class VerdictPayloadV3:
    verdict: str
    class_probs: Dict[str, float]
    calibrated_confidence: float
    evidence_attribution: List[Dict[str, Any]] = field(default_factory=list)
    calibration_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDiagnosticsV2:
    stop_reason: str
    gain_estimate: float
    kg_timeout_count: int
    zero_extraction_rounds: int
