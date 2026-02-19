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
