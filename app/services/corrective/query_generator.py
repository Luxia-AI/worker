from __future__ import annotations

from typing import Any, Dict, List

from app.services.corrective.query_designer import ClaimEntities, CorrectiveQueryDesigner, QualityGateResult
from app.services.corrective.query_designer import QueryPlan as _QueryPlan
from app.services.corrective.query_designer import build_plan as _build_plan
from app.services.corrective.query_designer import register_drift_from_url as _register_drift_from_url

_DESIGNER = CorrectiveQueryDesigner()
QueryPlan = _QueryPlan


def quality_gate(
    claim_type: str,
    entities: ClaimEntities,
    url: str,
    title: str,
    snippet: str,
) -> QualityGateResult:
    return _DESIGNER.quality_gate(claim_type, entities, url, title, snippet)


def select_results(
    claim_type: str,
    entities: ClaimEntities,
    results: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    return _DESIGNER.select_results(claim_type, entities, results)


def build_plan(claim: str) -> QueryPlan:
    return _build_plan(claim)


def register_drift_from_url(claim_type: str, url: str, title: str, snippet: str) -> None:
    _register_drift_from_url(claim_type, url, title, snippet)


__all__ = [
    "QueryPlan",
    "build_plan",
    "register_drift_from_url",
    "quality_gate",
    "select_results",
]
