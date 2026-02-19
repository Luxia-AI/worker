from __future__ import annotations

from typing import Dict, List

from app.services.verdict.v2.types import VerdictDecision


def _status_truth_weight(status: str) -> float:
    weights = {
        "STRONGLY_VALID": 1.0,
        "VALID": 1.0,
        "PARTIALLY_VALID": 0.75,
        "PARTIALLY_INVALID": 0.25,
        "INVALID": 0.20,
        "UNKNOWN": 0.45,
    }
    return float(weights.get(str(status or "UNKNOWN").upper(), 0.0))


def reconcile_verdict(statuses: List[str]) -> VerdictDecision:
    normalized = [str(s or "UNKNOWN").upper() for s in statuses]
    total = len(normalized)
    unresolved = sum(1 for s in normalized if s == "UNKNOWN")
    resolved = total - unresolved
    weighted_truth = sum(_status_truth_weight(s) for s in normalized) / max(1, total)
    resolved_ratio = resolved / max(1, total)

    has_valid = any(s == "VALID" for s in normalized)
    has_partial_valid = any(s == "PARTIALLY_VALID" for s in normalized)
    has_support = has_valid or has_partial_valid or any(s == "STRONGLY_VALID" for s in normalized)
    has_invalid = any(s in {"INVALID", "PARTIALLY_INVALID"} for s in normalized)

    all_valid = bool(normalized) and all(s == "VALID" for s in normalized)
    all_invalid_like = bool(normalized) and all(s in {"INVALID", "PARTIALLY_INVALID"} for s in normalized)
    all_unknown = bool(normalized) and all(s == "UNKNOWN" for s in normalized)
    contains_partial = any(s in {"PARTIALLY_VALID", "PARTIALLY_INVALID"} for s in normalized)

    if all_unknown:
        verdict = "UNVERIFIABLE"
    elif unresolved > 0:
        verdict = "PARTIALLY_TRUE" if has_support or has_invalid else "UNVERIFIABLE"
    elif all_invalid_like and not has_support:
        verdict = "FALSE"
    elif all_valid and not contains_partial and not has_invalid:
        verdict = "TRUE"
    elif has_invalid and not has_support:
        verdict = "FALSE"
    elif has_support and has_invalid:
        verdict = "PARTIALLY_TRUE"
    else:
        # Deterministic guardrail: any partial status blocks TRUE.
        verdict = "PARTIALLY_TRUE" if contains_partial or has_support else "UNVERIFIABLE"

    truthfulness_cap = min(100.0, (weighted_truth * 100.0) + 5.0)
    if unresolved > 0:
        truthfulness_cap = min(truthfulness_cap, 65.0)
    if resolved == 0:
        truthfulness_cap = min(truthfulness_cap, 20.0)

    return VerdictDecision(
        verdict=verdict,
        required_segments_count=total,
        resolved_segments_count=resolved,
        required_segments_resolved=(unresolved == 0),
        unresolved_segments=unresolved,
        matched_statuses=normalized,
        weighted_truth=max(0.0, min(1.0, weighted_truth)),
        truthfulness_cap=max(0.0, truthfulness_cap),
        resolved_ratio=max(0.0, min(1.0, resolved_ratio)),
        has_support=has_support,
        has_invalid=has_invalid,
        debug={"contains_partial": contains_partial, "all_valid": all_valid},
    )


def reconcile_from_breakdown(claim_breakdown: List[Dict[str, object]]) -> VerdictDecision:
    statuses = [str((seg or {}).get("status") or "UNKNOWN").upper() for seg in (claim_breakdown or [])]
    return reconcile_verdict(statuses)
