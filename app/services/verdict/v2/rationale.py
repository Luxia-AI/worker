from __future__ import annotations

from typing import Dict, List


def build_evidence_grounded_rationale(claim_breakdown: List[Dict[str, object]], verdict: str) -> str:
    statuses = [str((seg or {}).get("status") or "UNKNOWN").upper() for seg in (claim_breakdown or [])]
    unknown = sum(1 for s in statuses if s == "UNKNOWN")
    invalid = sum(1 for s in statuses if s in {"INVALID", "PARTIALLY_INVALID"})
    valid = sum(1 for s in statuses if s in {"VALID", "PARTIALLY_VALID", "STRONGLY_VALID"})

    if verdict == "UNVERIFIABLE":
        return (
            "Available admissible evidence is mixed or insufficiently decisive for this claim, "
            "so the result is UNVERIFIABLE."
        )
    if verdict == "FALSE":
        return "Admissible evidence contradicts the claim on required segments."
    if verdict == "TRUE":
        return "Admissible evidence consistently supports all required claim segments."
    if invalid > 0 and valid > 0:
        return "Evidence supports some parts of the claim while contradicting others."
    if unknown > 0:
        return "Some claim segments remain unresolved by admissible evidence."
    return "Evidence partially supports the claim with caveats."


def build_key_findings(claim_breakdown: List[Dict[str, object]]) -> List[str]:
    findings: List[str] = []
    for seg in claim_breakdown or []:
        status = str((seg or {}).get("status") or "UNKNOWN").upper()
        fact = str((seg or {}).get("supporting_fact") or "").strip()
        if status in {"VALID", "INVALID", "PARTIALLY_VALID", "PARTIALLY_INVALID"} and fact:
            findings.append(f"{status}: {fact}")
        if len(findings) >= 5:
            break
    return findings
