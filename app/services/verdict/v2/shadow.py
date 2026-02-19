from __future__ import annotations

from typing import Any, Dict, List


def compute_shadow_diff(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, Any]:
    diffs: Dict[str, Any] = {}
    keys = {
        "verdict",
        "confidence",
        "truthfulness_percent",
        "required_segments_resolved",
        "unresolved_segments",
    }
    for key in keys:
        left = v1.get(key)
        right = v2.get(key)
        if left != right:
            diffs[key] = {"v1": left, "v2": right}

    left_statuses = [str((x or {}).get("status") or "UNKNOWN") for x in (v1.get("claim_breakdown") or [])]
    right_statuses = [str((x or {}).get("status") or "UNKNOWN") for x in (v2.get("claim_breakdown") or [])]
    if left_statuses != right_statuses:
        diffs["claim_breakdown_statuses"] = {"v1": left_statuses, "v2": right_statuses}

    return {
        "parity": len(diffs) == 0,
        "diffs": diffs,
        "checked_keys": sorted(keys),
        "v1_schema_keys": _sorted_keys(v1),
        "v2_schema_keys": _sorted_keys(v2),
    }


def _sorted_keys(payload: Dict[str, Any]) -> List[str]:
    try:
        return sorted(str(k) for k in payload.keys())
    except Exception:
        return []
