from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class EvidenceFingerprint:
    snapshot_id: str


def make_evidence_snapshot_id(evidence: Iterable[Any], *, salt: str = "") -> EvidenceFingerprint:
    """
    Deterministic fingerprint of an ordered evidence list.
    Order intentionally affects the hash to preserve ranking semantics.
    """
    h = hashlib.sha256()
    h.update((salt or "").encode("utf-8"))
    for ev in evidence:
        ev_id = getattr(ev, "id", None) or getattr(ev, "eid", None) or ""
        domain = getattr(ev, "domain", None) or ""
        source = getattr(ev, "source", None) or getattr(ev, "source_url", None) or ""
        text = (getattr(ev, "text", None) or getattr(ev, "statement", None) or "")[:400]
        h.update(str(ev_id).encode("utf-8"))
        h.update(str(domain).encode("utf-8"))
        h.update(str(source).encode("utf-8"))
        h.update(str(text).encode("utf-8"))
        h.update(b"\n")
    return EvidenceFingerprint(snapshot_id=h.hexdigest())
