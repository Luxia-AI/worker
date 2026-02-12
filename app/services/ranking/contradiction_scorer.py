from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


@dataclass
class ContraScore:
    contradicting_ids: List[int]
    supporting_ids: List[int]
    contra_count: int
    support_count: int


class ContradictionScorer:
    def __init__(self, *, semantic_min: float = 0.35, label_field: str = "stance") -> None:
        self.semantic_min = float(semantic_min)
        self.label_field = str(label_field)

    def score(self, evidence: list[Any]) -> ContraScore:
        contra: List[int] = []
        supp: List[int] = []
        for idx, ev in enumerate(evidence):
            sem = float(getattr(ev, "semantic", 0.0) or 0.0)
            if sem <= 0.0 and isinstance(ev, dict):
                sem = float(
                    ev.get("semantic")
                    or ev.get("semantic_score")
                    or ev.get("sem_score")
                    or ev.get("final_score")
                    or ev.get("score")
                    or 0.0
                )
            if sem < self.semantic_min:
                continue
            stance = (
                (getattr(ev, self.label_field, "") if not isinstance(ev, dict) else ev.get(self.label_field, "")) or ""
            ).upper()
            ev_id = getattr(ev, "id", None) if not isinstance(ev, dict) else ev.get("id")
            if ev_id is None:
                ev_id = idx
            if stance in {"CONTRADICTS", "CONTRA", "REFUTES"}:
                contra.append(int(ev_id))
            elif stance in {"SUPPORTS", "PRO", "ENTAILS"}:
                supp.append(int(ev_id))
        return ContraScore(
            contradicting_ids=contra,
            supporting_ids=supp,
            contra_count=len(contra),
            support_count=len(supp),
        )
