from dataclasses import dataclass

from app.services.verdict.rationale_filter import filter_rationale


@dataclass
class _Ev:
    id: int
    semantic: float
    stance: str
    text: str


def test_rationale_filter_excludes_safety_only_statements():
    evidence = [
        _Ev(
            id=1,
            semantic=0.91,
            stance="CONTRADICTS",
            text="Clinical trial evidence shows no cure efficacy for this treatment.",
        ),
        _Ev(
            id=2,
            semantic=0.95,
            stance="CONTRADICTS",
            text="Drinking after surgery is safe and promotes recovery.",
        ),
    ]

    filtered = filter_rationale(evidence, required_polarity="CONTRADICTS", semantic_min=0.40)
    assert [ev.id for ev in filtered] == [1]
