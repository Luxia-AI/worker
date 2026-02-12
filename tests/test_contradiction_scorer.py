from dataclasses import dataclass

from app.services.ranking.contradiction_scorer import ContradictionScorer


@dataclass
class _Ev:
    id: int
    semantic: float
    stance: str


def test_contradiction_scorer_respects_semantic_threshold():
    scorer = ContradictionScorer(semantic_min=0.40)
    evidence = [
        _Ev(1, 0.90, "CONTRADICTS"),
        _Ev(2, 0.39, "CONTRADICTS"),  # below threshold
        _Ev(3, 0.80, "SUPPORTS"),
    ]
    result = scorer.score(evidence)
    assert result.contra_count == 1
    assert result.support_count == 1
    assert result.contradicting_ids == [1]
    assert result.supporting_ids == [3]
