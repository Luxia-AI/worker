from app.services.logic.evidence_strength import compute_evidence_strength


def test_evidence_strength_penalizes_hedged_and_rare_claims():
    ev = compute_evidence_strength(
        claim_text="This therapy prevents disease in all adults.",
        text_snippet="This therapy may help in rare cases, but evidence is inconclusive.",
        source_meta={"credibility": 0.9},
        stance_hint="SUPPORTS",
    )
    assert ev.hedge_penalty > 0.2
    assert ev.rarity_penalty > 0.2
    assert ev.support_strength < 0.7


def test_evidence_strength_supports_direct_high_quality_statement():
    ev = compute_evidence_strength(
        claim_text="Smoking increases lung cancer risk.",
        text_snippet="Meta-analysis confirmed smoking increases lung cancer risk.",
        source_meta={"credibility": 0.95},
        stance_hint="SUPPORTS",
    )
    assert ev.support_strength > 0.5
    assert ev.hedge_penalty < 0.2
