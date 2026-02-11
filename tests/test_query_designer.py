import pytest

from app.services.corrective.query_designer import CorrectiveQueryDesigner, QueryPlan, build_plan


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        ("The treatment improves blood pressure control in patients.", "EFFICACY"),
        ("This supplement is harmful and causes severe side effects.", "SAFETY"),
        ("The incidence is 12% among adults in this cohort.", "STATISTICAL"),
        ("The virus enters cells through the ACE2 receptor pathway.", "MECHANISM"),
        ("It is a myth that vaccines contain microchips.", "MYTH"),
        ("Everyone has a unique tongue print like fingerprints.", "UNIQUENESS"),
    ],
)
def test_classify_claim(claim: str, expected: str) -> None:
    designer = CorrectiveQueryDesigner()
    assert designer.classify_claim(claim) == expected


def test_extract_entities_lightweight_fields() -> None:
    designer = CorrectiveQueryDesigner()
    claim = "Adults taking aspirin reduce pain compared to placebo in children."

    entities = designer.extract_entities(claim)

    assert "adults" in entities.population
    assert entities.comparator
    assert entities.anchors


def test_build_plan_returns_6_to_8_queries_with_routes_and_negatives() -> None:
    plan = build_plan("Everyone has a unique tongue print in addition to fingerprints.")

    assert isinstance(plan, QueryPlan)
    assert 6 <= len(plan.queries) <= 8
    assert plan.claim_type == "UNIQUENESS"
    assert plan.negatives
    assert any(q.q.startswith("site:") or " site:" in q.q for q in plan.queries)
    assert any("-facebook" in q.q for q in plan.queries)


def test_query_plan_to_log_dict_shape() -> None:
    plan = build_plan("Vaccines are effective at preventing severe disease.")
    payload = plan.to_log_dict()

    assert set(payload.keys()) == {
        "claim",
        "claim_type",
        "extracted_entities",
        "queries",
        "negatives",
        "strategy_notes",
    }
    assert isinstance(payload["queries"], list)
    assert payload["queries"]
    assert {"q", "goal", "weight"} <= set(payload["queries"][0].keys())


def test_register_drift_from_url_updates_future_negatives() -> None:
    designer = CorrectiveQueryDesigner()
    claim = "Everyone has a unique tongue print in addition to fingerprints."

    before = designer.build_plan(claim)
    assert "pdf" not in before.negatives

    designer.register_drift_from_url(
        claim_type="UNIQUENESS",
        url="https://example.org/path/policy-manual.pdf",
        title="Operations manual",
        snippet="policy handbook for staff",
    )
    after = designer.build_plan(claim)

    assert any(x in after.negatives for x in {"pdf", "manual", "policy", "handbook"})


def test_quality_gate_and_adaptive_top_n() -> None:
    designer = CorrectiveQueryDesigner()
    claim = "Everyone has a unique tongue print in addition to fingerprints."
    claim_type = "UNIQUENESS"
    entities = designer.extract_entities(claim)

    good_1 = {
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
        "title": "Forensic study of unique tongue print patterns",
        "snippet": "Evidence for individuality and biometrics relevance.",
    }
    good_2 = {
        "url": "https://nature.com/articles/example",
        "title": "Unique tongue print individuality evidence",
        "snippet": "Biometrics and forensic discrimination study.",
    }
    bad = {
        "url": "https://quora.com/random-post",
        "title": "Community discussion thread",
        "snippet": "Personal opinions and anecdotes",
    }

    pass_1 = designer.quality_gate(claim_type, entities, **good_1)
    fail_1 = designer.quality_gate(claim_type, entities, **bad)
    assert pass_1.passed is True
    assert fail_1.passed is False

    assert designer.adaptive_top_n(claim_type, entities, [good_1, good_2, bad]) == 3
    assert designer.adaptive_top_n(claim_type, entities, [good_1, bad]) == 5
    assert designer.adaptive_top_n(claim_type, entities, [bad]) == 8
