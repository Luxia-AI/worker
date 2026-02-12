from app.services.verdict.policy_override import OverrideSignals, therapeutic_strong_override


def test_therapeutic_override_paths():
    verdict, reason = therapeutic_strong_override(
        OverrideSignals(
            high_grade_support=0,
            high_grade_contra=1,
            relevant_noncurative=0,
            relevant_any=1,
        )
    )
    assert verdict == "FALSE"
    assert reason == "high_grade_contradiction"

    verdict, reason = therapeutic_strong_override(
        OverrideSignals(
            high_grade_support=1,
            high_grade_contra=0,
            relevant_noncurative=0,
            relevant_any=1,
        )
    )
    assert verdict == "TRUE"
    assert reason == "high_grade_support"

    verdict, reason = therapeutic_strong_override(
        OverrideSignals(
            high_grade_support=0,
            high_grade_contra=0,
            relevant_noncurative=1,
            relevant_any=1,
        )
    )
    assert verdict == "MISLEADING"
    assert reason == "noncurative_relevant_evidence"

    verdict, reason = therapeutic_strong_override(
        OverrideSignals(
            high_grade_support=0,
            high_grade_contra=0,
            relevant_noncurative=0,
            relevant_any=0,
        )
    )
    assert verdict == "UNVERIFIABLE"
    assert reason == "no_relevant_evidence"
