from app.services.verdict.verdict_generator import VerdictGenerator


def test_merge_evidence_is_deterministic_and_case_insensitive_dedup():
    vg = VerdictGenerator.__new__(VerdictGenerator)

    ranked = [
        {
            "statement": "Vaccines do not cause autism.",
            "final_score": 0.72,
            "source_url": "https://b.example/2",
        },
        {
            "statement": "The WHO says vaccines do not cause autism.",
            "final_score": 0.71,
            "source_url": "https://a.example/1",
        },
    ]
    segment = [
        {
            "statement": "vaccines do not cause autism.",
            "final_score": 0.90,
            "source_url": "https://z.example/9",
        },
        {
            "statement": "Vaccines do not cause the flu.",
            "final_score": 0.70,
            "source_url": "https://c.example/3",
        },
    ]

    merged = vg._merge_evidence(ranked, segment)

    # case-insensitive dedupe should keep a single autism statement
    autism_count = sum(
        1 for ev in merged if " ".join((ev["statement"] or "").lower().split()) == "vaccines do not cause autism."
    )
    assert autism_count == 1

    # deterministic sorting by score desc, then source, then statement
    keys = [vg._deterministic_evidence_sort_key(ev) for ev in merged]
    assert keys == sorted(keys)
