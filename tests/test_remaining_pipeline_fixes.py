import asyncio

from app.services.ranking.hybrid_ranker import hybrid_rank
from app.services.verdict.verdict_generator import VerdictGenerator


def _vg() -> VerdictGenerator:
    vg = VerdictGenerator.__new__(VerdictGenerator)

    class _TrustPolicyStub:
        def decompose_claim(self, claim):
            return [claim]

        def compute_adaptive_trust(self, claim, evidence, top_k=10):
            return {
                "coverage": 0.85,
                "agreement": 0.9,
                "diversity": 0.6,
                "trust_post": 0.6,
                "is_sufficient": True,
                "num_subclaims": 1,
            }

    vg.trust_policy = _TrustPolicyStub()
    vg._log_subclaim_coverage = lambda *args, **kwargs: None
    vg._last_predicate_queries_generated = []
    return vg


def test_mrna_change_dna_forced_false_with_low_truthfulness():
    vg = _vg()
    claim = "mRNA COVID-19 vaccines change human DNA"
    evidence = [
        {
            "statement": "mRNA vaccines do not alter human DNA.",
            "source_url": "https://www.cdc.gov/refute",
            "final_score": 0.91,
            "credibility": 0.95,
        },
        {
            "statement": "Vaccine mRNA does not integrate into the human genome.",
            "source_url": "https://www.who.int/refute",
            "final_score": 0.88,
            "credibility": 0.95,
        },
        {
            "statement": "mRNA vaccines are used to prevent severe COVID-19.",
            "source_url": "https://example.org/background",
            "final_score": 0.56,
            "credibility": 0.7,
        },
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.95,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.91},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.88},
            {"evidence_id": 2, "statement": evidence[2]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.56},
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["verdict"] == "FALSE"
    assert out["truthfulness_percent"] <= 15.0
    assert out["confidence"] >= 0.75
    assert out["contradiction_override_fired"] is True


def test_background_only_evidence_stays_unknown_and_truthfulness_bounded():
    vg = _vg()
    claim = "X cures Y"
    evidence = [
        {
            "statement": "X is a commonly discussed intervention.",
            "source_url": "https://example.org/bg",
            "final_score": 0.7,
            "credibility": 0.7,
        }
    ]
    llm = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "PARTIALLY_VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.7}
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] == "UNKNOWN"
    assert out["truthfulness_percent"] <= 45.0
    assert out["truthfulness_invariant_applied"] is True


def test_mixed_support_and_explicit_refute_refute_wins():
    vg = _vg()
    claim = "Vitamin C always prevents the common cold"
    evidence = [
        {
            "statement": "Vitamin C may slightly reduce cold duration in some groups.",
            "source_url": "https://example.org/support",
            "final_score": 0.72,
            "credibility": 0.8,
        },
        {
            "statement": "Vitamin C does not prevent the common cold in the general population.",
            "source_url": "https://www.nih.gov/refute",
            "final_score": 0.89,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.92,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.72},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.89},
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["explicit_refutes_found"] is True
    assert out["claim_breakdown"][0]["status"] in {"INVALID", "PARTIALLY_INVALID"}


def test_unverifiable_confidence_cap_is_enforced():
    vg = _vg()
    claim = "Claim with no decisive evidence."
    evidence = [
        {
            "statement": "Background context only.",
            "source_url": "https://example.org",
            "final_score": 0.4,
            "credibility": 0.6,
        }
    ]
    llm = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.97,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": evidence[0]["statement"],
                "relevance": "NEUTRAL",
                "relevance_score": 0.4,
            }
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["verdict"] in {"PARTIALLY_TRUE", "UNVERIFIABLE"}
    assert out["verdict"] != "TRUE"
    assert out["confidence"] <= 0.60


def test_predicate_targeted_query_generation_patterns():
    vg = _vg()
    claim = "mRNA vaccines change your DNA"
    queries = vg._predicate_refute_query_hints(claim)
    text = " | ".join(q.lower() for q in queries)
    assert "do not" in text
    assert "cannot" in text
    assert "no evidence" in text
    assert "does not" in text


def test_valid_support_segment_not_downgraded_to_unverifiable():
    vg = _vg()
    claim = "Calcium builds strong bones"
    evidence = [
        {
            "statement": "Calcium is needed to build and maintain strong bones.",
            "source_url": "https://ods.od.nih.gov/factsheets/Calcium-Consumer/",
            "final_score": 0.58,
            "sem_score": 0.88,
            "credibility": 0.95,
        },
        {
            "statement": "Calcium is stored in bones and teeth.",
            "source_url": "https://ods.od.nih.gov/factsheets/Calcium-Consumer/",
            "final_score": 0.66,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.78,
        "rationale": "Claim is partially supported but phrasing is somewhat ambiguous.",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.73},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.36},
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE"}


def test_food_claim_not_refuted_by_supplement_only_evidence():
    vg = _vg()
    claim = "A healthy diet rich in vegetables and fruit may help reduce the risk of some cancers"
    evidence = [
        {
            "statement": (
                "There is no clear evidence that vitamin A supplementation decreases cancer risk "
                "in people consuming a healthy diet."
            ),
            "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK13273/",
            "final_score": 0.62,
            "credibility": 0.95,
        },
        {
            "statement": "Fruit and vegetables can help reduce risk of some cancers.",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/article/abc",
            "final_score": 0.71,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "FALSE",
        "confidence": 0.92,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "REFUTES", "relevance_score": 0.8},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.8},
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    ev_map = out["evidence_map"]
    supplement_row = next(x for x in ev_map if "supplementation" in x["statement"].lower())
    assert supplement_row["intervention_match"] is False
    assert supplement_row["relevance"] != "REFUTES"
    assert out["verdict"] != "FALSE"


def test_vitamin_c_immune_function_paraphrase_is_not_unknown():
    vg = _vg()
    claim = "Vitamin C contributes to the normal function of the immune system"
    evidence = [
        {
            "statement": "Vitamin C helps your immune system.",
            "source_url": "https://health.clevelandclinic.org/vitamin-c",
            "final_score": 0.60,
            "sem_score": 1.0,
            "credibility": 0.95,
        },
        {
            "statement": "Vitamin C supports immune health.",
            "source_url": "https://health.clevelandclinic.org/vitamin-c",
            "final_score": 0.50,
            "sem_score": 0.99,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.6,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.60},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.50},
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    segment = out["claim_breakdown"][0]
    assert segment["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE", "UNVERIFIABLE"}


def test_cross_subclaim_evidence_does_not_invalidate_unrelated_segment():
    vg = _vg()
    claim = "Vitamin D for bone growth in children and DHA for eye and brain development"
    evidence = [
        {
            "statement": "DHA is important for brain and eye development.",
            "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7468918/",
            "final_score": 0.91,
            "credibility": 0.95,
        }
    ]
    llm = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.7,
        "rationale": "test",
        "claim_breakdown": [
            {"claim_segment": "Vitamin D for bone growth in children", "status": "UNKNOWN"},
            {"claim_segment": "DHA for eye and brain development", "status": "VALID"},
        ],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.9}
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    segments = {item["claim_segment"]: item for item in out["claim_breakdown"]}
    assert segments["Vitamin D for bone growth in children"]["status"] == "UNKNOWN"
    assert segments["Vitamin D for bone growth in children"].get("supporting_fact", "") in {"", None}
    assert segments["DHA for eye and brain development"]["status"] in {"VALID", "PARTIALLY_VALID", "UNKNOWN"}


def test_preserve_partially_true_when_segments_are_mixed_or_unknown():
    vg = _vg()
    claim = "A supports B and C supports D"
    evidence = [
        {
            "statement": "A supports B in observational studies.",
            "source_url": "https://example.org/a-b",
            "final_score": 0.7,
            "credibility": 0.8,
        }
    ]
    payload = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.6,
        "truthfulness_percent": 55.0,
        "rationale": "mixed evidence",
        "claim_breakdown": [
            {"claim_segment": "A supports B", "status": "VALID", "supporting_fact": evidence[0]["statement"]},
            {"claim_segment": "C supports D", "status": "UNKNOWN"},
        ],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.7}
        ],
    }
    out = vg._enforce_binary_verdict_payload(claim, payload, evidence=evidence)
    assert out["verdict"] == "PARTIALLY_TRUE"
    statuses = [str(item.get("status") or "").upper() for item in out["claim_breakdown"]]
    assert "UNKNOWN" in statuses


def test_vitamin_d_necessary_for_bone_growth_not_unknown():
    vg = _vg()
    claim = "Vitamin D is needed for the normal growth and development of bone in children"
    evidence = [
        {
            "statement": "Vitamin D is necessary for normal bone growth.",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/21911908/",
            "final_score": 0.71,
            "sem_score": 0.0,
            "kg_score": 0.95,
            "credibility": 0.95,
        },
        {
            "statement": "Vitamin D helps the body use calcium and phosphorus to make strong bones and teeth.",
            "source_url": "https://www.cancer.gov/about-cancer/causes-prevention/risk/diet",
            "final_score": 0.42,
            "sem_score": 0.77,
            "kg_score": 0.0,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.8,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.71},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.42},
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    segment = out["claim_breakdown"][0]
    assert segment["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE", "UNVERIFIABLE"}


def test_as_part_of_low_saturated_fat_claim_not_unknown():
    vg = _vg()
    claim = "as part of a diet low in saturated fat and cholesterol"
    evidence = [
        {
            "statement": "Extra virgin olive oil is low in saturated fat and high in monounsaturated fatty acids.",
            "source_url": "https://health.clevelandclinic.org/foods-that-lower-cholesterol",
            "final_score": 0.57,
            "credibility": 0.95,
        },
        {
            "statement": "Saturated fat and cholesterol are mainly found in animal products.",
            "source_url": "https://health.clevelandclinic.org/foods-that-lower-cholesterol",
            "final_score": 0.55,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.60,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.35},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.31},
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    segment = out["claim_breakdown"][0]
    assert segment["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE", "UNVERIFIABLE"}


def test_soluble_fiber_from_foods_claim_not_unknown():
    vg = _vg()
    claim = "Soluble fiber from foods such as oat bran"
    evidence = [
        {
            "statement": "Soluble fiber is found in oats, beans, peas, and most fruits.",
            "source_url": "https://newsinhealth.nih.gov/2019/07/rough-up-your-diet",
            "final_score": 0.58,
            "credibility": 0.95,
        },
        {
            "statement": "Oats are a great source of soluble fiber.",
            "source_url": "https://newsinhealth.nih.gov/2019/07/rough-up-your-diet",
            "final_score": 0.55,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.60,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.38},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.36},
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    segment = out["claim_breakdown"][0]
    assert segment["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE"}


def test_iron_claim_not_validated_by_unrelated_vitamin_k_fact():
    vg = _vg()
    claim = "Iron contributes to the normal formation of red blood cells and hemoglobin"
    evidence = [
        {
            "statement": "Vitamin K is needed in the liver for formation of several blood clotting factors.",
            "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK13273/",
            "final_score": 0.62,
            "credibility": 0.95,
        }
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.8,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.62}
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] == "UNKNOWN"
    assert out["verdict"] == "UNVERIFIABLE"


def test_stanol_and_risk_factor_semicolon_claim_not_unknown():
    vg = _vg()
    claim = (
        "Plant stanol esters have been shown to reduce blood cholesterol; "
        "blood cholesterol is a risk factor in the development of coronary heart disease"
    )
    evidence = [
        {
            "statement": "Plant stanol esters have been shown to reduce blood cholesterol.",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/25411276/",
            "final_score": 0.70,
            "credibility": 0.95,
        },
        {
            "statement": "Blood cholesterol is a risk factor in the development of coronary heart disease.",
            "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK13273/",
            "final_score": 0.68,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.6,
        "rationale": "test",
        "claim_breakdown": [
            {"claim_segment": "Plant stanol esters have been shown to reduce blood cholesterol", "status": "UNKNOWN"},
            {
                "claim_segment": "blood cholesterol is a risk factor in the development of coronary heart disease",
                "status": "UNKNOWN",
            },
        ],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.70},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.68},
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    statuses = {s["status"] for s in out["claim_breakdown"]}
    assert "UNKNOWN" not in statuses
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE"}


def test_event_schedule_snippet_not_treated_as_medical_support():
    vg = _vg()
    claim = "Regular caffeine consumption from coffee is linked to a lower risk of depression"
    statement = "Welcome & networking coffee took place from 08:30-10:00"

    assert vg._segment_topic_guard_ok(claim, statement) is False
    assert vg.compute_predicate_match(claim, statement) == 0.0


def test_true_verdict_downgraded_when_only_partial_support():
    vg = _vg()
    claim = "Iron contributes to the normal formation of red blood cells and hemoglobin"
    evidence = [
        {
            "statement": (
                "Cell content of hemoglobin precursors can help clarify heme synthesis " "and iron incorporation."
            ),
            "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK13273/",
            "final_score": 0.58,
            "credibility": 0.95,
        }
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.82,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.58}
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["verdict"] != "TRUE"


def test_laughter_claim_does_not_validate_blood_sugar_segment_with_unrelated_fact():
    vg = _vg()
    claim = (
        "Laughter releases endorphins that can decrease pain and "
        "has been shown to help lower blood sugar levels after a meal"
    )
    evidence = [
        {
            "statement": "Regular exercise can boost endorphins, which are natural pain fighters.",
            "source_url": "https://health.clevelandclinic.org/foods-to-reduce-uterine-fibroids",
            "final_score": 0.49,
            "credibility": 0.95,
        },
        {
            "statement": "Carbohydrates help control blood glucose and insulin metabolism.",
            "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK459280/",
            "final_score": 0.29,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.92,
        "rationale": "test",
        "claim_breakdown": [
            {"claim_segment": "Laughter releases endorphins that can decrease pain", "status": "VALID"},
            {"claim_segment": "has been shown to help lower blood sugar levels after a meal", "status": "VALID"},
        ],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.49},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.29},
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    statuses = [s.get("status") for s in (out.get("claim_breakdown") or [])]
    assert "UNKNOWN" in statuses
    assert out["verdict"] != "TRUE"


def test_key_findings_are_grounded_and_empty_when_all_segments_unknown():
    vg = _vg()
    claim = "Regular caffeine consumption from coffee is linked to a lower risk of depression"
    evidence = [
        {
            "statement": "Welcome & networking coffee took place from 08:30-10:00",
            "source_url": "https://example.org/event",
            "final_score": 0.40,
            "credibility": 0.5,
        }
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "key_findings": ["Caffeine reduces depression risk."],
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.4}
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] == "UNKNOWN"
    assert out["key_findings"] == []


def test_vitamin_c_immune_function_paraphrase_not_unverifiable():
    vg = _vg()
    claim = "Vitamin C contributes to the normal function of the immune system"
    evidence = [
        {
            "statement": "Vitamin C helps your immune system.",
            "source_url": "https://health.clevelandclinic.org/vitamin-c",
            "final_score": 0.60,
            "sem_score": 1.0,
            "credibility": 0.95,
        },
        {
            "statement": "Vitamin C supports immune health.",
            "source_url": "https://health.clevelandclinic.org/vitamin-c",
            "final_score": 0.50,
            "sem_score": 0.99,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "UNVERIFIABLE",
        "confidence": 0.6,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.6},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.5},
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE"}


def test_predicate_guard_generic_requirement_paraphrase():
    # Claim-agnostic requirement/support mapping.
    assert VerdictGenerator._predicate_guard_ok(
        "Magnesium is needed for normal muscle function",
        "Magnesium supports normal muscle function.",
    )


def test_segment_recovery_query_hints_are_generic():
    vg = _vg()
    hints = vg._segment_recovery_query_hints("Zinc contributes to normal cognitive performance")
    joined = " ".join(hints).lower()
    assert "evidence" in joined
    assert "mechanism" in joined


def test_intervention_alignment_not_broken_by_plain_vitamin_mentions():
    vg = _vg()
    claim = (
        "Low-fat diets rich in fruits and vegetables may contain dietary fiber and Vitamin A or Vitamin C "
        "and may reduce cancer risk"
    )
    evidence = "Fruits and vegetables can help reduce risk of some cancers."
    intervention_match, anchors_ok = vg._intervention_alignment(claim, evidence)
    assert intervention_match is True
    assert anchors_ok is True


def test_parenthetical_diet_claim_not_stuck_unknown_with_clear_support():
    vg = _vg()
    claim = (
        "Low-fat diets rich in fruits and vegetables (foods that are low in fat and may contain dietary fiber "
        "& Vitamin A or Vitamin C) may reduce the risk of some types of cancer"
    )
    evidence = [
        {
            "statement": "Fruits and vegetables lower the risk of certain types of cancer.",
            "source_url": "https://health.clevelandclinic.org/example",
            "final_score": 0.52,
            "credibility": 0.95,
        },
        {
            "statement": "Grains, fruits, and vegetables contain dietary fiber.",
            "source_url": "https://pubmed.ncbi.nlm.nih.gov/10089116/",
            "final_score": 0.56,
            "credibility": 0.95,
        },
    ]
    llm = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.6,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "UNKNOWN"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.52},
            {"evidence_id": 1, "statement": evidence[1]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.56},
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] in {"VALID", "PARTIALLY_VALID"}
    assert out["verdict"] in {"TRUE", "PARTIALLY_TRUE"}


def test_cultures_in_claim_uses_improve_as_canonical_predicate():
    vg = _vg()
    triplet = vg._extract_canonical_predicate_triplet(
        "Live cultures in yogurt improve lactose digestion in individuals who have difficulty digesting lactose"
    )
    canonical = str(triplet.get("canonical_predicate") or "")
    assert canonical == "improve"


def test_hybrid_rank_filters_off_action_evidence_for_assertive_claim():
    query = "Live cultures in yogurt improve lactose digestion in individuals who have difficulty digesting lactose"
    semantic = [
        {
            "statement": "Probiotics are often found in yogurt and dietary supplements.",
            "source_url": "https://www.mayoclinic.org/example",
            "score": 0.95,
            "entities": ["probiotics", "yogurt"],
        },
        {
            "statement": (
                "Administration improved clinical outcomes for lactose digestion " "in lactose-intolerant individuals."
            ),
            "source_url": "https://pmc.ncbi.nlm.nih.gov/example",
            "score": 0.72,
            "entities": ["lactose digestion", "lactose intolerance"],
        },
    ]
    ranked = hybrid_rank(semantic_results=semantic, kg_results=[], query_entities=[], query_text=query)
    assert ranked
    assert "improved clinical outcomes for lactose digestion" in ranked[0]["statement"].lower()


def test_action_claim_true_is_blocked_when_evidence_map_is_neutral_only():
    vg = _vg()
    claim = "Omega-3 supplements prevent heart attacks"
    evidence = [
        {
            "statement": "A 28% reduction in heart attacks was found for adults taking omega-3s vs placebo.",
            "source_url": "https://www.nhlbi.nih.gov/news/20\
            24/omega-3s-heart-health-exploring-potential-benefits-and-risks",
            "final_score": 0.36,
            "credibility": 0.7,
        }
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.45}
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["verdict"] in {"PARTIALLY_TRUE", "UNVERIFIABLE"}
    assert out["verdict"] != "TRUE"


def test_subjectless_fragment_claim_is_evaluated_conservatively():
    vg = _vg()
    claim = "may reduce the risk of type 2 diabetes"
    evidence = [
        {
            "statement": "staying active lowers risk of type 2 diabetes",
            "source_url": (
                "https://odphp.health.gov/myhealthfinder/health-conditions/obesity/"
                "stay-active-you-get-older-quick-tips"
            ),
            "final_score": 0.77,
            "credibility": 0.95,
        }
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.92,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.77}
        ],
    }

    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["verdict"] != "TRUE"
    assert out["truthfulness_percent"] <= 60.0
    assert out["analysis_counts"]["claim_fragmentary"] is True


def test_negated_claim_not_validated_by_positive_support_fact():
    vg = _vg()
    claim = "Vitamin C does not support immune health"
    evidence = [
        {
            "statement": "Vitamin C supports immune health.",
            "source_url": "https://health.clevelandclinic.org/vitamin-c",
            "final_score": 0.70,
            "credibility": 0.95,
        }
    ]
    llm = {
        "verdict": "TRUE",
        "confidence": 0.9,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "SUPPORTS", "relevance_score": 0.7}
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["verdict"] != "TRUE"
    assert out["claim_breakdown"][0]["status"] in {"INVALID", "PARTIALLY_INVALID", "UNKNOWN"}


def test_liver_detox_claim_not_validated_by_generic_water_fact():
    vg = _vg()
    claim = "Drinking lemon water detoxifies the liver"
    evidence = [
        {
            "statement": "Drinking water also keeps your teeth and mouth healthy.",
            "source_url": "https://www.healthdirect.gov.au/drinking-water-and-your-health",
            "final_score": 0.42,
            "credibility": 0.7,
        }
    ]
    llm = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.75,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "VALID"}],
        "evidence_map": [
            {"evidence_id": 0, "statement": evidence[0]["statement"], "relevance": "NEUTRAL", "relevance_score": 0.42}
        ],
    }
    out = vg._parse_verdict_result(llm, claim, evidence)
    assert out["claim_breakdown"][0]["status"] == "UNKNOWN"


def test_llm_rationale_generation_uses_claim_breakdown_and_relevant_evidence():
    vg = _vg()

    captured = {"prompt": ""}

    class _FakeLLM:
        async def ainvoke(self, prompt, response_format="json", priority=None, temperature=0.0, call_tag=""):
            captured["prompt"] = prompt
            return {"rationale": "According to the cited studies, the claim is supported with some limits."}

    vg.llm_service = _FakeLLM()
    payload = {
        "verdict": "PARTIALLY_TRUE",
        "rationale": "fallback rationale",
        "claim_breakdown": [
            {
                "claim_segment": "Poor sleep is a significant risk factor for obesity",
                "status": "PARTIALLY_VALID",
                "supporting_fact": "Short sleep duration is a new risk factor for obesity.",
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3632337",
            }
        ],
        "evidence_map": [
            {
                "statement": "Short sleep duration is a new risk factor for obesity.",
                "relevance": "SUPPORTS",
                "relevance_score": 0.62,
                "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3632337",
            },
            {
                "statement": "Background context statement.",
                "relevance": "NEUTRAL",
                "relevance_score": 0.10,
                "source_url": "https://example.org/context",
            },
        ],
    }

    rationale = asyncio.run(vg._generate_llm_rationale("Poor sleep and obesity risk", payload))
    assert "supported" in rationale.lower()
    assert "claim breakdown" in captured["prompt"].lower()
    assert "relevant evidence only" in captured["prompt"].lower()
    assert "background context statement" not in captured["prompt"].lower()


def test_rationale_is_humanized_and_not_old_template():
    vg = _vg()
    claim = "Diets rich in fruits and vegetables may reduce the risk of certain cancers"
    payload = {
        "verdict": "PARTIALLY_TRUE",
        "confidence": 0.7,
        "truthfulness_percent": 58.0,
        "rationale": "test",
        "claim_breakdown": [{"claim_segment": claim, "status": "PARTIALLY_VALID"}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Fruit and vegetable intake is associated with lower risk for some cancers.",
                "relevance": "SUPPORTS",
                "relevance_score": 0.67,
            }
        ],
    }
    out = vg._enforce_binary_verdict_payload(
        claim,
        payload,
        evidence=[
            {
                "statement": "Fruit and vegetable intake is associated with lower risk for some cancers.",
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/10089116/",
            }
        ],
    )
    r = str(out.get("rationale") or "")
    assert "Based on the strongest direct evidence" not in r
    assert "at a glance" in r.lower() or "according to the available evidence" in r.lower()
    assert "evidence summary" in r.lower()


def test_rationale_includes_key_evidence_reference_when_available():
    vg = _vg()
    claim = "Moderate coffee consumption does not cause dehydration"
    payload = {
        "verdict": "FALSE",
        "confidence": 0.75,
        "truthfulness_percent": 20.0,
        "rationale": "",
        "claim_breakdown": [{"claim_segment": claim, "status": "INVALID"}],
        "evidence_map": [
            {
                "evidence_id": 0,
                "statement": "Caffeine can cause dehydration.",
                "relevance": "REFUTES",
                "relevance_score": 0.83,
                "source_url": "https://pubmed.ncbi.nlm.nih.gov/22740040/",
            }
        ],
    }
    out = vg._enforce_binary_verdict_payload(
        claim,
        payload,
        evidence=[
            {"statement": "Caffeine can cause dehydration.", "source_url": "https://pubmed.ncbi.nlm.nih.gov/22740040/"}
        ],
    )
    r = str(out.get("rationale") or "")
    assert "key evidence" in r.lower()
    assert "pubmed.ncbi.nlm.nih.gov" in r.lower()
