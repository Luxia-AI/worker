from app.services.corrective.pipeline import CorrectivePipeline


def test_fixed_threshold_uses_trust_post_not_top_score():
    # Even if a top ranking score would be high, threshold decision should use trust_post.
    # _top_ranking_score = 0.95
    payload = CorrectivePipeline._trust_payload_fixed(trust_post=0.61, threshold=0.70)

    assert payload["trust_metric_name"] == "trust_post"
    assert payload["trust_metric_value"] == 0.61
    assert payload["trust_threshold_met"] is False


def test_adaptive_threshold_uses_is_sufficient_flag():
    payload = CorrectivePipeline._trust_payload_adaptive({"trust_post": 0.41, "is_sufficient": True})
    assert payload["trust_metric_name"] == "adaptive_is_sufficient"
    assert payload["trust_metric_value"] == 0.41
    assert payload["trust_threshold_met"] is True
