from app.services.eval.regression_suite import compute_regression_metrics, thresholds_pass


def test_regression_metrics_compute_expected_values():
    rows = [
        {
            "expected_verdict": "TRUE",
            "predicted_verdict": "TRUE",
            "expected_domains": ["pubmed.ncbi.nlm.nih.gov"],
            "predicted_domains": ["pubmed.ncbi.nlm.nih.gov", "nih.gov"],
        },
        {
            "expected_verdict": "FALSE",
            "predicted_verdict": "FALSE",
            "expected_domains": ["cdc.gov"],
            "predicted_domains": ["who.int", "cdc.gov"],
        },
        {
            "expected_verdict": "TRUE",
            "predicted_verdict": "FALSE",
            "expected_domains": ["cochrane.org"],
            "predicted_domains": ["example.org"],
        },
    ]
    metrics = compute_regression_metrics(rows)
    assert metrics["samples"] == 3.0
    assert metrics["accuracy"] == 0.6667
    assert metrics["f1_true"] > 0.6
    assert metrics["evidence_precision_at_3"] == 0.4


def test_regression_threshold_gate():
    metrics = {
        "accuracy": 0.90,
        "f1_true": 0.85,
        "evidence_precision_at_3": 0.72,
    }
    assert thresholds_pass(metrics, min_accuracy=0.85, min_f1_true=0.80, min_evidence_precision_at_3=0.70)
    assert not thresholds_pass(metrics, min_accuracy=0.95, min_f1_true=0.80, min_evidence_precision_at_3=0.70)
