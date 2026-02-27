from __future__ import annotations

from typing import Any, Dict, List


def _normalize_binary_verdict(value: Any) -> str:
    verdict = str(value or "").strip().upper()
    if verdict in {"TRUE", "FALSE"}:
        return verdict
    if verdict in {"PARTIALLY_TRUE", "MOSTLY_TRUE"}:
        return "TRUE"
    if verdict in {"PARTIALLY_FALSE", "MISLEADING"}:
        return "FALSE"
    return ""


def compute_regression_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = 0
    correct = 0
    tp = fp = fn = 0
    evidence_hits = 0
    evidence_total = 0

    for row in rows or []:
        expected = _normalize_binary_verdict(row.get("expected_verdict"))
        predicted = _normalize_binary_verdict(row.get("predicted_verdict"))
        if not expected or not predicted:
            continue
        total += 1
        if expected == predicted:
            correct += 1
        if predicted == "TRUE" and expected == "TRUE":
            tp += 1
        elif predicted == "TRUE" and expected == "FALSE":
            fp += 1
        elif predicted == "FALSE" and expected == "TRUE":
            fn += 1

        expected_domains = {str(x).strip().lower() for x in (row.get("expected_domains") or []) if str(x).strip()}
        predicted_domains = [str(x).strip().lower() for x in (row.get("predicted_domains") or []) if str(x).strip()][:3]
        if expected_domains and predicted_domains:
            evidence_total += len(predicted_domains)
            evidence_hits += sum(1 for d in predicted_domains if d in expected_domains)

    accuracy = (correct / total) if total else 0.0
    precision_true = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall_true = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1_true = (
        (2.0 * precision_true * recall_true / (precision_true + recall_true)) if (precision_true + recall_true) else 0.0
    )
    evidence_precision_at_3 = (evidence_hits / evidence_total) if evidence_total else 0.0
    return {
        "samples": float(total),
        "accuracy": float(round(accuracy, 4)),
        "f1_true": float(round(f1_true, 4)),
        "evidence_precision_at_3": float(round(evidence_precision_at_3, 4)),
    }


def thresholds_pass(
    metrics: Dict[str, float],
    min_accuracy: float,
    min_f1_true: float,
    min_evidence_precision_at_3: float,
) -> bool:
    return (
        float(metrics.get("accuracy", 0.0)) >= float(min_accuracy)
        and float(metrics.get("f1_true", 0.0)) >= float(min_f1_true)
        and float(metrics.get("evidence_precision_at_3", 0.0)) >= float(min_evidence_precision_at_3)
    )
