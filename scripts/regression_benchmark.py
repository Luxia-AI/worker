from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from app.services.eval.regression_suite import compute_regression_metrics, thresholds_pass


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("cases") or []
        if isinstance(rows, list):
            return [x for x in rows if isinstance(x, dict)]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Run claim-verification regression benchmark checks.")
    parser.add_argument("--input", required=True, help="Path to JSON rows with expected/predicted labels.")
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    parser.add_argument("--min-f1-true", type=float, default=0.82)
    parser.add_argument("--min-evidence-p3", type=float, default=0.70)
    args = parser.parse_args()

    rows = _load_rows(Path(args.input))
    metrics = compute_regression_metrics(rows)
    passed = thresholds_pass(
        metrics=metrics,
        min_accuracy=args.min_accuracy,
        min_f1_true=args.min_f1_true,
        min_evidence_precision_at_3=args.min_evidence_p3,
    )
    print(json.dumps({"metrics": metrics, "passed": bool(passed)}, ensure_ascii=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
