import argparse
import asyncio
import json
import os
import re
import sys
import types
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import aiohttp

_WORKER_DEPS: Dict[str, Any] | None = None


def _get_worker_deps() -> Dict[str, Any]:
    """
    Lazy-load worker modules so this script works when executed as:
    `python worker/scripts/cse_query_replay.py`
    while keeping module-level imports lint-clean.
    """
    global _WORKER_DEPS
    if _WORKER_DEPS is not None:
        return _WORKER_DEPS

    worker_root = Path(__file__).resolve().parents[1]
    if str(worker_root) not in sys.path:
        sys.path.insert(0, str(worker_root))

    from app.constants.config import GOOGLE_CSE_SEARCH_URL, GOOGLE_CSE_TIMEOUT
    from app.services.corrective.trusted_search import TrustedSearch
    from app.services.ranking.adaptive_trust_policy import AdaptiveTrustPolicy

    _WORKER_DEPS = {
        "GOOGLE_CSE_SEARCH_URL": GOOGLE_CSE_SEARCH_URL,
        "GOOGLE_CSE_TIMEOUT": GOOGLE_CSE_TIMEOUT,
        "TrustedSearch": TrustedSearch,
        "AdaptiveTrustPolicy": AdaptiveTrustPolicy,
    }
    return _WORKER_DEPS


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "to",
    "for",
    "of",
    "in",
    "on",
    "with",
    "by",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "through",
}


def _tokenize(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b", (text or "").lower())
        if t not in _STOPWORDS and len(t) > 2
    }


def _load_claims(path: str | None, inline_claims: List[str]) -> List[str]:
    claims: List[str] = []
    if path:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw or raw.startswith("#"):
                    continue
                if raw.startswith("{") and raw.endswith("}"):
                    try:
                        obj = json.loads(raw)
                        claim = str(obj.get("claim") or obj.get("text") or obj.get("post_text") or "").strip()
                        if claim:
                            claims.append(claim)
                            continue
                    except Exception:
                        pass
                claims.append(raw)

    for claim in inline_claims or []:
        c = (claim or "").strip()
        if c:
            claims.append(c)

    out: List[str] = []
    for c in claims:
        if c not in out:
            out.append(c)
    return out


def _item_text(item: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("snippet") or ""),
            str(item.get("link") or ""),
        ]
    ).lower()


def _subclaim_profiles(ts: Any, claim: str) -> List[Dict[str, Any]]:
    deps = _get_worker_deps()
    policy = deps["AdaptiveTrustPolicy"]()
    raw_subclaims = policy.decompose_claim(claim)
    subclaims = ts.merge_subclaims([s for s in raw_subclaims if s and s.strip()])
    if not subclaims:
        subclaims = [claim]

    profiles: List[Dict[str, Any]] = []
    for sub in subclaims:
        anchors = ts._extract_subclaim_anchors(sub, entities=[])
        phrases = [a for a in anchors if " " in a]
        token_pool = set()
        token_pool |= _tokenize(sub)
        for a in anchors:
            token_pool |= _tokenize(a)
        profiles.append(
            {
                "subclaim": sub,
                "tokens": token_pool,
                "phrases": phrases,
            }
        )
    return profiles


def _relevance_score(item_text: str, profile: Dict[str, Any]) -> float:
    tokens = profile["tokens"]
    if not tokens:
        return 0.0
    overlap = len(tokens & _tokenize(item_text)) / max(1, len(tokens))
    phrases = profile["phrases"]
    phrase_hit = 1.0 if any(p in item_text for p in phrases) else 0.0
    return (0.75 * overlap) + (0.25 * phrase_hit)


async def _query_cse(
    session: aiohttp.ClientSession,
    api_key: str,
    cse_id: str,
    query: str,
    num: int,
) -> Tuple[List[Dict[str, Any]], str]:
    deps = _get_worker_deps()
    cse_search_url = deps["GOOGLE_CSE_SEARCH_URL"]
    cse_timeout = deps["GOOGLE_CSE_TIMEOUT"]
    encoded = urllib.parse.quote_plus(query)
    url = cse_search_url.format(key=api_key, cse=cse_id, query=encoded) + f"&num={num}"
    try:
        async with session.get(url, timeout=cse_timeout) as resp:
            data = await resp.json(content_type=None)
    except Exception as e:
        return [], f"request_error:{e}"

    if "error" in data:
        msg = str((data.get("error") or {}).get("message") or data.get("error"))
        return [], f"api_error:{msg}"

    items = data.get("items") or []
    cleaned: List[Dict[str, Any]] = []
    for it in items:
        cleaned.append(
            {
                "title": it.get("title", ""),
                "snippet": it.get("snippet", ""),
                "link": it.get("link", ""),
            }
        )
    return cleaned, ""


async def _evaluate_claim(
    ts: Any,
    session: aiohttp.ClientSession,
    claim: str,
    cse_id: str,
    api_key: str,
    max_queries: int,
    num_results: int,
    relevance_threshold: float,
) -> Dict[str, Any]:
    profiles = _subclaim_profiles(ts, claim)
    queries = await ts.generate_search_queries(
        post_text=claim,
        failed_entities=[],
        max_queries=max_queries,
        subclaims=[p["subclaim"] for p in profiles],
        entities=[],
    )
    if not queries:
        queries = [claim]

    covered = [False for _ in profiles]
    query_rows: List[Dict[str, Any]] = []
    total_results = 0
    total_relevant = 0
    total_trusted = 0

    for q in queries:
        items, error = await _query_cse(session, api_key=api_key, cse_id=cse_id, query=q, num=num_results)
        query_total = len(items)
        query_relevant = 0
        query_trusted = 0
        score_samples: List[float] = []

        for it in items:
            txt = _item_text(it)
            if ts.is_trusted(it.get("link", "")):
                query_trusted += 1

            best_idx = -1
            best_score = 0.0
            for idx, profile in enumerate(profiles):
                score = _relevance_score(txt, profile)
                if score > best_score:
                    best_score = score
                    best_idx = idx
            score_samples.append(best_score)
            if best_score >= relevance_threshold:
                query_relevant += 1
                if best_idx >= 0:
                    covered[best_idx] = True

        total_results += query_total
        total_relevant += query_relevant
        total_trusted += query_trusted

        query_rows.append(
            {
                "query": q,
                "results": query_total,
                "relevant": query_relevant,
                "precision": round((query_relevant / max(1, query_total)), 3),
                "trusted_results": query_trusted,
                "allowlist_precision": round((query_trusted / max(1, query_total)), 3),
                "avg_relevance_score": round(sum(score_samples) / max(1, len(score_samples)), 3),
                "error": error,
            }
        )

    coverage = sum(1 for x in covered if x) / max(1, len(covered))
    uncovered = [profiles[i]["subclaim"] for i, ok in enumerate(covered) if not ok]
    avg_query_precision = sum(row["precision"] for row in query_rows) / max(1, len(query_rows))
    avg_allowlist_precision = sum(row["allowlist_precision"] for row in query_rows) / max(1, len(query_rows))

    return {
        "claim": claim,
        "queries": query_rows,
        "queries_generated": len(queries),
        "subclaims": [p["subclaim"] for p in profiles],
        "subclaim_coverage": round(coverage, 3),
        "first_page_precision": round(total_relevant / max(1, total_results), 3),
        "avg_query_precision": round(avg_query_precision, 3),
        "first_page_allowlist_precision": round(total_trusted / max(1, total_results), 3),
        "avg_query_allowlist_precision": round(avg_allowlist_precision, 3),
        "uncovered_subclaims": uncovered,
    }


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    claims = _load_claims(args.claims_file, args.claim)
    if not claims:
        raise RuntimeError("No claims provided. Use --claim or --claims-file.")

    api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY", "")
    cse_id = args.cse_id or os.getenv("GOOGLE_CSE_ID", "")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY. Set env var or pass --google-api-key.")
    if not cse_id:
        raise RuntimeError("Missing CSE id. Set GOOGLE_CSE_ID or pass --cse-id.")

    deps = _get_worker_deps()
    trusted_search_cls = deps["TrustedSearch"]
    ts = trusted_search_cls(google_api_key=api_key, google_cse_id=cse_id, serper_api_key=os.getenv("SERPER_API_KEY"))

    if args.deterministic_only:

        async def _no_llm(self, text, failed_entities, entities=None, subclaims=None):  # noqa: ANN001
            return []

        ts.reformulate_queries = types.MethodType(_no_llm, ts)

    claim_reports: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession() as session:
        for idx, claim in enumerate(claims, start=1):
            report = await _evaluate_claim(
                ts=ts,
                session=session,
                claim=claim,
                cse_id=cse_id,
                api_key=api_key,
                max_queries=args.max_queries,
                num_results=args.results_per_query,
                relevance_threshold=args.relevance_threshold,
            )
            claim_reports.append(report)
            print(
                f"[{idx}/{len(claims)}] coverage={report['subclaim_coverage']:.3f} "
                f"precision={report['first_page_precision']:.3f} "
                f"allowlist_precision={report['first_page_allowlist_precision']:.3f}"
            )
            print(f"claim: {claim}")
            for row in report["queries"][: min(5, len(report["queries"]))]:
                print(
                    "  - "
                    f"p={row['precision']:.3f} "
                    f"allow={row['allowlist_precision']:.3f} "
                    f"n={row['results']} q={row['query']}"
                )
            if report["uncovered_subclaims"]:
                print(f"  uncovered_subclaims={len(report['uncovered_subclaims'])}")

    overall = {
        "claims": len(claim_reports),
        "avg_subclaim_coverage": round(
            sum(r["subclaim_coverage"] for r in claim_reports) / max(1, len(claim_reports)),
            3,
        ),
        "avg_first_page_precision": round(
            sum(r["first_page_precision"] for r in claim_reports) / max(1, len(claim_reports)),
            3,
        ),
        "avg_first_page_allowlist_precision": round(
            sum(r["first_page_allowlist_precision"] for r in claim_reports) / max(1, len(claim_reports)),
            3,
        ),
    }
    return {"overall": overall, "claims": claim_reports, "cse_id": cse_id}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay claim batch against Google CSE and score first-page precision/coverage."
    )
    parser.add_argument("--claim", action="append", default=[], help="Claim text (repeatable).")
    parser.add_argument("--claims-file", help="Path to TXT or JSONL with claims.")
    parser.add_argument("--google-api-key", help="Google API key (defaults to env GOOGLE_API_KEY).")
    parser.add_argument("--cse-id", default="3326f393eaed144c1", help="Google CSE cx id.")
    parser.add_argument("--max-queries", type=int, default=6, help="Max generated queries per claim.")
    parser.add_argument("--results-per-query", type=int, default=10, help="Google CSE results per query (max 10).")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.22,
        help="Lexical relevance threshold for counting a result as relevant.",
    )
    parser.add_argument(
        "--deterministic-only",
        action="store_true",
        help="Disable LLM reformulation and evaluate deterministic query planner only.",
    )
    parser.add_argument("--output-json", help="Optional path to write full JSON report.")
    args = parser.parse_args()

    result = asyncio.run(_run(args))
    print("\nOverall:")
    print(json.dumps(result["overall"], indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=True)
        print(f"Saved report: {args.output_json}")


if __name__ == "__main__":
    main()
