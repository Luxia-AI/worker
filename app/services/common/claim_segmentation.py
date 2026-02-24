from __future__ import annotations

import re
from typing import List

_CONJUNCTIVES = [" and ", " or ", " but ", " however ", " although ", " while "]
_PREDICATE_RE = re.compile(
    (
        r"\b("
        r"helps?|prevents?|reduces?|increases?|causes?|improves?|worsens?|protects?|"
        r"needed for|required for|important for|"
        r"associated with|linked to|leads to|"
        r"contribut(?:e|es|ed|ing)\s+to|support(?:s|ed|ing)?\s+"
        r"|required for|necessary for|needed for|important for|"
        r"do not cause|does not cause|don't cause|doesn't cause|cannot cause|can't cause|"
        r"is|are|was|were"
        r")\b"
    ),
    flags=re.IGNORECASE,
)
_CLAUSE_VERB_RE = re.compile(
    (
        r"\b("
        r"is|are|was|were|be|been|being|do|does|did|have|has|had|"
        r"helps?|prevents?|reduces?|increases?|causes?|improves?|worsens?|protects?|"
        r"contribut(?:e|es|ed|ing)|support(?:s|ed|ing)?|"
        r"needed|required|important|"
        r"linked|associated|leads?"
        r")\b"
    ),
    flags=re.IGNORECASE,
)
_LIST_BULLETS_RE = re.compile(r"\s*[-*•]\s+")
_CONTRAST_VERB_RE = re.compile(
    r"\b(kills?|cures?|treats?|prevents?|causes?|contains?|targets?|affects?|works?)\b",
    flags=re.IGNORECASE,
)
_AUX_LEADING_RE = re.compile(
    r"^(?:has|have|had|is|are|was|were|can|could|may|might|must|should|would|will|been|being)\b",
    flags=re.IGNORECASE,
)
_RELATIVE_CLAUSE_RE = re.compile(
    r"^(?P<head>.+?)\s+\b(?:that|which|who)\b\s+(?P<tail>.+)$",
    flags=re.IGNORECASE,
)


def _clean(text: str, keep_conj_prefix: bool = True) -> str:
    s = re.sub(r"\s+", " ", (text or "")).strip(" ,.")
    if not keep_conj_prefix:
        s = re.sub(r"^(and|or|but|however|although|while)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r",?\s*according to (medical|scientific) research\.?$", "", s, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", s).strip(" ,.")


def _extract_subject_predicate(text: str) -> str:
    last = None
    for m in _PREDICATE_RE.finditer(text or ""):
        last = m
    if not last:
        return ""
    return _clean((text or "")[: last.end()])


def _extract_subject_head(text: str) -> str:
    t = _clean(text, keep_conj_prefix=False)
    if not t:
        return ""
    tokens = re.findall(r"[A-Za-z][\w'-]*", t)
    if not tokens:
        return ""
    boundaries = {
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "because",
        "while",
        "although",
        "though",
        "if",
        "when",
        "where",
        "after",
        "before",
    }
    verb_like = {
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "can",
        "could",
        "may",
        "might",
        "must",
        "should",
        "would",
        "will",
        "help",
        "helps",
        "reduce",
        "reduces",
        "increase",
        "increases",
        "decrease",
        "decreases",
        "improve",
        "improves",
        "cause",
        "causes",
        "prevent",
        "prevents",
        "contain",
        "contains",
        "link",
        "linked",
    }
    head: List[str] = []
    for idx, tok in enumerate(tokens):
        low = tok.lower()
        if idx > 0 and low in boundaries:
            break
        if idx > 0 and (
            low in verb_like or low.endswith("ed") or low.endswith("ing") or (low.endswith("s") and len(low) > 3)
        ):
            break
        head.append(tok)
        if len(head) >= 5:
            break
    if not head:
        return tokens[0]
    return _clean(" ".join(head), keep_conj_prefix=False)


def _looks_independent_clause(text: str) -> bool:
    t = _clean(text, keep_conj_prefix=False).lower()
    if not t:
        return False
    tokens = re.findall(r"\b[\w']+\b", t)
    if len(tokens) < 2:
        return False
    # Pronoun + verb is likely an independent clause ("they reduce ...").
    if tokens[0] in {"it", "they", "we", "he", "she", "you"} and _CLAUSE_VERB_RE.search(t):
        return True
    return bool(_CLAUSE_VERB_RE.search(t))


def _expand_enumeration(sentence: str) -> List[str]:
    pred = _PREDICATE_RE.search(sentence or "")
    if not pred:
        return []
    head = _clean((sentence or "")[: pred.start()])
    tail = _clean((sentence or "")[pred.start() :], keep_conj_prefix=False)
    if "," not in head:
        return []

    normalized = re.sub(r"\s*,\s*and\s+", ", ", head, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+and\s+", ", ", normalized, flags=re.IGNORECASE)
    items = [_clean(x) for x in normalized.split(",") if _clean(x)]
    if len(items) < 2:
        return []

    first = items[0]
    q = re.search(r"\b(rich in|low in|high in|with|without|deficient in)\b", first, flags=re.IGNORECASE)
    subject_root = _clean(first[: q.start()]) if q else _clean(" ".join(first.split()[:2]))
    qualifier_prefix = (q.group(1).strip() + " ") if q else ""
    out: List[str] = []
    for idx, item in enumerate(items):
        phrase = item
        if (
            idx > 0
            and subject_root
            and re.match(
                r"^(rich in|low in|high in|with|without|deficient in)\b",
                item,
                flags=re.IGNORECASE,
            )
        ):
            phrase = f"{subject_root} {item}"
        elif idx > 0 and len(item.split()) <= 4:
            if qualifier_prefix and subject_root:
                phrase = f"{subject_root} {qualifier_prefix}{item}"
            elif subject_root:
                phrase = f"{subject_root} {item}"
            elif qualifier_prefix:
                phrase = qualifier_prefix + item

        out.append(_clean(f"{phrase} {tail}", keep_conj_prefix=False))
    return out


def _split_on_lists(text: str) -> List[str]:
    numbered = re.split(r"\s+(?:\d+\.|\(\d+\))\s+", text or "")
    if len(numbered) > 1:
        return [_clean(item, keep_conj_prefix=False) for item in numbered if _clean(item, keep_conj_prefix=False)]

    bulleted = _LIST_BULLETS_RE.split(text or "")
    if len(bulleted) > 1:
        return [_clean(item, keep_conj_prefix=False) for item in bulleted if _clean(item, keep_conj_prefix=False)]
    return []


def _split_on_comparatives(text: str) -> List[str]:
    m = re.search(
        r"\b(more|less|higher|lower|greater|fewer|better|worse)\b(.+?)\bthan\b(.+)$",
        text or "",
        flags=re.IGNORECASE,
    )
    if m:
        left = _clean((text or "")[: m.start()], keep_conj_prefix=False)
        comp_mid = _clean(m.group(1) + m.group(2), keep_conj_prefix=False)
        right = _clean(m.group(3), keep_conj_prefix=False)
        if left and comp_mid and right:
            return [left, f"{comp_mid} than {right}".strip()]

    comparatives = [
        " more than ",
        " less than ",
        " higher than ",
        " lower than ",
        " greater than ",
        " fewer than ",
        " better than ",
        " worse than ",
    ]
    for comp in comparatives:
        low = (text or "").lower()
        if comp in low:
            parts = (text or "").split(comp)
            if len(parts) == 2:
                return [
                    _clean(parts[0], keep_conj_prefix=False),
                    f"{comp.strip()} {_clean(parts[1], keep_conj_prefix=False)}",
                ]
    return []


def _base_verb(verb: str) -> str:
    v = (verb or "").lower().strip()
    if not v:
        return ""
    irregular = {"is": "be", "are": "be", "was": "be", "were": "be", "has": "have", "does": "do"}
    if v in irregular:
        return irregular[v]
    if len(v) > 4 and v.endswith("ies"):
        return v[:-3] + "y"
    if len(v) > 4 and v.endswith("es"):
        return v[:-2]
    if len(v) > 3 and v.endswith("s"):
        return v[:-1]
    return v


def _split_on_contrast_not(text: str) -> List[str]:
    """
    Split contrast clauses like "X, not Y" into two verifiable segments.
    """
    sentence = _clean(text, keep_conj_prefix=False)
    if not sentence:
        return []

    m = re.search(r"^(?P<left>.+?),\s*not\s+(?P<right>.+)$", sentence, flags=re.IGNORECASE)
    if not m:
        return []

    left = _clean(m.group("left"), keep_conj_prefix=False)
    right = _clean(m.group("right"), keep_conj_prefix=False)
    if not left or not right:
        return []

    if _looks_independent_clause(right):
        return [left, right]

    clause = re.match(r"^(?P<subject>.+?)\s+(?P<verb>\w+)\s+.+$", left)
    if clause and _CONTRAST_VERB_RE.search(clause.group("verb")):
        subject = _clean(clause.group("subject"), keep_conj_prefix=False)
        verb = _base_verb(clause.group("verb"))
        aux = "does not" if subject.lower() in {"he", "she", "it", "this", "that"} else "do not"
        rebuilt = _clean(f"{subject} {aux} {verb} {right}", keep_conj_prefix=False)
        if rebuilt:
            return [left, rebuilt]

    return [left, _clean(f"not {right}", keep_conj_prefix=False)]


def _split_on_conjunctives(text: str) -> List[str]:
    # Let semicolons drive top-level segmentation first; conjunction splitting across
    # semicolon boundaries can create malformed segments by borrowing the wrong subject.
    if ";" in (text or ""):
        return []

    def _split_top_level(raw: str, conj_token: str) -> List[str]:
        low = (raw or "").lower()
        parts: List[str] = []
        depth = 0
        last = 0
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch in "([{":
                depth += 1
            elif ch in ")]}" and depth > 0:
                depth -= 1
            if depth == 0 and low.startswith(conj_token, i):
                chunk = _clean(raw[last:i])
                if chunk:
                    parts.append(chunk)
                i += len(conj_token)
                last = i
                continue
            i += 1
        tail = _clean(raw[last:])
        if tail:
            parts.append(tail)
        return parts

    for conj in _CONJUNCTIVES:
        low_full = (text or "").lower()
        if conj not in low_full:
            continue

        parts = _split_top_level((text or ""), conj)
        if len(parts) <= 1:
            continue

        # Shared-predicate expansion for "X causes A or B" style claims.
        if conj.strip() in {"and", "or"} and len(parts) == 2:
            left = parts[0]
            right = _clean(parts[1], keep_conj_prefix=False)
            # Avoid bad splits for coordinated noun phrases under one predicate:
            # "needed for growth and development ..."
            if (
                conj.strip() == "and"
                and not _looks_independent_clause(right)
                and re.search(r"\b(needed for|required for|important for|growth)\b", left, flags=re.IGNORECASE)
                and re.search(r"\b(development|growth|maturation|bone)\b", right, flags=re.IGNORECASE)
            ):
                return [_clean(text, keep_conj_prefix=False)]
            # Handle "X ... to/for A and B" by inheriting the governing preposition.
            if right and not _looks_independent_clause(right):
                prep = re.match(
                    r"^(?P<prefix>.+?\b(?:to|for|against|in|with|into|from|on)\b)\s+(?P<obj>.+)$",
                    left,
                    flags=re.IGNORECASE,
                )
                if prep and len(right.split()) <= 8:
                    return [left, _clean(f"{prep.group('prefix')} {right}", keep_conj_prefix=False)]
            if right and not _looks_independent_clause(right):
                subject_pred = _extract_subject_predicate(left)
                if subject_pred:
                    return [left, _clean(f"{subject_pred} {right}", keep_conj_prefix=False)]

        result: List[str] = []
        for i, part in enumerate(parts):
            if i == 0:
                result.append(part)
            else:
                # Preserve conjunction prefix for compatibility with existing query merging.
                result.append(f"{conj.strip()} {_clean(part, keep_conj_prefix=False)}")
        return result
    return []


def _split_on_relative_clause(text: str) -> List[str]:
    """
    Split claims like:
      "X is Y that/which Z ..."
    into:
      - "X is Y"
      - "X Z ..."
    This improves partial-resolution behavior for general factual claims.
    """
    sentence = _clean(text, keep_conj_prefix=False)
    if not sentence:
        return []
    match = _RELATIVE_CLAUSE_RE.match(sentence)
    if not match:
        return []
    head = _clean(match.group("head"), keep_conj_prefix=False)
    tail = _clean(match.group("tail"), keep_conj_prefix=False)
    if not head or not tail:
        return []
    if not _looks_independent_clause(tail):
        return []

    subject = _extract_subject_head(head)
    if not subject:
        return []
    rebuilt_tail = _clean(f"{subject} {tail}", keep_conj_prefix=False)
    if not rebuilt_tail:
        return []
    return [head, rebuilt_tail]


def split_claim_into_segments(claim: str, min_segment_chars: int = 10) -> List[str]:
    """
    Shared deterministic claim segmentation for adaptive trust and verdict generation.
    """
    if not claim or not claim.strip():
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", claim.strip()) if s.strip()]
    raw_segments: List[str] = []
    for sentence in sentences:
        expanded = _expand_enumeration(sentence)
        if expanded:
            raw_segments.extend(expanded)
            continue

        by_lists = _split_on_lists(sentence)
        if by_lists:
            raw_segments.extend(by_lists)
            continue

        by_comparatives = _split_on_comparatives(sentence)
        if by_comparatives:
            raw_segments.extend(by_comparatives)
            continue

        by_contrast_not = _split_on_contrast_not(sentence)
        if by_contrast_not:
            raw_segments.extend(by_contrast_not)
            continue

        by_relative = _split_on_relative_clause(sentence)
        if by_relative:
            raw_segments.extend(by_relative)
            continue

        by_conjunctive = _split_on_conjunctives(sentence)
        if by_conjunctive:
            raw_segments.extend(by_conjunctive)
            continue

        by_semicolon = [_clean(p, keep_conj_prefix=False) for p in re.split(r"\s*;\s*", sentence) if _clean(p)]
        raw_segments.extend(by_semicolon or [_clean(sentence, keep_conj_prefix=False)])

    def _has_predicate_tokens(text: str) -> bool:
        t = (text or "").lower().strip()
        if not t:
            return False
        verb_like = re.search(
            r"\b("
            r"is|are|was|were|be|been|being|"
            r"has|have|had|do|does|did|can|could|may|might|must|should|would|will|"
            r"helps?|prevents?|reduces?|decreases?|lowers?|increases?|causes?|improves?|worsens?|protects?|"
            r"needed|required|important|"
            r"associated|linked|leads?|results?|"
            r"treats?|cures?|supports?|maintains?|builds?|promotes?|regulates?"
            r")\b",
            t,
            flags=re.IGNORECASE,
        )
        if verb_like:
            return True
        return bool(re.search(r"\b\w+(?:ed|ing)\b", t))

    # Merge noun-phrase fragments forward so each segment contains a predicate.
    normalized_segments: List[str] = []
    i = 0
    while i < len(raw_segments):
        current = _clean(raw_segments[i], keep_conj_prefix=False)
        if not current:
            i += 1
            continue
        if not _has_predicate_tokens(current) and i + 1 < len(raw_segments):
            merged = _clean(f"{current} {raw_segments[i + 1]}", keep_conj_prefix=False)
            if merged:
                normalized_segments.append(merged)
            i += 2
            continue
        normalized_segments.append(current)
        i += 1

    # If a segment starts with an auxiliary ("has been shown ..."), inherit
    # the subject head from the previous segment to keep it verifiable.
    for i in range(1, len(normalized_segments)):
        seg = _clean(normalized_segments[i], keep_conj_prefix=False)
        if not seg or not _AUX_LEADING_RE.match(seg):
            continue
        subject_head = _extract_subject_head(normalized_segments[i - 1])
        if not subject_head:
            continue
        normalized_segments[i] = _clean(f"{subject_head} {seg}", keep_conj_prefix=False)

    seen = set()
    out: List[str] = []
    for seg in normalized_segments:
        s = _clean(seg)
        key = s.lower().strip()
        if len(key) < min_segment_chars:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(s)

    if not out:
        cleaned = _clean(claim, keep_conj_prefix=False)
        return [cleaned] if cleaned else []
    return out


__all__ = ["split_claim_into_segments"]
