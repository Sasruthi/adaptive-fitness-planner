"""Shared IR / generation metrics for Adaptive Fitness Planner evals."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Set


_WORD = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> Set[str]:
    return {t.lower() for t in _WORD.findall(text or "") if len(t) > 2}


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def hit_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = [r.lower() for r in retrieved[:k]]
    gold = {g.lower() for g in relevant}
    return 1.0 if any(r in gold for r in top) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = {r.lower() for r in retrieved[:k]}
    gold = {g.lower() for g in relevant}
    return len(top & gold) / len(gold)


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    gold = {g.lower() for g in relevant}
    hits = sum(1 for r in top if r.lower() in gold)
    return hits / len(top)


def mrr(retrieved: Sequence[str], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    gold = {g.lower() for g in relevant}
    for i, r in enumerate(retrieved, start=1):
        if r.lower() in gold:
            return 1.0 / i
    return 0.0


def keyword_hit(texts: Sequence[str], must_contain_any: List[str]) -> float:
    """1 if any retrieved text contains any required keyword (case-insensitive)."""
    if not must_contain_any:
        return 0.0  # empty labels must not auto-pass
    blob = " ".join(texts).lower()
    return 1.0 if any(k.lower() in blob for k in must_contain_any) else 0.0


def keyword_precision_at_k(texts: Sequence[str], must_contain_any: List[str], k: int) -> float:
    """Fraction of top-k passages that contain at least one required keyword."""
    if not must_contain_any:
        return 0.0  # empty labels must not auto-pass
    top = list(texts)[:k]
    if not top:
        return 0.0
    keys = [k.lower() for k in must_contain_any]
    hits = sum(1 for t in top if any(k in (t or "").lower() for k in keys))
    return hits / len(top)


def keyword_mrr(texts: Sequence[str], must_contain_any: List[str]) -> float:
    """1/rank of the first passage containing any required keyword."""
    if not must_contain_any:
        return 0.0  # empty labels must not auto-pass
    keys = [k.lower() for k in must_contain_any]
    for i, t in enumerate(texts, start=1):
        low = (t or "").lower()
        if any(k in low for k in keys):
            return 1.0 / i
    return 0.0


def answer_relevance(answer: str, must_contain_any: List[str]) -> float:
    """1 if the generated answer mentions any expected keyword/concept."""
    if not must_contain_any:
        return 0.0
    low = (answer or "").lower()
    return 1.0 if any(k.lower() in low for k in must_contain_any) else 0.0


def groundedness(answer: str, contexts: Sequence[str]) -> float:
    """
    Lexical faithfulness proxy: share of answer content words that also
    appear in the retrieved context (0–1). Empty answer → 0.
    """
    ans_toks = tokenize(answer)
    if not ans_toks:
        return 0.0
    ctx_toks = set()
    for c in contexts:
        ctx_toks |= tokenize(c)
    if not ctx_toks:
        return 0.0
    return len(ans_toks & ctx_toks) / len(ans_toks)


def citation_rate(answer: str, source_names: Sequence[str]) -> float:
    """1 if the answer mentions at least one retrieved source name."""
    names = [s for s in source_names if s and len(s.strip()) > 2]
    if not names:
        return 0.0
    low = (answer or "").lower()
    return 1.0 if any(s.lower() in low for s in names) else 0.0


def forbidden_hit(answer: str, forbidden_any: List[str]) -> float:
    """1 if answer contains a forbidden string (hallucination / wrong-mode)."""
    if not forbidden_any:
        return 0.0
    low = (answer or "").lower()
    return 1.0 if any(f.lower() in low for f in forbidden_any) else 0.0
