#!/usr/bin/env python3
"""
Guideline RAG evaluation — retrieval AND generation.

Retrieval metrics (keyword-anchored labels from JSONL):
  Hit@k, Precision@k, MRR, mean similarity score

Generation metrics (LLM answer grounded on retrieved passages):
  Answer relevance, Groundedness (lexical faithfulness), Citation rate

Usage:
  cd backend && source ../env/bin/activate
  python ../evals/run_rag_eval.py
  python ../evals/run_rag_eval.py --skip-generation   # retrieval only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

# Load .env if present
for env_path in (BACKEND / ".env", ROOT / ".env"):
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass
        break

from evals.metrics import (  # noqa: E402
    answer_relevance,
    citation_rate,
    groundedness,
    keyword_hit,
    keyword_mrr,
    keyword_precision_at_k,
    mean,
)
from evals.qdrant_lock import ensure_qdrant_for_eval  # noqa: E402

ensure_qdrant_for_eval()
from app.services.rag_retrieval import retrieve_multi_query  # noqa: E402

DATA = Path(__file__).resolve().parent / "datasets" / "guideline_rag.jsonl"
K = 5

GEN_SYSTEM = """You are a fitness/nutrition assistant for India.
Answer ONLY using the guideline passages below. Cite source names.
If passages are insufficient, say so briefly. Keep answers under 180 words."""


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_context(chunks: list) -> str:
    lines = []
    for c in chunks[:K]:
        lines.append(
            f"- ({c.get('source_name')}, p.{c.get('page_number')}): {c.get('text', '')[:400]}"
        )
    return "\n".join(lines) if lines else "(no passages retrieved)"


def generate_answer(query: str, context: str) -> str | None:
    """Call configured LLM; return None if unavailable."""
    try:
        from app.llm import get_llm
        llm = get_llm(temperature=0.1, max_tokens=350, with_azure_fallback=True)
        prompt = (
            f"{GEN_SYSTEM}\n\n"
            f"GUIDELINE PASSAGES:\n{context}\n\n"
            f"USER QUESTION: {query}\n\nANSWER:"
        )
        msg = llm.invoke(prompt)
        return (getattr(msg, "content", None) or str(msg)).strip()
    except Exception as e:
        print(f"  [WARN] generation skipped: {type(e).__name__}: {e}")
        return None


def evaluate_retrieval(cases: list) -> dict:
    hit_scores, prec_scores, mrr_scores, mean_scores = [], [], [], []

    print("=" * 64)
    print(f"  RETRIEVAL EVAL  —  {len(cases)} queries, k={K}")
    print("=" * 64)

    for case in cases:
        filters = {}
        if case.get("content_types"):
            filters["content_type__in"] = case["content_types"]
            filters["trust_tier__in"] = ["Tier 1", "Tier 2"]
        chunks = retrieve_multi_query([case["query"]], filters, top_k_per_query=K)
        texts = [c.get("text", "") for c in chunks]
        keys = case.get("must_contain_any") or []

        h = keyword_hit(texts, keys)
        p = keyword_precision_at_k(texts, keys, K)
        r = keyword_mrr(texts, keys)
        ms = mean([float(c.get("score") or 0.0) for c in chunks]) if chunks else 0.0

        hit_scores.append(h)
        prec_scores.append(p)
        mrr_scores.append(r)
        mean_scores.append(ms)

        status = "PASS" if h else "FAIL"
        top = chunks[0] if chunks else None
        top_desc = (
            f"{top.get('source_name')} p.{top.get('page_number')} score={top.get('score')}"
            if top else "<empty>"
        )
        print(f"[{status}] {case['id']}: Hit={h:.0f} P@{K}={p:.2f} MRR={r:.2f} | {case['query'][:50]}")
        print(f"         top: {top_desc}")

    summary = {
        f"Hit@{K}": mean(hit_scores),
        f"Precision@{K}": mean(prec_scores),
        "MRR": mean(mrr_scores),
        "Mean_score": mean(mean_scores),
        "n": len(cases),
        "passes": int(sum(hit_scores)),
    }

    print("\n--- Retrieval metrics ---")
    print(f"  Hit@{K}:         {summary[f'Hit@{K}']:.3f}  ({summary['passes']}/{summary['n']})")
    print(f"  Precision@{K}:  {summary[f'Precision@{K}']:.3f}")
    print(f"  MRR:           {summary['MRR']:.3f}")
    print(f"  Mean score:    {summary['Mean_score']:.3f}")
    print("  (labels = must_contain_any keywords in JSONL; swap for gold chunk_ids for true IR)\n")
    return summary


def evaluate_generation(cases: list) -> dict | None:
    rel_scores, ground_scores, cite_scores = [], [], []
    n_gen = 0

    print("=" * 64)
    print(f"  GENERATION EVAL  —  {len(cases)} queries (LLM + groundedness)")
    print("=" * 64)

    for case in cases:
        filters = {}
        if case.get("content_types"):
            filters["content_type__in"] = case["content_types"]
            filters["trust_tier__in"] = ["Tier 1", "Tier 2"]
        chunks = retrieve_multi_query([case["query"]], filters, top_k_per_query=K)
        texts = [c.get("text", "") for c in chunks]
        sources = [c.get("source_name", "") for c in chunks]
        context = build_context(chunks)

        answer = generate_answer(case["query"], context)
        if answer is None:
            print(f"[SKIP] {case['id']}: LLM unavailable")
            continue

        n_gen += 1
        keys = case.get("must_answer_any") or case.get("must_contain_any") or []
        rel = answer_relevance(answer, keys)
        grd = groundedness(answer, texts)
        cite = citation_rate(answer, sources)

        rel_scores.append(rel)
        ground_scores.append(grd)
        cite_scores.append(cite)

        status = "PASS" if rel >= 1.0 and grd >= 0.25 else "FAIL"
        print(f"[{status}] {case['id']}: relevance={rel:.0f} groundedness={grd:.2f} citation={cite:.0f}")
        print(f"         Q: {case['query'][:55]}")
        preview = answer.replace("\n", " ")[:120]
        print(f"         A: {preview}{'…' if len(answer) > 120 else ''}")

    if n_gen == 0:
        print("\n--- Generation metrics ---")
        print("  SKIPPED — set GROQ_API_KEY or Azure OpenAI env vars to enable.\n")
        return None

    summary = {
        "Answer_relevance": mean(rel_scores),
        "Groundedness": mean(ground_scores),
        "Citation_rate": mean(cite_scores),
        "n": n_gen,
    }

    print("\n--- Generation metrics ---")
    print(f"  Answer relevance:  {summary['Answer_relevance']:.3f}  ({sum(rel_scores):.0f}/{n_gen})")
    print(f"  Groundedness:      {summary['Groundedness']:.3f}  (lexical overlap with retrieved context)")
    print(f"  Citation rate:     {summary['Citation_rate']:.3f}  (mentions a retrieved source name)")
    print()
    return summary


def main():
    parser = argparse.ArgumentParser(description="Guideline RAG retrieval + generation eval")
    parser.add_argument("--skip-generation", action="store_true", help="Only run retrieval metrics")
    args = parser.parse_args()

    cases = load_jsonl(DATA)
    print(f"\nGuideline RAG eval dataset: {DATA.name} ({len(cases)} cases)\n")

    ret = evaluate_retrieval(cases)
    gen = None if args.skip_generation else evaluate_generation(cases)

    print("=" * 64)
    print("  FINAL SUMMARY")
    print("=" * 64)
    print("RETRIEVAL")
    print(f"  Hit@{K}={ret[f'Hit@{K}']:.3f}  Precision@{K}={ret[f'Precision@{K}']:.3f}  "
          f"MRR={ret['MRR']:.3f}  Mean_score={ret['Mean_score']:.3f}")
    if gen:
        print("GENERATION")
        print(f"  Relevance={gen['Answer_relevance']:.3f}  Groundedness={gen['Groundedness']:.3f}  "
              f"Citation={gen['Citation_rate']:.3f}")
    elif not args.skip_generation:
        print("GENERATION: skipped (no LLM)")
    print()
    # Fail CI / local runs when retrieval is weak
    fail = ret.get(f"Hit@{K}", 0) < 0.5 or ret.get("MRR", 0) < 0.4
    if gen and gen.get("Answer_relevance", 1) < 0.5:
        fail = True
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
