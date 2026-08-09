#!/usr/bin/env python3
"""
Behind-the-scenes RAG tracer — prints real processing for one query.

Usage:
  cd backend && source ../env/bin/activate   # if you use that venv
  python ../evals/trace_rag_query.py
  python ../evals/trace_rag_query.py "how to do tadasana"
  python ../evals/trace_rag_query.py "protein for vegetarians India" --no-images

Shows: embedding models, Qdrant collection shape, query vector sample,
retrieved text chunks, CLIP image matches, and the tool briefing the
ReAct LLM would see (without calling the chat LLM).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

for env_path in (BACKEND / ".env", ROOT / ".env"):
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass
        break


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _preview(text: str, n: int = 420) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t if len(t) <= n else t[:n] + "…"


def show_vector_db(client) -> None:
    from app.services.rag_retrieval import COLLECTION_NAME, IMAGE_COLLECTION_NAME

    _hr("1) VECTOR DB (Qdrant) — collections")
    cols = [c.name for c in client.get_collections().collections]
    print(f"Path (QDRANT_PATH): {client._client.__dict__.get('location', '(see env)')}")
    print(f"Collections present: {cols}")

    for name in (COLLECTION_NAME, IMAGE_COLLECTION_NAME):
        if name not in cols:
            print(f"\n[{name}] MISSING — run embed scripts first")
            continue
        info = client.get_collection(name)
        cfg = info.config.params.vectors
        # Named vs unnamed vectors
        if hasattr(cfg, "size"):
            dim, dist = cfg.size, cfg.distance
        else:
            # dict of named vectors
            first = next(iter(cfg.values())) if cfg else None
            dim = getattr(first, "size", "?")
            dist = getattr(first, "distance", "?")
        print(f"\n[{name}]")
        print(f"  points_count : {info.points_count}")
        print(f"  vector_size  : {dim}")
        print(f"  distance     : {dist}")

        pts, _ = client.scroll(collection_name=name, limit=1, with_payload=True, with_vectors=True)
        if not pts:
            print("  (empty)")
            continue
        p = pts[0]
        payload = p.payload or {}
        print(f"  sample point id: {p.id}")
        print(f"  payload keys   : {sorted(payload.keys())}")
        # Show a few payload fields
        for k in ("chunk_id", "source_id", "source_name", "page_number", "section_title",
                  "content_type", "trust_tier", "image_url", "caption"):
            if k in payload:
                val = payload[k]
                if isinstance(val, str) and len(val) > 120:
                    val = val[:120] + "…"
                print(f"    {k}: {val}")
        vec = p.vector
        if isinstance(vec, dict):
            vec = next(iter(vec.values()))
        if vec is not None:
            print(f"  vector dim     : {len(vec)}")
            print(f"  vector sample  : [{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}, …, {vec[-1]:.4f}]")
            print("  (this is what 'an embedding in the DB' looks like — a list of floats)")


def show_embeddings(query: str) -> None:
    from app.services.embedder import (
        encode_one, encode_text_clip, get_shared_embed_model, get_shared_clip_model,
        CLIP_MODEL_NAME, CLIP_DIM,
    )
    from app.services.rag_retrieval import _clip_query_variants

    _hr("2) EMBEDDING MODELS + QUERY VECTOR")
    m = get_shared_embed_model()
    print(f"Text model : all-MiniLM-L6-v2  (384-d)  loaded={m is not None}")
    print(f"Image model: {CLIP_MODEL_NAME}  ({CLIP_DIM}-d)  loaded={get_shared_clip_model() is not None}")
    print("Rule: MiniLM ↔ text chunks; CLIP text tower ↔ image vectors. Never mix spaces.")

    variants = _clip_query_variants(query)
    print(f"\nQuery: {query!r}")
    print(f"CLIP/text variants (yoga expansions):")
    for i, v in enumerate(variants):
        print(f"  [{i}] {v}")

    v = encode_one(query)
    print(f"\nMiniLM query vector: dim={len(v)}")
    print(f"  first 8: {[round(x, 4) for x in v[:8]]}")
    print(f"  L2 norm ≈ {sum(x*x for x in v)**0.5:.4f} (normalized → ~1.0)")

    try:
        cv = encode_text_clip(query)
        print(f"\nCLIP text query vector: dim={len(cv)}")
        print(f"  first 8: {[round(x, 4) for x in cv[:8]]}")
    except Exception as e:
        print(f"\nCLIP encode failed: {e}")


def show_text_retrieval(query: str) -> list:
    from app.services.rag_retrieval import (
        retrieve_multi_query, rerank_guideline_chunks, _clip_query_variants,
    )

    _hr("3) TEXT RAG — retrieve chunks from fitness_guidelines")
    rag_queries = [query]
    for variant in _clip_query_variants(query)[1:]:
        if variant not in rag_queries:
            rag_queries.append(variant)

    print("Queries sent to MiniLM → Qdrant:")
    for q in rag_queries:
        print(f"  • {q}")

    filters = {"trust_tier__in": ["Tier 1", "Tier 2"]}
    print(f"\nFilters: {filters}")
    print("Process: embed each query → cosine top-k → dedupe by chunk_id → sort by score → rerank")

    chunks = retrieve_multi_query(rag_queries, filters, top_k_per_query=4)
    print(f"\nRaw unique chunks: {len(chunks)}")
    chunks = rerank_guideline_chunks(query, chunks)
    print(f"After rerank: {len(chunks)}")

    for i, c in enumerate(chunks[:6], 1):
        print(f"\n--- chunk #{i} ---")
        print(f"  score      : {c.get('score')}")
        print(f"  source     : {c.get('source_name')} ({c.get('source_id')})")
        print(f"  page       : {c.get('page_number')}")
        print(f"  section    : {c.get('section_title')}")
        print(f"  trust/type : {c.get('trust_tier')} / {c.get('content_type')}")
        print(f"  chunk_id   : {c.get('chunk_id')}")
        print(f"  image_urls : {c.get('image_urls')}")
        print(f"  text       : {_preview(c.get('text', ''), 500)}")

    return chunks


def show_image_retrieval(query: str) -> list:
    from app.services.rag_retrieval import (
        retrieve_guideline_images, images_from_matching_chunks,
        IMAGE_SCORE_THRESHOLD, DEMO_IMAGE_SOURCE_IDS,
    )

    _hr("4) IMAGE RAG — yoga/protocol demos (CLIP hybrid)")
    print(f"Demo sources allowed: {DEMO_IMAGE_SOURCE_IDS}")
    print(f"Production IMAGE_SCORE_THRESHOLD: {IMAGE_SCORE_THRESHOLD}")
    print("Hybrid steps:")
    print("  1) MiniLM text anchors → likely PDF pages")
    print("  2) CLIP text query vs image vectors in guideline_images")
    print("  3) Caption MiniLM + lexical boost re-rank")
    print("  4) Keep only confident scores")

    # Same order as answer_fitness_question (needs chunks for same-page path)
    from app.services.rag_retrieval import retrieve_multi_query, rerank_guideline_chunks, _clip_query_variants
    rag_queries = [query] + _clip_query_variants(query)[1:]
    chunks = rerank_guideline_chunks(
        query,
        retrieve_multi_query(rag_queries, {"trust_tier__in": ["Tier 1", "Tier 2"]}, top_k_per_query=4),
    )

    same_page = images_from_matching_chunks(query, chunks, top_k=2)
    print(f"\n(A) Same-page images from text hits: {len(same_page)}")
    for img in same_page:
        print(f"  • {img.get('source_id')} p.{img.get('page_number')} score={img.get('score')} "
              f"url={img.get('image_url')}")
        print(f"    caption: {_preview(img.get('caption', ''), 160)}")

    # threshold 0 so we can SEE ranking even below production floor
    hybrid = retrieve_guideline_images(
        query, top_k=4, source_ids=list(DEMO_IMAGE_SOURCE_IDS), score_threshold=0.0,
    )
    print(f"\n(B) CLIP hybrid top-4 (threshold=0 for visibility): {len(hybrid)}")
    for img in hybrid:
        flag = "✓ would show" if float(img.get("score") or 0) >= IMAGE_SCORE_THRESHOLD else "· below prod threshold"
        print(f"  [{flag}] score={img.get('score')} clip={img.get('clip_score')} "
              f"{img.get('source_id')} p.{img.get('page_number')}")
        print(f"       url={img.get('image_url')}")
        print(f"       caption: {_preview(img.get('caption', ''), 160)}")

    chosen = same_page or [
        i for i in hybrid if float(i.get("score") or 0) >= IMAGE_SCORE_THRESHOLD
    ][:2]
    print(f"\n→ Images the tool would keep (same-page first, else hybrid ≥ threshold): {len(chosen)}")
    return chosen


def show_tool_briefing(query: str, chunks: list, images: list) -> None:
    _hr("5) WHAT THE ReAct LLM SEES NEXT (tool briefing, not final UI text)")
    print("answer_fitness_question does NOT call the chat LLM.")
    print("It returns a briefing string; LangGraph LLM call #2 writes the user answer.\n")

    parts = [
        "TURN_INTENT=exercise_qa  MEDIA=yoga_protocol",
        "GUIDELINE PASSAGES:",
    ]
    for c in chunks[:5]:
        body = (c.get("text") or "")[:400]
        parts.append(f"- ({c.get('source_name')}, p.{c.get('page_number')}): {body}")
    if images:
        parts.append("MATCHED DEMONSTRATION PHOTO(S) (UI will render these):")
        for img in images:
            parts.append(
                f"- {img.get('source_name')} p.{img.get('page_number')} "
                f"(score {img.get('score')}): {(img.get('caption') or '')[:120]}"
            )
    briefing = "\n".join(parts)
    print(briefing[:3500])
    if len(briefing) > 3500:
        print("\n… [truncated]")

    _hr("6) AFTER THIS (not executed here)")
    print("LLM call #2: narrate steps from passages, cite pages, tie to photos.")
    print("process_user_message: attach guideline_images to JSON response.")
    print("React ChatPage: render text bubble + GuidelineImagesStrip.")


def main():
    parser = argparse.ArgumentParser(description="Trace RAG processing for one query")
    parser.add_argument("query", nargs="?", default="how to do pranayama")
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args()
    query = args.query.strip()

    from evals.qdrant_lock import ensure_qdrant_for_eval
    path = ensure_qdrant_for_eval()
    print(f"Using Qdrant at: {path}")

    from app.services.rag_retrieval import get_qdrant, reset_qdrant_client
    reset_qdrant_client()
    client = get_qdrant()

    show_vector_db(client)
    show_embeddings(query)
    chunks = show_text_retrieval(query)
    images = [] if args.no_images else show_image_retrieval(query)
    show_tool_briefing(query, chunks, images)

    _hr("DONE")
    print("Re-run with another query:")
    print('  python ../evals/trace_rag_query.py "how to do tadasana"')
    print('  python ../evals/trace_rag_query.py "water intake daily" --no-images')


if __name__ == "__main__":
    main()
