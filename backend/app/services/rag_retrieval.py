"""
RAG Retrieval Service
=====================
Semantic search over India-first guideline chunks stored in Qdrant.
Returns cited passages that ground GPT-4o's plan recommendations.

Design decisions:
- Embedding model: all-MiniLM-L6-v2 (free, same model used during ingestion)
- Filters: trust_tier, content_type, category applied as Qdrant payload filters
- Returns top-k chunks with full citation metadata
- Falls back to unfiltered search if filtered returns < min_results
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

_DEFAULT_QDRANT = Path(__file__).resolve().parents[3] / "rag" / "qdrant_local"
QDRANT_PATH     = Path(os.getenv("QDRANT_PATH", str(_DEFAULT_QDRANT)))
COLLECTION_NAME = "fitness_guidelines"
EMBED_DIM       = 384

# ── Embedding model (loads once, cached) ─────────────────────────────────────
def get_embed_model():
    from app.services.embedder import get_shared_embed_model
    return get_shared_embed_model()

def embed_query(text: str) -> List[float]:
    from app.services.embedder import encode_one, get_shared_embed_model
    if get_shared_embed_model() is None:
        raise RuntimeError(
            "Embedding model unavailable — cannot run semantic RAG. "
            "Install sentence-transformers / check network for model download."
        )
    return encode_one(text)


# ── Qdrant client (lazy init) ─────────────────────────────────────────────────
_qdrant = None

def get_qdrant() -> QdrantClient:
    global _qdrant, QDRANT_PATH
    if _qdrant is None:
        QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(_DEFAULT_QDRANT)))
        _qdrant = QdrantClient(path=str(QDRANT_PATH))
    return _qdrant


def reset_qdrant_client() -> None:
    """Drop cached client so the next get_qdrant() re-reads QDRANT_PATH."""
    global _qdrant
    if _qdrant is not None:
        try:
            _qdrant.close()
        except Exception:
            pass
    _qdrant = None


# ── Build Qdrant payload filter from rag_filters dict ────────────────────────
def build_qdrant_filter(rag_filters: Dict) -> Optional[Filter]:
    conditions = []

    if rag_filters.get("trust_tier__in"):
        conditions.append(FieldCondition(
            key="trust_tier",
            match=MatchAny(any=rag_filters["trust_tier__in"]),
        ))

    if rag_filters.get("content_type__in"):
        conditions.append(FieldCondition(
            key="content_type",
            match=MatchAny(any=rag_filters["content_type__in"]),
        ))

    if rag_filters.get("category__in"):
        conditions.append(FieldCondition(
            key="category",
            match=MatchAny(any=rag_filters["category__in"]),
        ))

    if not conditions:
        return None

    return Filter(must=conditions)


# ── Main retrieval function ───────────────────────────────────────────────────
def retrieve_guidelines(
    query: str,
    rag_filters: Dict,
    top_k: int = 5,
    min_results: int = 2,
) -> List[Dict]:
    """
    Semantic search over guideline chunks.

    Returns list of dicts:
      text, source_id, source_name, trust_tier,
      content_type, page_number, section_title, score
    """
    client   = get_qdrant()
    q_vector = embed_query(query)
    qfilter  = build_qdrant_filter(rag_filters)

    try:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vector,
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
        )
        chunks = results.points
    except Exception as e:
        print(f"[RAG] Qdrant error: {e}")
        chunks = []

    # Soft widen: if filtered result is thin, keep filtered hits and only
    # top-up with unfiltered (never replace Tier filters entirely).
    if len(chunks) < min_results:
        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=q_vector,
                limit=top_k,
                with_payload=True,
            )
            seen = {getattr(c, "id", None) for c in chunks}
            for c in results.points:
                cid = getattr(c, "id", None)
                if cid in seen:
                    continue
                # Prefer still respecting trust_tier when present on payload
                tier = (c.payload or {}).get("trust_tier", "")
                if rag_filters.get("trust_tier__in") and tier not in rag_filters["trust_tier__in"]:
                    continue
                chunks.append(c)
                if len(chunks) >= top_k:
                    break
        except Exception:
            pass

    return [_format_chunk(c) for c in chunks]


def retrieve_multi_query(
    queries: List[str],
    rag_filters: Dict,
    top_k_per_query: int = 3,
) -> List[Dict]:
    """
    Run multiple semantic queries and deduplicate results.
    Used for plan generation where we need guidelines for
    exercise + nutrition + safety all at once.
    """
    seen_ids = set()
    all_chunks = []

    for query in queries:
        chunks = retrieve_guidelines(query, rag_filters, top_k=top_k_per_query)
        for chunk in chunks:
            cid = chunk["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                all_chunks.append(chunk)

    # Sort by score descending
    return sorted(all_chunks, key=lambda x: x["score"], reverse=True)


def _format_chunk(point) -> Dict:
    p = point.payload
    return {
        "chunk_id":     p.get("chunk_id", ""),
        "text":         p.get("text", ""),
        "source_id":    p.get("source_id", ""),
        "source_name":  p.get("source_name", ""),
        "trust_tier":   p.get("trust_tier", ""),
        "content_type": p.get("content_type", ""),
        "category":     p.get("category", ""),
        "page_number":  p.get("page_number"),
        "section_title":p.get("section_title"),
        "score":        round(point.score, 4),
    }
