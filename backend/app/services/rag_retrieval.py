"""
RAG retrieval over local Qdrant collections.

Collections:
  fitness_guidelines  — MiniLM text chunks (WHO/ICMR/Fit India/CYP PDFs)
  guideline_images    — CLIP image vectors from extract_pdf_images.py

Text path: encode query with all-MiniLM-L6-v2 → cosine top-k + payload filters.
Image path: encode query with CLIP text tower → cosine vs image embeddings,
then optional caption MiniLM re-rank + lexical boosts. Demos are limited to
DEMO_IMAGE_SOURCE_IDS (yoga/Fit India booklets), never food-guideline art.

Env: QDRANT_PATH (default: backend/rag/qdrant_local).
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue

_DEFAULT_QDRANT = Path(__file__).resolve().parents[2] / "rag" / "qdrant_local"
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", str(_DEFAULT_QDRANT)))
COLLECTION_NAME = "fitness_guidelines"
IMAGE_COLLECTION_NAME = "guideline_images"
EMBED_DIM = 384

# Final score floor for showing an image (CLIP + caption/lexical blend).
# Tune with evals/run_multimodal_rag_eval.py.
IMAGE_SCORE_THRESHOLD = 0.30

# Blend MiniLM(query, caption) into CLIP score so correctly captioned pages win.
CAPTION_RERANK_WEIGHT = 0.18
# Longer captions (technique paragraphs) improve MiniLM re-rank recall.
CAPTION_RERANK_CHARS = 700

# Extra bump when distinctive pose/technique tokens co-occur in the caption.
# Generic tokens (pranayam) stay light so Bhrāmarī/DHYĀNA pages don't outrank
# alternate-nostril demos for a nostril-breathing ask.
_LEXICAL_CAPTION_TERMS = (
    ("nostril", 0.18),
    ("anulom", 0.16),
    ("vilom", 0.14),
    ("nadi", 0.14),
    ("shodhan", 0.14),
    ("mountain", 0.16),
    ("tadasana", 0.16),
    ("tada", 0.12),
    ("tree", 0.14),
    ("vrksa", 0.14),
    ("dhyana", 0.16),
    ("meditat", 0.16),
    ("sambhavi", 0.14),
    ("butterfly", 0.14),
    ("titali", 0.14),
    ("camel", 0.12),
    ("ustra", 0.12),
    ("naukasana", 0.18),
    ("navasana", 0.16),
    ("boat", 0.14),
    ("pranayam", 0.04),
)

DEMO_IMAGE_SOURCE_IDS = ("SRC009", "SRC003")
YOG_PROTOCOL_SOURCE_ID = "SRC009"


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

    if rag_filters.get("source_id__in"):
        conditions.append(FieldCondition(
            key="source_id",
            match=MatchAny(any=rag_filters["source_id__in"]),
        ))

    if not conditions:
        return None

    return Filter(must=conditions)


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
    client = get_qdrant()
    q_vector = embed_query(query)
    qfilter = build_qdrant_filter(rag_filters)

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

    return sorted(all_chunks, key=lambda x: x["score"], reverse=True)


def _format_chunk(point) -> Dict:
    p = point.payload
    return {
        "chunk_id": p.get("chunk_id", ""),
        "text": p.get("text", ""),
        "source_id": p.get("source_id", ""),
        "source_name": p.get("source_name", ""),
        "trust_tier": p.get("trust_tier", ""),
        "content_type": p.get("content_type", ""),
        "category": p.get("category", ""),
        "page_number": p.get("page_number"),
        "section_title": p.get("section_title"),
        "score": round(point.score, 4),
        "image_urls": p.get("image_urls") or [],
    }


_PROTOCOL_AGE_BANDS = (
    (18, 35, "18-35"),
    (35, 50, "35-50"),
    (50, 65, "50-65"),
)


def match_protocol_age_band(query: str) -> Optional[str]:
    """
    Map a user age mention (e.g. 50-60) onto Fit India protocol bands
    (18-35 / 35-50 / 50-65). Returns the band label or None.
    """
    import re as _re

    ages = [
        int(x)
        for x in _re.findall(r"\b(\d{2})\b", query or "")
        if 10 <= int(x) <= 100
    ]
    if not ages:
        return None
    lo, hi = (min(ages), max(ages)) if len(ages) > 1 else (ages[0], ages[0])
    mid = (lo + hi) / 2.0
    for a, b, label in _PROTOCOL_AGE_BANDS:
        if a <= mid <= b:
            return label
    for a, b, label in _PROTOCOL_AGE_BANDS:
        if lo <= b and hi >= a:
            return label
    return None


def rerank_guideline_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
    """
    Post-retrieve re-rank: boost age-band protocol tables; demote tiny
    YouTube/link-only pages that otherwise dominate 'yoga protocol' queries.
    """
    if not chunks:
        return chunks
    band = match_protocol_age_band(query)
    ranked = []
    for c in chunks:
        text = (c.get("text") or "").replace("–", "-")
        score = float(c.get("score") or 0.0)
        boost = 0.0
        if band and f"Yoga Protocol for {band}" in text:
            boost += 0.2
        elif band and band in text:
            boost += 0.12
        low = text.lower()
        if len(text) < 400 and ("youtube.com" in low or "playlist" in low):
            boost -= 0.3
        ranked.append((score + boost, c))
    ranked.sort(key=lambda x: x[0], reverse=True)
    out = []
    for adj, c in ranked:
        cc = dict(c)
        cc["score"] = round(adj, 4)
        out.append(cc)
    return out


def retrieve_guidelines_by_pages(
    source_id: str,
    page_numbers: List[int],
    limit: int = 8,
) -> List[Dict]:
    """
    Fetch guideline text chunks for exact (source_id, page) pairs.
    Used to pull Technique/Benefits text for the same pages CLIP just matched.
    """
    if not source_id or not page_numbers:
        return []
    pages = sorted({int(p) for p in page_numbers if p is not None})
    if not pages:
        return []
    client = get_qdrant()
    try:
        qfilter = Filter(must=[
            FieldCondition(key="source_id", match=MatchValue(value=source_id)),
            FieldCondition(key="page_number", match=MatchAny(any=pages)),
        ])
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qfilter,
            limit=limit,
            with_payload=True,
        )
        return [_format_chunk(p) for p in points]
    except Exception as e:
        print(f"[RAG] page-lookup failed: {e}")
        return []


_IMAGE_COLLECTION_MISSING_WARNED = False

# Expand Sanskrit / short names into plain-English demo phrasing for CLIP text.
_YOGA_CLIP_EXPAND = (
    (re.compile(r"nadi|anulom|vilom|shodhan", re.I),
     "alternate nostril breathing pranayama yoga demonstration photo"),
    (re.compile(r"pranayam|prāṇāyām", re.I),
     "pranayama breathing yoga demonstration photo"),
    (re.compile(r"tadasana|tāḍāsana|mountain pose", re.I),
     "Tada means palm tree or mountain standing asana arms raised interlocking fingers"),
    (re.compile(r"vrksasana|vṛkṣāsana|tree pose", re.I),
     "tree pose balance foot on thigh arms up join palms yoga"),
    (re.compile(r"trikonasana|triangle pose", re.I),
     "triangle pose side bend arm up yoga demonstration"),
    (re.compile(r"dhyana|dhyāna|sambhavi|śambhavī|meditation", re.I),
     "seated meditation dhyana yoga posture demonstration photo"),
    (re.compile(r"butterfly|titali|titli", re.I),
     "butterfly pose seated yoga hip opening demonstration photo"),
    (re.compile(r"kapalabhati|kapālabhāti", re.I),
     "kapalabhati forceful exhalation pranayama yoga"),
    (re.compile(r"naukasana|navasana|naukāsana|boat pose", re.I),
     "naukasana boat pose supine raise legs and trunk core strength"),
)


def _clip_query_variants(query: str) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []
    variants = [q]
    for pat, expansion in _YOGA_CLIP_EXPAND:
        if pat.search(q) and expansion not in variants:
            variants.append(expansion)
    return variants


def _fold_yoga_text(s: str) -> str:
    """Lowercase + strip common Sanskrit diacritics for token overlap."""
    out = (s or "").lower()
    for src, dst in (
        ("ā", "a"), ("ī", "i"), ("ū", "u"), ("ṛ", "r"),
        ("ṅ", "n"), ("ñ", "n"), ("ṭ", "t"), ("ḍ", "d"),
        ("ṇ", "n"), ("ś", "s"), ("ṣ", "s"), ("ṃ", "m"), ("ḥ", "h"),
    ):
        out = out.replace(src, dst)
    return out


def _lexical_caption_boost(query: str, caption: str) -> float:
    """Additive boost when distinctive pose/technique tokens co-occur in caption."""
    q = _fold_yoga_text(query)
    c = _fold_yoga_text(caption)
    if not q or not c:
        return 0.0
    boost = 0.0
    for term, weight in _LEXICAL_CAPTION_TERMS:
        t = _fold_yoga_text(term)
        if t and t in q and t in c:
            boost += weight

    want_tada = ("tadasana" in q) or ("mountain pose" in q)
    want_tree = ("vrksasana" in q) or ("tree pose" in q)
    want_nostril = any(t in q for t in ("anulom", "vilom", "nadi", "nostril", "shodhan"))

    if want_tada and not want_tree:
        if "tada" in c or "mountain" in c or "palm tree" in c:
            boost += 0.22
        # Demote tree-pose pages when the ask is mountain/tadasana.
        if "perineum" in c or ("foot on" in c and "thigh" in c):
            boost -= 0.15

    if want_tree:
        if "perineum" in c or ("foot" in c and "thigh" in c) or "vrksa" in c:
            boost += 0.25
        if ("tada" in c or "mountain" in c) and "palm" in c:
            boost -= 0.18

    if want_nostril:
        if "nostril" in c or "alternate" in c or "anulom" in c or "nadi" in c:
            boost += 0.22

    if "naukasana" in q or "navasana" in q or "boat pose" in q:
        if "nauk" in c or "boat" in c or "nava" in c:
            boost += 0.25

    return max(-0.2, min(boost, 0.5))


def images_from_matching_chunks(
    query: str,
    chunks: List[Dict],
    top_k: int = 2,
) -> List[Dict]:
    """
    Prefer demo images that sit on the SAME page as a grounded text hit.

    Fixes cases like Naukasana (Fit India SRC003 p.18): text RAG is correct but
    CYP-only CLIP returns unrelated Common Yoga Protocol standing poses.

    Ranking is SEMANTIC: MiniLM(query ↔ page text) blended with the chunk's
    RAG score and a soft page-quality signal (demotes TOC/index pages).
    No pose-name whitelist gate — soft OCR helpers only aid ranking elsewhere.

    image_urls may be missing from older Qdrant payloads — fall back to
    pdf_images_map.json by (source_id, page_number).
    """
    if not chunks or not (query or "").strip():
        return []

    from app.services.embedder import encode_one, encode_texts
    import numpy as np

    q_txt = np.asarray(encode_one(query), dtype=np.float32)
    q_txt = q_txt / (np.linalg.norm(q_txt) + 1e-9)

    page_map = _load_pdf_images_map()
    # Pin likely pages via text RAG + OCR expansions (ranking aid, not a name gate).
    anchors = _text_anchor_pages(query, ["SRC009", "SRC003"], top_n=4)

    def _page_quality(text: str) -> float:
        """Higher = better demo page; TOC/index pages rank lower."""
        score = 0.0
        if any(w in text for w in (
            "technique", "how to", "final position", "what does it measure",
            "sthiti", "benefits",
        )):
            score += 5
        if "caution" in text:
            score += 1
        if "acknowledgement" in text or "table of content" in text:
            score -= 8
        if text.count("years of age") >= 2 and "technique" not in text:
            score -= 4
        return score

    candidates = []
    page_texts = []
    seen_pages = set()
    for c in chunks:
        sid = c.get("source_id")
        page = c.get("page_number")
        if not sid or page is None:
            continue
        text_l = _fold_yoga_text(c.get("text") or "")
        pq = _page_quality(text_l)
        if pq < 0:
            continue

        urls = list(c.get("image_urls") or [])
        if not urls and page_map:
            urls = list(page_map.get(str(sid), {}).get(str(int(page)), []) or [])
        if not urls:
            continue

        page_text = (c.get("text") or "")[:CAPTION_RERANK_CHARS]
        candidates.append((c, urls, pq, sid, int(page)))
        page_texts.append(page_text if page_text.strip() else " ")
        seen_pages.add((str(sid), int(page)))

    # Pull in anchored pages even if they were not in the first RAG hit list.
    for (asid, apage), asc in anchors.items():
        key = (str(asid), int(apage))
        if key in seen_pages:
            continue
        urls = list(page_map.get(str(asid), {}).get(str(int(apage)), []) or [])
        if not urls:
            continue
        fake = {
            "source_id": asid,
            "source_name": asid,
            "page_number": int(apage),
            "text": "",
            "score": float(asc),
        }
        candidates.append((fake, urls, 3.0, asid, int(apage)))
        page_texts.append(" ")
        seen_pages.add(key)

    if not candidates:
        return []

    try:
        page_mat = encode_texts(page_texts)
        sims = (page_mat @ q_txt).tolist()
    except Exception:
        sims = [0.0] * len(candidates)

    scored = []
    for (c, urls, pq, sid, page), text_sim in zip(candidates, sims):
        rag = float(c.get("score") or 0.0)
        blend = (0.50 * rag) + (0.40 * float(text_sim)) + (0.02 * pq)
        for ak, av in anchors.items():
            if str(ak[0]) == str(sid) and int(ak[1]) == int(page):
                blend += 0.28 + 0.15 * float(av)
                break
        scored.append((blend, c, urls, sid, page))

    scored.sort(key=lambda x: -x[0])

    out: List[Dict] = []
    seen_urls = set()
    preferred_sid = None
    for blend, c, urls, sid, page in scored:
        if preferred_sid is not None and str(sid) != str(preferred_sid):
            continue  # do not mix CYP + Fit India demos in one turn
        for url in urls:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if preferred_sid is None:
                preferred_sid = sid
            out.append({
                "image_url": url,
                "source_id": sid,
                "source_name": c.get("source_name") or sid,
                "page_number": int(page),
                "caption": (c.get("text") or "")[:220],
                "score": round(float(blend), 4),
            })
            if len(out) >= top_k:
                return out
    return out


def _load_pdf_images_map() -> Dict[str, Dict[str, List[str]]]:
    """source_id → {page_str → [image_url, ...]} from extract_pdf_images output."""
    candidates = [
        Path(__file__).resolve().parents[2]
        / "data" / "adaptive-fitness-planner-data" / "chunks" / "pdf_images_map.json",
        Path(__file__).resolve().parents[3]
        / "data" / "adaptive-fitness-planner-data" / "chunks" / "pdf_images_map.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import json
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def _text_anchor_pages(
    query: str,
    source_ids: List[str],
    top_n: int = 4,
) -> Dict[tuple, float]:
    """
    Find likely PDF pages via MiniLM text RAG (fitness_guidelines).

    CYP OCR often lacks the full Latin pose name (p.21 says "Tāda means
    mountain" not "tadasana"). Expanded variants bridge that gap so we can
    pin demos to the right page instead of trusting CLIP alone.
    """
    if not query.strip() or not source_ids:
        return {}
    variants = _clip_query_variants(query)
    qfold = _fold_yoga_text(query)
    scores: Dict[tuple, float] = {}
    rag_filters = {
        "trust_tier__in": ["Tier 1", "Tier 2"],
        "source_id__in": list(source_ids),
    }
    for variant in variants:
        chunks = retrieve_guidelines(
            variant, rag_filters, top_k=6, min_results=0,
        )
        for c in chunks:
            sid = c.get("source_id")
            pg = c.get("page_number")
            if sid not in source_ids or pg is None:
                continue
            key = (sid, int(pg))
            sc = float(c.get("score") or 0.0)
            text = _fold_yoga_text(c.get("text") or "")
            if "technique" in text:
                sc += 0.05
            if "tadasana" in qfold or "mountain pose" in qfold:
                if "tada" in text or "palm tree" in text or "mountain" in text:
                    sc += 0.12
            if "vrksasana" in qfold or "tree pose" in qfold:
                if "perineum" in text or ("foot" in text and "thigh" in text):
                    sc += 0.15
            if any(t in qfold for t in ("anulom", "vilom", "nadi", "nostril")):
                if "nostril" in text or "alternate" in text:
                    sc += 0.12
            if any(t in qfold for t in ("naukasana", "navasana", "boat pose")):
                if "nauk" in text or "boat" in text:
                    sc += 0.18
            scores[key] = max(scores.get(key, 0.0), sc)
    if not scores:
        return {}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dict(ranked[:top_n])


def retrieve_guideline_images(
    query: str,
    top_k: int = 2,
    source_ids: Optional[List[str]] = None,
    score_threshold: float = IMAGE_SCORE_THRESHOLD,
) -> List[Dict]:
    """
    Retrieve demo photos for a yoga/protocol question.

    Hybrid ranking (embeddings are fine; pure CLIP rank is not):
      1) MiniLM text RAG pins likely pages (handles OCR naming gaps).
      2) CLIP scores visual match among images.
      3) Caption MiniLM + lexical boost re-rank; strong caption hits can
         surface pages CLIP ranked poorly (e.g. tadasana p.21).

    Returns [] when nothing is confident — callers must not swap in
    unrelated catalog art.
    """
    from app.services.embedder import encode_text_clip

    if source_ids is None:
        source_ids = list(DEMO_IMAGE_SOURCE_IDS)

    client = get_qdrant()
    try:
        existing = [c.name for c in client.get_collections().collections]
        if IMAGE_COLLECTION_NAME not in existing:
            global _IMAGE_COLLECTION_MISSING_WARNED
            if not _IMAGE_COLLECTION_MISSING_WARNED:
                print(
                    f"[RAG] '{IMAGE_COLLECTION_NAME}' collection not found — "
                    "run backend/rag/embed_images_clip.py to enable image retrieval."
                )
                _IMAGE_COLLECTION_MISSING_WARNED = True
            return []
    except Exception as e:
        print(f"[RAG] Qdrant error checking collections: {e}")
        return []

    variants = _clip_query_variants(query)
    if not variants:
        return []

    qfilter = None
    if source_ids:
        qfilter = Filter(must=[
            FieldCondition(key="source_id", match=MatchAny(any=source_ids)),
        ])

    anchor_pages = _text_anchor_pages(
        query, source_ids or list(DEMO_IMAGE_SOURCE_IDS),
    )

    def _row_from_payload(p: Dict, clip_score: float) -> Dict:
        return {
            "image_url": p.get("image_url"),
            "source_id": p.get("source_id"),
            "source_name": p.get("source_name"),
            "page_number": p.get("page_number"),
            "caption": p.get("caption", ""),
            "clip_score": float(clip_score),
        }

    best: Dict[str, Dict] = {}
    fetch_k = max(top_k * 8, 24)
    clip_floor = max(0.15, score_threshold - 0.15)

    for variant in variants:
        try:
            q_vector = encode_text_clip(variant)
        except Exception as e:
            print(f"[RAG] CLIP query embedding failed: {e}")
            continue
        try:
            results = client.query_points(
                collection_name=IMAGE_COLLECTION_NAME,
                query=q_vector,
                query_filter=qfilter,
                limit=fetch_k,
                with_payload=True,
            )
        except Exception as e:
            print(f"[RAG] Qdrant error (images): {e}")
            continue
        for r in results.points:
            clip_score = float(r.score)
            if clip_score < clip_floor:
                continue
            p = r.payload or {}
            url = p.get("image_url")
            if not url:
                continue
            row = _row_from_payload(p, clip_score)
            prev = best.get(url)
            if prev is None or clip_score > prev["clip_score"]:
                best[url] = row

    try:
        from app.services.embedder import encode_one, encode_texts
        import numpy as np

        q_txt = np.asarray(encode_one(query), dtype=np.float32)
        for variant in variants[1:]:
            q_txt = q_txt + np.asarray(encode_one(variant), dtype=np.float32)
        q_txt = q_txt / (np.linalg.norm(q_txt) + 1e-9)

        scrolled, _ = client.scroll(
            collection_name=IMAGE_COLLECTION_NAME,
            scroll_filter=qfilter,
            limit=300,
            with_payload=True,
            with_vectors=False,
        )
        payloads = []
        caption_texts = []
        for pt in scrolled:
            p = pt.payload or {}
            url = p.get("image_url")
            cap = (p.get("caption") or "").strip()
            if not url or not cap:
                continue
            payloads.append(p)
            caption_texts.append(cap[:CAPTION_RERANK_CHARS])
            # Inject text-anchored pages even if CLIP ranked them poorly.
            key = (p.get("source_id"), int(p.get("page_number") or -1))
            if key in anchor_pages and url not in best:
                best[url] = _row_from_payload(p, clip_score=0.22)

        caption_scored = []
        if caption_texts:
            cap_mat = encode_texts(caption_texts)
            sims = cap_mat @ q_txt
            for p, text_sim in zip(payloads, sims.tolist()):
                caption_scored.append((float(text_sim), p))

        caption_scored.sort(key=lambda x: x[0], reverse=True)
        for text_sim, p in caption_scored[:max(top_k * 4, 8)]:
            url = p.get("image_url")
            if url in best:
                best[url]["caption_sim"] = text_sim
                continue
            # Allow caption inject only for strong caption matches.
            if text_sim >= 0.38:
                row = _row_from_payload(p, clip_score=0.22)
                row["caption_sim"] = text_sim
                best[url] = row

        missing = [
            (url, row) for url, row in best.items()
            if "caption_sim" not in row and (row.get("caption") or "").strip()
        ]
        if missing:
            miss_caps = [row["caption"][:CAPTION_RERANK_CHARS] for _, row in missing]
            miss_mat = encode_texts(miss_caps)
            miss_sims = miss_mat @ q_txt
            for (url, row), text_sim in zip(missing, miss_sims.tolist()):
                row["caption_sim"] = float(text_sim)

        for row in best.values():
            cap = (row.get("caption") or "").strip()
            row.setdefault("caption_sim", 0.0)
            lex = _lexical_caption_boost(query, cap)
            page_key = (row.get("source_id"), int(row.get("page_number") or -1))
            anchor_boost = 0.22 if page_key in anchor_pages else 0.0
            row["score"] = round(
                row["clip_score"]
                + CAPTION_RERANK_WEIGHT * row["caption_sim"]
                + lex
                + anchor_boost,
                4,
            )
            row["lex"] = lex
            row["anchored"] = page_key in anchor_pages
    except Exception as e:
        print(f"[RAG] caption re-rank skipped: {e}")
        for row in best.values():
            lex = _lexical_caption_boost(query, row.get("caption") or "")
            page_key = (row.get("source_id"), int(row.get("page_number") or -1))
            anchor_boost = 0.22 if page_key in anchor_pages else 0.0
            row["score"] = round(row["clip_score"] + lex + anchor_boost, 4)
            row["lex"] = lex
            row["anchored"] = page_key in anchor_pages
            row.setdefault("caption_sim", 0.0)

    if not best:
        return []

    out: List[Dict] = []
    for row in sorted(best.values(), key=lambda x: x["score"], reverse=True):
        ok_clip = row["clip_score"] >= score_threshold
        ok_text = bool(row.get("anchored")) and row["score"] >= 0.35
        ok_cap = (
            row.get("caption_sim", 0) >= 0.4
            and row.get("lex", 0) > 0
            and row["score"] >= 0.4
        )
        # Keep if CLIP clears threshold OR text-anchored OR strong caption.
        # No hard pose-name reject filter — soft OCR helpers rank, not gate.
        if not (ok_clip or ok_text or ok_cap):
            continue
        out.append({
            "image_url": row["image_url"],
            "source_id": row["source_id"],
            "source_name": row["source_name"],
            "page_number": row["page_number"],
            "caption": row["caption"],
            "score": row["score"],
        })
        if len(out) >= top_k:
            break
    return out
