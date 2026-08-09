"""
CLIP ingest for guideline demonstration photos
==============================================
Embeds PNGs listed in pdf_images_map.json into Qdrant collection
`guideline_images` so chat can retrieve demos by visual/semantic match.

Pipeline position
-----------------
  extract_pdf_images.py  →  pdf_images_map.json (+ optional all_chunks.json)
  embed_images_clip.py   →  Qdrant guideline_images (512-dim cosine, CLIP)

Model / store
-------------
- Embedding: clip-ViT-B-32 via sentence-transformers (image tower only at ingest)
- Store:     backend/rag/qdrant_local (same disk store as fitness_guidelines)
- Dim:       512 (CLIP_DIM)

What is embedded vs metadata
----------------------------
- Embedded: raw image pixels (CLIP image tower).
- Metadata only: pseudo-caption = first text chunk on the same (source_id, page).
  Captions support citations / caption re-rank at query time; they are not the
  ingest embedding.

Which sources are ingested
--------------------------
DEMO_EMBED_SOURCE_IDS — pose/technique booklets only (default SRC009, SRC003).
Food/booklet art may still appear in pdf_images_map for chunk.image_urls, but
is excluded here so it does not pollute yoga/demo retrieval.

Payload schema (per point)
--------------------------
  source_id, source_name, trust_tier, country, page_number,
  image_url (/static/...), caption (str)

Ops
---
Local Qdrant is single-writer — stop the API before running this script.
Rebuild drops and recreates the collection.

Run:
    python backend/rag/extract_pdf_images.py   # if map/PNGs stale
    python backend/rag/embed_images_clip.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "services"))
from embedder import encode_images_clip_batch, CLIP_DIM  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]                     # backend/
DATA_ROOT    = PROJECT_ROOT / "data" / "adaptive-fitness-planner-data"
CHUNK_DIR    = DATA_ROOT / "chunks"
# image_url is "/static/..." relative to backend/
BACKEND_ROOT = PROJECT_ROOT
IMAGES_MAP_FILE = CHUNK_DIR / "pdf_images_map.json"
CHUNK_FILE      = CHUNK_DIR / "all_chunks.json"

QDRANT_PATH     = Path(__file__).resolve().parent / "qdrant_local"
COLLECTION_NAME = "guideline_images"
BATCH_SIZE      = 16
# Page text for metadata / retrieval re-rank; keep long enough for late headings.
CAPTION_MAX_CHARS = 700

# Pose/technique demo sources only (see module docstring).
DEMO_EMBED_SOURCE_IDS = ("SRC009", "SRC003")


def _load_images_map() -> Dict[str, Dict[str, List[str]]]:
    if not IMAGES_MAP_FILE.exists():
        raise FileNotFoundError(
            f"{IMAGES_MAP_FILE} not found — run extract_pdf_images.py first."
        )
    return json.loads(IMAGES_MAP_FILE.read_text(encoding="utf-8"))


def _build_caption_lookup() -> Dict[str, str]:
    """
    Map 'source_id::page_number' → best chunk text on that page (truncated).

    Prefers chunks that look like pose technique pages (title + Technique)
    over short header/TOC fragments, so caption re-rank is less noisy.
    """
    lookup: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    if not CHUNK_FILE.exists():
        print(f"[WARN] {CHUNK_FILE} not found — captions will be empty. "
              f"Run chunk_all_sources.py first for better metadata.")
        return lookup
    chunks = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        key = f"{c.get('source_id')}::{c.get('page_number')}"
        low = text.lower()
        score = 0.0
        if "technique" in low:
            score += 3.0
        if "āsana" in text.lower() or "asana" in low or "prāṇāyām" in text.lower() or "pranayam" in low:
            score += 2.0
        if re.search(r"[A-ZĀĪŪṚṄÑṬḌṆŚṢṂḤ]{3,}[A-ZĀĪŪṚṄÑṬḌṆŚṢṂḤa-zāīūṛṅñṭḍṇśṣṃḥ]*ĀSANA", text):
            score += 2.0
        if "benefits" in low:
            score += 0.5
        # Penalize pure TOC / index fragments.
        if low.count("asana") + text.lower().count("āsana") >= 4 and "technique" not in low:
            score -= 5.0
        score += min(len(text), 400) / 400.0
        if key not in scores or score > scores[key]:
            scores[key] = score
            lookup[key] = text[:CAPTION_MAX_CHARS].strip()
    return lookup


def _source_metadata(source_id: str) -> Dict:
    """
    Resolve source_name / trust_tier / country from the chunker manifest.
    FILE:<stem> ids without a manifest row get Unknown placeholders.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from chunk_all_sources import load_manifest, MANIFESTS  # type: ignore
        manifest = load_manifest(MANIFESTS)
        row = manifest.get(source_id)
        if row:
            return {
                "source_name": row.get("source_name", source_id),
                "trust_tier": row.get("trust_tier", "Unknown"),
                "country": row.get("country", "Unknown"),
            }
    except Exception:
        pass
    return {"source_name": source_id, "trust_tier": "Unknown", "country": "Unknown"}


def run():
    images_map = _load_images_map()
    caption_lookup = _build_caption_lookup()

    # Local Qdrant is single-writer — stop the API if this raises AlreadyLocked.
    client = QdrantClient(path=str(QDRANT_PATH))
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=CLIP_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection: {COLLECTION_NAME} (dim={CLIP_DIM}, cosine)")

    rows = []
    skipped_sources = []
    for source_id, pages in images_map.items():
        if source_id not in DEMO_EMBED_SOURCE_IDS:
            skipped_sources.append(source_id)
            continue
        meta = _source_metadata(source_id)
        for page_number, urls in pages.items():
            for url in urls:
                rows.append((source_id, int(page_number), url, meta))

    if skipped_sources:
        print(
            f"Skipping non-demo sources for CLIP ingest: {', '.join(sorted(skipped_sources))} "
            f"(kept for chunk image_urls only)"
        )
    print(f"Embedding {len(rows)} images across "
          f"{len({r[0] for r in rows})} demo source doc(s)\n")

    point_id = 0
    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start: batch_start + BATCH_SIZE]
        local_paths = []
        valid_batch = []
        for item in batch:
            _, _, url, _ = item
            path = BACKEND_ROOT / url.lstrip("/")
            if not path.is_file():
                print(f"[WARN] missing image file, skip: {path}")
                continue
            local_paths.append(path)
            valid_batch.append(item)
        if not valid_batch:
            continue

        try:
            vectors = encode_images_clip_batch(local_paths)
        except Exception as e:
            print(f"[FATAL] CLIP embedding failed: {e}")
            print("        Check that Pillow + sentence-transformers are installed and "
                  "the clip-ViT-B-32 weights could be downloaded (first run needs network).")
            client.close()
            raise

        points = []
        for (source_id, page_number, url, meta), vector in zip(valid_batch, vectors):
            caption = caption_lookup.get(f"{source_id}::{page_number}", "")
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "source_id":     source_id,
                    "source_name":   meta["source_name"],
                    "trust_tier":    meta["trust_tier"],
                    "country":       meta["country"],
                    "page_number":   page_number,
                    "image_url":     url,
                    "caption":       caption,  # metadata only — not embedded
                },
            ))
            point_id += 1

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Upserted {min(batch_start + BATCH_SIZE, len(rows))}/{len(rows)}")

    print(f"\n{'='*55}")
    print(f"Collection '{COLLECTION_NAME}' — {client.count(COLLECTION_NAME).count} image vectors stored")

    from embedder import encode_text_clip
    for test_query in ["how to do alternate nostril breathing pranayama", "butterfly yoga pose"]:
        print(f"\nTEST RETRIEVAL — '{test_query}'")
        q_vec = encode_text_clip(test_query)
        results = client.query_points(
            collection_name=COLLECTION_NAME, query=q_vec, limit=3, with_payload=True,
        )
        for r in results.points:
            p = r.payload
            print(f"  score={r.score:.4f} | {p['source_id']} p.{p['page_number']} | {p['image_url']}")
            print(f"    caption: {p['caption'][:120]}...")

    client.close()


if __name__ == "__main__":
    run()
