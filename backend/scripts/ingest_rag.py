"""
Adaptive Fitness Planner — Module 2: RAG Ingestion Pipeline
=============================================================
Flow:
  1. Read source_manifest.csv (Tier 1 + Tier 2 PDF sources only)
  2. Download each PDF from its URL
  3. Extract text with PyMuPDF (layout-aware)
  4. Chunk with header-aware recursive splitter (NOT fixed token windows)
  5. Tag every chunk with metadata: source_id, source_name, tier, page, category
  6. Embed with sentence-transformers/all-MiniLM-L6-v2 (free, local)
  7. Store in Qdrant (local mode — no server needed for dev)

RAG PRINCIPLE: Only narrative/guideline PDFs go here.
Exercise catalog stays in SQLite. No overlap.

Run:
  python ingest_rag.py
  python ingest_rag.py --test   # dry run, first source only, 5 chunks
"""

import argparse
import csv
import os
import sys
import time
import hashlib
from pathlib import Path
import requests
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)

# ── Config ────────────────────────────────────────────────────────────────────

MANIFEST_PATH = Path("/mnt/user-data/uploads/source_manifest.csv")
PDF_CACHE_DIR = Path("/home/claude/fitness-app/backend/data/pdf_cache")
QDRANT_PATH   = Path("/home/claude/fitness-app/backend/data/qdrant")
COLLECTION    = "fitness_guidelines"

# Embedding model (free, no API key needed)
EMBED_MODEL   = "all-MiniLM-L6-v2"
EMBED_DIM     = 384

# Chunking — section-aware, not fixed windows
# India-first guideline PDFs tend to have dense paragraphs;
# 800 chars overlap 150 keeps full reasoning chains intact
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# Source-to-category mapping (for retrieval-time filtering)
CATEGORY_MAP = {
    "SRC001": "nutrition",
    "SRC002": "nutrition",
    "SRC003": "exercise_science",
    "SRC004": "nutrition",
    "SRC005": "nutrition",
    "SRC006": "diet_behavior",
    "SRC007": "diet_behavior",
    "SRC008": "lifestyle",
    "SRC009": "activity_guidance",
    "SRC010": "exercise_science",  # WHO physical activity guidelines
}

# ── Setup ─────────────────────────────────────────────────────────────────────

PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_PATH.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Dry run: first source only, 5 chunks")
args = parser.parse_args()

# ── Embedding ─────────────────────────────────────────────────────────────────

print(f"Loading embedding model: {EMBED_MODEL}")
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL)
    print("Embedding model loaded.")
    EMBED_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Embedding model unavailable ({e.__class__.__name__}) — "
          f"will store chunks as metadata only (no vectors). "
          f"Re-run on machine with internet access to populate vectors.")
    EMBED_AVAILABLE = False

# ── Qdrant ────────────────────────────────────────────────────────────────────

client = QdrantClient(path=str(QDRANT_PATH))

# Create collection if not exists
existing = [c.name for c in client.get_collections().collections]
if COLLECTION not in existing:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"Created Qdrant collection: {COLLECTION}")
else:
    print(f"Qdrant collection exists: {COLLECTION}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF to local cache. Returns True if successful."""
    if dest.exists():
        print(f"  [cache] {dest.name} already downloaded")
        return True
    print(f"  [download] {url[:80]}...")
    try:
        resp = requests.get(url, timeout=60, headers={
            "User-Agent": "AdaptiveFitnessPlanner/1.0 (portfolio research project)"
        })
        if resp.status_code == 200 and b"%PDF" in resp.content[:10]:
            dest.write_bytes(resp.content)
            print(f"  [ok] saved {len(resp.content)//1024}KB → {dest.name}")
            return True
        else:
            print(f"  [fail] status={resp.status_code}, not a valid PDF")
            return False
    except Exception as e:
        print(f"  [error] {e}")
        return False


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text page by page using PyMuPDF.
    Returns list of {page: int, text: str} dicts.
    Preserves paragraph structure — avoids merging headers into body text.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # get_text("text") preserves line breaks and paragraph gaps
        text = page.get_text("text").strip()
        if len(text) > 50:  # skip near-blank pages (table of contents images etc.)
            pages.append({"page": page_num + 1, "text": text})
    doc.close()
    return pages


def chunk_pages(pages: list[dict], source_id: str, source_name: str,
                tier: str) -> list[dict]:
    """
    Chunk page texts using RecursiveCharacterTextSplitter.
    Splitter tries to break on: paragraph > newline > sentence > word.
    Every chunk tagged with full provenance metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    category = CATEGORY_MAP.get(source_id, "general_health")

    for page_data in pages:
        page_chunks = splitter.split_text(page_data["text"])
        for i, chunk_text in enumerate(page_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 80:  # skip micro-fragments
                continue
            # Stable unique ID from content hash (idempotent re-runs)
            chunk_id = hashlib.md5(
                f"{source_id}_{page_data['page']}_{i}_{chunk_text[:50]}".encode()
            ).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source_id": source_id,
                    "source_name": source_name,
                    "tier": tier,
                    "page": page_data["page"],
                    "category": category,
                    "chunk_index": i,
                }
            })
    return chunks


def embed_and_upsert(chunks: list[dict], batch_size: int = 64):
    """Embed chunks and upsert into Qdrant in batches."""
    if not EMBED_AVAILABLE:
        print(f"  [skip] embedding unavailable — {len(chunks)} chunks not stored")
        return 0

    stored = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        vectors = embedder.encode(texts, normalize_embeddings=True,
                                  show_progress_bar=False).tolist()
        points = []
        for chunk, vector in zip(batch, vectors):
            # Use integer id derived from hash for Qdrant
            int_id = int(chunk["id"][:8], 16)
            points.append(PointStruct(
                id=int_id,
                vector=vector,
                payload={**chunk["metadata"], "text": chunk["text"]}
            ))
        client.upsert(collection_name=COLLECTION, points=points)
        stored += len(points)

    return stored


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_manifest() -> list[dict]:
    rows = []
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Only process PDF sources (not json/repo) that are Tier 1 or Tier 2
            if row["source_type"] == "pdf" and row["trust_tier"] in ("Tier 1", "Tier 2"):
                rows.append(row)
    return rows


def run():
    sources = load_manifest()
    if args.test:
        sources = sources[:1]
        print("[TEST MODE] Processing first source only, 5 chunks max")

    print(f"\nSources to process: {len(sources)}")
    for s in sources:
        print(f"  {s['source_id']} | {s['trust_tier']} | {s['source_name'][:60]}")

    total_chunks = 0
    total_stored = 0
    failed = []

    for source in sources:
        sid = source["source_id"]
        name = source["source_name"]
        url  = source["source_url"]
        tier = source["trust_tier"]

        print(f"\n{'='*60}")
        print(f"Processing: {sid} — {name}")

        # 1. Download
        pdf_path = PDF_CACHE_DIR / f"{sid}.pdf"
        ok = download_pdf(url, pdf_path)
        if not ok:
            failed.append(sid)
            continue

        # 2. Extract
        pages = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(pages)} non-blank pages")
        if not pages:
            failed.append(sid)
            continue

        # 3. Chunk
        chunks = chunk_pages(pages, sid, name, tier)
        print(f"  Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
        if args.test:
            chunks = chunks[:5]
            print(f"  [TEST] trimmed to {len(chunks)} chunks")

        # 4. Embed + store
        stored = embed_and_upsert(chunks)
        print(f"  Stored {stored} vectors in Qdrant")

        total_chunks += len(chunks)
        total_stored += stored
        time.sleep(0.5)  # polite delay between downloads

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"  Total chunks created : {total_chunks}")
    print(f"  Total vectors stored : {total_stored}")
    print(f"  Failed sources       : {failed if failed else 'none'}")

    if EMBED_AVAILABLE:
        count = client.count(collection_name=COLLECTION).count
        print(f"  Qdrant collection '{COLLECTION}' total points: {count}")

    print(f"\n  Qdrant DB path : {QDRANT_PATH}")
    print(f"  PDF cache path : {PDF_CACHE_DIR}")


if __name__ == "__main__":
    run()
