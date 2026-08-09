"""
Embed guideline text chunks into local Qdrant (fitness_guidelines).

Inputs:  data/.../chunks/all_chunks.json from chunk_all_sources.py
Outputs: backend/rag/qdrant_local collection fitness_guidelines
         (all-MiniLM-L6-v2, 384-dim, cosine)

Stop the API before running — local Qdrant is single-writer.

Run:  python backend/rag/embed_and_store.py
"""

import json, time
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue
)

try:
    from sentence_transformers import SentenceTransformer
    MODEL_NAME = "all-MiniLM-L6-v2"
    model = SentenceTransformer(MODEL_NAME)
    EMBED_DIM = 384
    print(f"Embedding model loaded: {MODEL_NAME} ({EMBED_DIM}-dim)")
    USE_MODEL = True
except Exception as e:
    print(f"sentence-transformers unavailable ({e}), using random vectors for schema test")
    import numpy as np
    USE_MODEL = False
    EMBED_DIM = 384

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
DATA_ROOT       = PROJECT_ROOT / "data" / "adaptive-fitness-planner-data"
CHUNK_FILE      = DATA_ROOT / "chunks" / "all_chunks.json"
QDRANT_PATH     = Path(__file__).resolve().parent / "qdrant_local"
COLLECTION_NAME = "fitness_guidelines"
BATCH_SIZE      = 32

QDRANT_PATH.mkdir(parents=True, exist_ok=True)

# ── Connect to local Qdrant (persisted on disk) ───────────────────────────────
client = QdrantClient(path=str(QDRANT_PATH))

# Drop + recreate for clean rebuild
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    client.delete_collection(COLLECTION_NAME)
    print(f"Dropped existing collection: {COLLECTION_NAME}")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
)
print(f"Created collection: {COLLECTION_NAME} (dim={EMBED_DIM}, cosine)")

# ── Load chunks ───────────────────────────────────────────────────────────────
with open(CHUNK_FILE, encoding="utf-8") as f:
    all_loaded_chunks = json.load(f)
print(f"Loaded {len(all_loaded_chunks)} chunks from {CHUNK_FILE}")

# FIX: previously every chunk was embedded here with no filtering, which
# meant json_exercise chunks (raw exercise records) ended up mixed into
# the guideline collection, diluting retrieval for nutrition/safety
# questions. Exercises get their own collection — see embed_exercises.py,
# which builds `exercise_semantic` from the cleaned SQL table instead.
EXCLUDED_DOC_TYPES = {"json_exercise"}
chunks = [c for c in all_loaded_chunks if c.get("doc_type") not in EXCLUDED_DOC_TYPES]
excluded_count = len(all_loaded_chunks) - len(chunks)
if excluded_count:
    print(f"Excluding {excluded_count} chunks with doc_type in {EXCLUDED_DOC_TYPES} "
          f"(these belong in exercise_semantic via embed_exercises.py, not here)")
print(f"Embedding {len(chunks)} guideline chunks into '{COLLECTION_NAME}'\n")

# ── Embed + upsert in batches ─────────────────────────────────────────────────
def embed_texts(texts: List[str]) -> List[List[float]]:
    if USE_MODEL:
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    else:
        import numpy as np
        vecs = np.random.rand(len(texts), EMBED_DIM).astype("float32")
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs.tolist()

points = []
total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(total_batches):
    batch = chunks[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
    texts = [c["text"] for c in batch]
    vectors = embed_texts(texts)

    for i, (chunk, vector) in enumerate(zip(batch, vectors)):
        point = PointStruct(
            id=batch_idx * BATCH_SIZE + i,
            vector=vector,
            payload={
                "chunk_id":     chunk["chunk_id"],
                "text":         chunk["text"],
                "source_id":    chunk["source_id"],
                "source_name":  chunk["source_name"],
                "trust_tier":   chunk["trust_tier"],
                "category":     chunk.get("category", chunk.get("content_type")),
                "country":      chunk["country"],
                "page_number":  chunk["page_number"],
                "section_title":chunk["section_title"],
                "content_type": chunk["content_type"],
                "char_count":   chunk["char_count"],
                # Legacy same-page proximity photos (CLIP collection is preferred).
                "image_urls":   chunk.get("image_urls") or [],
            }
        )
        points.append(point)

    client.upsert(collection_name=COLLECTION_NAME, points=points[-len(batch):])

    if (batch_idx + 1) % 5 == 0 or batch_idx == total_batches - 1:
        print(f"  Upserted batch {batch_idx+1}/{total_batches} "
              f"({min((batch_idx+1)*BATCH_SIZE, len(chunks))}/{len(chunks)} chunks)")

# ── Verify with a test retrieval ──────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"Collection '{COLLECTION_NAME}' — {client.count(COLLECTION_NAME).count} vectors stored")
print()
print("TEST RETRIEVAL — 'protein requirements for vegetarians India'")

query = "protein requirements for vegetarians India"
q_vec = embed_texts([query])[0]

from qdrant_client.models import Query as QdrantQuery

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=q_vec,
    limit=3,
    with_payload=True,
)

for r in results.points:
    p = r.payload
    print(f"\n  Score: {r.score:.4f} | {p['source_id']} | page {p['page_number']} | {p['content_type']}")
    print(f"  Source: {p['source_name']}")
    print(f"  Section: {p['section_title']}")
    print(f"  Text: {p['text'][:200]}...")

print()
print("TEST RETRIEVAL — 'physical activity guidelines high blood pressure'")
q2_vec = embed_texts(["physical activity guidelines high blood pressure"])[0]
results2 = client.query_points(
    collection_name=COLLECTION_NAME,
    query=q2_vec,
    limit=3,
    with_payload=True,
)
for r in results2.points:
    p = r.payload
    print(f"\n  Score: {r.score:.4f} | {p['source_id']} | page {p['page_number']} | {p['content_type']}")
    print(f"  Source: {p['source_name']}")
    print(f"  Text: {p['text'][:200]}...")

client.close()