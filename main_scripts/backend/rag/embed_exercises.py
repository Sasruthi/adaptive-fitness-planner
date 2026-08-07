"""
FILE LOCATION: backend/rag/embed_exercises.py

Builds the `exercise_semantic` Qdrant collection from the SQLite
Exercise table (post load_db.py) — NOT from the raw JSON, so this stays
a single source of truth and only ever contains deduped, cleaned data.

Run after scripts/load_db.py (SQL DB must exist first):
    python rag/embed_exercises.py
"""

from pathlib import Path
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from app.models.models import Exercise  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "fitness.db"
QDRANT_PATH = Path(__file__).resolve().parent / "qdrant_local"
COLLECTION_NAME = "exercise_semantic"
EMBED_DIM = 384
BATCH_SIZE = 64

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)


def build_embedding_text(ex: Exercise) -> str:
    return (
        f"{ex.name}. Targets {ex.target_muscle or ex.body_part}. "
        f"Body part: {ex.body_part or 'unknown'}. "
        f"Equipment: {ex.equipment or 'none'}. "
        f"{(ex.description or '').strip()}"
    ).strip()


def run():
    session = Session()
    exercises = session.query(Exercise).all()
    print(f"Loaded {len(exercises)} exercises from {DB_PATH}")
    if not exercises:
        print("No exercises found — run scripts/load_db.py first.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(path=str(QDRANT_PATH))

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Dropped existing collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    for batch_start in range(0, len(exercises), BATCH_SIZE):
        batch = exercises[batch_start: batch_start + BATCH_SIZE]
        texts = [build_embedding_text(e) for e in batch]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = [
            PointStruct(
                id=ex.id,
                vector=vec.tolist(),
                payload={
                    "name": ex.name,
                    "description": ex.description or "",
                    "body_part": ex.body_part or "",
                    "target_muscle": ex.target_muscle or "",
                    "equipment": ex.equipment or "",
                    "difficulty": ex.difficulty_level or "",
                    "gif_url": ex.gif_url or "",
                    "image_url": getattr(ex, "image_url", None) or "",
                    "video_url": ex.video_url or "",
                }
            )
            for ex, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Upserted {min(batch_start + BATCH_SIZE, len(exercises))}/{len(exercises)}")

    print(f"\nDone. '{COLLECTION_NAME}' — {client.count(COLLECTION_NAME).count} vectors stored.")


if __name__ == "__main__":
    run()
