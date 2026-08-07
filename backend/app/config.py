"""Path configuration for DB and Qdrant."""
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = BACKEND_ROOT / "data" / "fitness.db"
QDRANT_PATH = BACKEND_ROOT / "rag" / "qdrant_local"
