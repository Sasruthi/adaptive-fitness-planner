"""
FILE LOCATION: backend/app/services/nutrition_lookup.py

Semantic lookup over the nutrition_items table (INDB — 1,014 Indian
recipes/foods, see models/nutrition_models.py for provenance).

Purpose: the diet-plan LLM is good at choosing WHICH Indian dishes make
sense for a meal slot, and at writing notes — it is NOT reliable for
stating calories/macros. This module takes whatever dish name the LLM
suggested and resolves it to a REAL row's verified macros, using the
same embedding approach as exercise_rag.py (meaning-based match, not
exact substring — "roti" must still find "Missi roti (Gram flour bread)").

Nothing here invents numbers. If nothing matches above a confidence
threshold, the caller is told explicitly (`matched=False`) rather than
silently guessing.
"""

from pathlib import Path
from typing import Dict, List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.models.nutrition_models import NutritionItem

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "fitness.db"
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

MATCH_CONFIDENCE_THRESHOLD = 0.45  # cosine similarity floor before we trust a match

_corpus_names: List[str] = []
_corpus_embeddings = None  # lazy-built, cached in process memory


def _get_embed_model():
    from app.services.embedder import get_shared_embed_model
    return get_shared_embed_model()


def _build_corpus():
    """Embed every food_name once per process and cache it — 1,014 rows
    is cheap, and this avoids re-querying/re-embedding on every lookup."""
    global _corpus_names, _corpus_embeddings
    if _corpus_embeddings is not None:
        return
    session = Session()
    try:
        rows = session.query(NutritionItem.food_code, NutritionItem.food_name).all()
    finally:
        session.close()
    _corpus_names = [(r[0], r[1]) for r in rows]
    model = _get_embed_model()
    _corpus_embeddings = model.encode(
        [n for _, n in _corpus_names], normalize_embeddings=True
    )


def lookup_food(query: str, top_k: int = 3) -> List[Dict]:
    """Semantic search over food names. Returns ranked matches with score."""
    _build_corpus()
    if not _corpus_names:
        return []
    import numpy as np
    model = _get_embed_model()
    q_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = _corpus_embeddings @ q_vec  # cosine similarity (both normalized)
    top_idx = np.argsort(-scores)[:top_k]

    session = Session()
    try:
        results = []
        for idx in top_idx:
            code, name = _corpus_names[idx]
            item = session.query(NutritionItem).filter_by(food_code=code).first()
            if not item:
                continue
            results.append({
                "food_code": item.food_code,
                "food_name": item.food_name,
                "score": round(float(scores[idx]), 4),
                "energy_kcal": item.energy_kcal,
                "protein_g": item.protein_g,
                "carb_g": item.carb_g,
                "fat_g": item.fat_g,
                "fibre_g": item.fibre_g,
                "servings_unit": item.servings_unit,
                "serving_energy_kcal": item.serving_energy_kcal,
                "serving_protein_g": item.serving_protein_g,
                "serving_carb_g": item.serving_carb_g,
                "serving_fat_g": item.serving_fat_g,
            })
        return results
    finally:
        session.close()


def get_verified_macros(dish_name: str, per_serving: bool = True) -> Optional[Dict]:
    """
    Best-match lookup for a single dish name. Returns None (not a guess)
    if nothing clears the confidence threshold — caller must handle that
    case explicitly rather than fabricating a number.
    """
    hits = lookup_food(dish_name, top_k=1)
    if not hits or hits[0]["score"] < MATCH_CONFIDENCE_THRESHOLD:
        return None
    h = hits[0]
    if per_serving and h.get("serving_energy_kcal"):
        return {
            "matched_food": h["food_name"],
            "food_code": h["food_code"],
            "match_score": h["score"],
            "unit": h.get("servings_unit") or "serving",
            "calories": round(h["serving_energy_kcal"], 1),
            "protein_g": round(h["serving_protein_g"], 1),
            "carb_g": round(h["serving_carb_g"], 1),
            "fat_g": round(h["serving_fat_g"], 1),
        }
    return {
        "matched_food": h["food_name"],
        "food_code": h["food_code"],
        "match_score": h["score"],
        "unit": "100g",
        "calories": round(h["energy_kcal"], 1),
        "protein_g": round(h["protein_g"], 1),
        "carb_g": round(h["carb_g"], 1),
        "fat_g": round(h["fat_g"], 1),
    }
