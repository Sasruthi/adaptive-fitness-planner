"""
Exercise Retrieval Service
==========================
Queries SQLite exercise catalog using the sql_filters
produced by the conversational intake (Module 3).

Principles:
- Structured data stays in SQL, never goes through RAG
- Health flags become hard exclusion filters, not suggestions
- Falls back gracefully if filters return zero results
- Returns rich exercise objects including gif_url for frontend rendering
"""

from pathlib import Path
from typing import List, Dict, Optional
from sqlalchemy import create_engine, or_, and_
from sqlalchemy.orm import sessionmaker

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.models.models import Exercise

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "fitness.db"

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)


def get_exercises(
    sql_filters: Dict,
    limit: int = 20,
    min_results: int = 3,
) -> List[Dict]:
    """
    Retrieve exercises matching the user's profile filters.
    Equipment filter is treated as a HARD constraint — never relaxed.
    Only body_part and difficulty fall back if results are thin.
    """
    from app.services.exercise_rag import (
        _is_bodyweight_only_filter,
        _requires_apparatus,
    )

    session = Session()
    try:
        # Over-fetch when bodyweight so we can drop bar/rings moves tagged "body only"
        equip = sql_filters.get("equipment__in") or []
        strict_floor = _is_bodyweight_only_filter(equip)
        fetch_limit = limit * 4 if strict_floor else limit

        exercises = _query(session, sql_filters, fetch_limit)

        # Fallback 1: relax difficulty only (NOT equipment)
        if len(exercises) < min_results and sql_filters.get("difficulty_level"):
            relaxed = {k: v for k, v in sql_filters.items() if k != "difficulty_level"}
            exercises = _query(session, relaxed, fetch_limit)

        # Fallback 2: relax body_part only if still thin (keep equipment hard)
        if len(exercises) < min_results and sql_filters.get("body_part__in"):
            relaxed = {k: v for k, v in sql_filters.items()
                       if k not in ("body_part__in", "difficulty_level")}
            exercises = _query(session, relaxed, fetch_limit)

        out = [_serialize(e) for e in exercises]
        if strict_floor:
            out = [
                e for e in out
                if not _requires_apparatus(e.get("name", ""), e.get("description", ""))
            ]
        return out[:limit]

    finally:
        session.close()


def _query(session, filters: Dict, limit: int) -> List[Exercise]:
    q = session.query(Exercise)

    # Body part filter (OR across target parts)
    if filters.get("body_part__in"):
        parts = filters["body_part__in"]
        conditions = [
            Exercise.body_part.ilike(f"%{p}%") for p in parts
        ] + [
            Exercise.target_muscle.ilike(f"%{p}%") for p in parts
        ]
        q = q.filter(or_(*conditions))

    # Equipment filter
    if filters.get("equipment__in"):
        equip_conditions = [
            Exercise.equipment.ilike(f"%{e}%")
            for e in filters["equipment__in"]
        ]
        q = q.filter(or_(*equip_conditions))

    # Difficulty filter — also allow NULL/empty so hasaneyldrm rows
    # (no difficulty field in source) aren't excluded when user is beginner/etc.
    if filters.get("difficulty_level"):
        level = filters["difficulty_level"]
        q = q.filter(or_(
            Exercise.difficulty_level.ilike(f"%{level}%"),
            Exercise.difficulty_level.is_(None),
            Exercise.difficulty_level == "",
        ))

    # Hard exclusions (health flags)
    if filters.get("exclude_body_parts"):
        for excluded in filters["exclude_body_parts"]:
            q = q.filter(
                ~Exercise.body_part.ilike(f"%{excluded}%")
            )

    # Prefer exercises with media (gif_url), then by rating
    q = q.order_by(
        Exercise.has_media.desc(),
        Exercise.rating.desc().nullslast(),
    )

    return q.limit(limit).all()


def _serialize(e: Exercise) -> Dict:
    return {
        "id":             e.id,
        "name":           e.name,
        "description":    e.description or "",
        "category":       e.category or "",
        "body_part":      e.body_part or "",
        "target_muscle":  e.target_muscle or "",
        "equipment":      e.equipment or "",
        "difficulty":     e.difficulty_level or "",
        "rating":         e.rating,
        "gif_url":        e.gif_url or "",
        "image_url":      getattr(e, "image_url", None) or "",
        "video_url":      e.video_url or "",
        "has_media":      e.has_media,
    }


def get_taxonomy(kind: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Returns controlled vocabulary lists for the frontend slot-filling UI.
    kind: 'body_part' | 'equipment' | 'muscle' | None (returns all)
    """
    from app.models.models import Taxonomy
    session = Session()
    try:
        q = session.query(Taxonomy)
        if kind:
            q = q.filter(Taxonomy.kind == kind)
        result: Dict[str, List[str]] = {}
        for t in q.all():
            result.setdefault(t.kind, []).append(t.name)
        return result
    finally:
        session.close()