"""
FILE LOCATION: backend/app/routers/exercises.py

/api/exercises/search    — POST: search exercises with filters
/api/exercises/taxonomy  — GET:  body parts, equipment, muscles for UI dropdowns
/api/exercises/{id}      — GET:  single exercise detail
"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from sqlalchemy import or_
from app.schemas.models import ExerciseSearchRequest, ExerciseResponse
from app.services.exercise_retrieval import get_exercises, get_taxonomy, Session
from app.models.models import Exercise

router = APIRouter(prefix="/api/exercises", tags=["Exercises"])


@router.post("/search", response_model=List[ExerciseResponse])
async def search(req: ExerciseSearchRequest):
    """
    Search exercises with optional filters.
    Used by the frontend to let users browse exercises outside the plan.
    """
    try:
        sql_filters = {}
        if req.body_part:  sql_filters["body_part__in"]  = [req.body_part]
        if req.equipment:  sql_filters["equipment__in"]   = [req.equipment]
        if req.difficulty: sql_filters["difficulty_level"]= req.difficulty

        # Free-text query: name search WITH filters still applied
        if req.query:
            session = Session()
            try:
                q = session.query(Exercise).filter(Exercise.name.ilike(f"%{req.query}%"))
                if req.body_part:
                    q = q.filter(Exercise.body_part.ilike(f"%{req.body_part}%"))
                if req.equipment:
                    q = q.filter(Exercise.equipment.ilike(f"%{req.equipment}%"))
                if req.difficulty:
                    q = q.filter(Exercise.difficulty_level.ilike(f"%{req.difficulty}%"))
                results = q.limit(req.limit).all()
                return [ExerciseResponse(**_serialize(e)) for e in results]
            finally:
                session.close()

        exercises = get_exercises(sql_filters, limit=req.limit)
        return [ExerciseResponse(**e) for e in exercises]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/taxonomy")
async def taxonomy():
    """
    Returns controlled vocabulary lists for the frontend.
    Used to populate dropdowns during the conversational intake UI.
    """
    try:
        return get_taxonomy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{exercise_id}", response_model=ExerciseResponse)
async def get_exercise(exercise_id: int):
    """
    Get a single exercise by ID.
    Used when user taps an exercise card in the frontend.
    """
    session = Session()
    try:
        ex = session.query(Exercise).filter(Exercise.id == exercise_id).first()
        if not ex:
            raise HTTPException(status_code=404, detail=f"Exercise {exercise_id} not found")
        return ExerciseResponse(**_serialize(ex))
    finally:
        session.close()


def _serialize(e: Exercise) -> dict:
    return {
        "id":            e.id,
        "name":          e.name,
        "description":   e.description or "",
        "category":      e.category or "",
        "body_part":     e.body_part or "",
        "target_muscle": e.target_muscle or "",
        "equipment":     e.equipment or "",
        "difficulty":    e.difficulty_level or "",
        "rating":        e.rating,
        "gif_url":       e.gif_url or "",
        "image_url":     getattr(e, "image_url", None) or "",
        "video_url":     getattr(e, "video_url", None) or "",
        "has_media":     e.has_media or False,
    }
