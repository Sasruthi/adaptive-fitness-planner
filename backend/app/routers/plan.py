"""
FILE LOCATION: backend/app/routers/plan.py

/api/plan/generate  — POST: generate full plan from profile + filters
/api/plan/save      — POST: persist plan to DB via action tools
/api/plan/{email}   — GET:  retrieve user's active plan
/api/plan/calories  — POST: calculate daily calorie target
"""
import json
from fastapi import APIRouter, HTTPException
from app.schemas.models import (
    GeneratePlanRequest, SavePlanRequest, SavePlanResponse,
    CalorieRequest,
)
from app.services.plan_agent import generate_plan_agentic
from app.tools.actions import save_plan, calculate_calories, get_user_plan
from app.conversation.state import build_sql_filters, build_rag_filters

router = APIRouter(prefix="/api/plan", tags=["Plan"])


def _normalize_profile(profile: dict) -> dict:
    p = dict(profile)
    if not p.get("sex") and p.get("gender"):
        p["sex"] = p["gender"]
    return p


@router.post("/generate")
async def generate(req: GeneratePlanRequest):
    """
    Generate a complete 7-day fitness plan.
    Uses the LangGraph plan agent (tool-calling loop) with a direct-pipeline fallback.
    After getting the plan, call /api/plan/save to persist it.

    sql_filters/rag_filters are optional — if the caller doesn't send them
    (or sends null), they're derived from the profile server-side. This
    is intentionally defensive: a previous version required them, which
    meant any client that didn't have them cached (e.g. after a schema
    change on this end) got a 422 with no useful error message instead of
    a working plan.
    """
    try:
        profile = _normalize_profile(req.profile)
        sql_filters = req.sql_filters or build_sql_filters(profile)
        rag_filters = req.rag_filters or build_rag_filters(profile)
        plan = generate_plan_agentic(profile, sql_filters, rag_filters)
        return {"success": True, "plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan generation failed: {str(e)}")


@router.post("/save", response_model=SavePlanResponse)
async def save(req: SavePlanRequest):
    """
    Persist the generated plan to the database.
    Deactivates any previous active plan for this user.
    Returns plan_id to use for workout logging.
    """
    try:
        result = save_plan(
            user_email=req.user_email,
            user_name=req.user_name,
            goal=req.goal,
            plan_json=json.dumps(req.plan),
            constraints_json="{}",
        )
        return SavePlanResponse(
            success=result["success"],
            plan_id=result.get("plan_id"),
            user_id=result.get("user_id"),
            message=result.get("message", result.get("error", "Unknown error")),
            email_sent=result.get("email_sent", False),
            email_note=result.get("email_note"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@router.get("/{user_email}")
async def get_plan(user_email: str):
    """
    Retrieve the user's current active plan from the database.
    Returns full week_plan, diet_plan, safety_notes, and citations.
    """
    try:
        result = get_user_plan(user_email)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calories")
async def calories(req: CalorieRequest):
    """
    Calculate daily calorie target (Mifflin-St Jeor BMR).
    Returns target calories, macros, and India-specific meal calorie guide.
    Can be called mid-conversation to personalise the diet section.
    """
    try:
        result = calculate_calories(
            age=req.age, sex=req.sex,
            weight_kg=req.weight_kg, height_cm=req.height_cm,
            activity_level=req.activity_level, goal=req.goal,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
