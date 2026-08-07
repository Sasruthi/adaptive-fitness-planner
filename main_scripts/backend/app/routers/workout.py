"""
FILE LOCATION: backend/app/routers/workout.py

/api/workout/log           — POST: log completed exercises
/api/workout/progress/{email} — GET: weekly completion stats + streak
/api/workout/reminder      — POST: send email reminder
"""
from fastapi import APIRouter, HTTPException
from app.schemas.models import (
    LogWorkoutRequest, LogWorkoutResponse,
    ProgressResponse, ReminderRequest,
)
from app.mcp_server.server import log_workout, get_workout_progress, send_reminder

router = APIRouter(prefix="/api/workout", tags=["Workout"])


@router.post("/log", response_model=LogWorkoutResponse)
async def log(req: LogWorkoutRequest):
    """
    Log completed exercises for a workout session.
    Call this after the user marks exercises as done in the frontend.
    """
    try:
        exercises_str = ", ".join(req.exercise_names)
        result = log_workout(
            user_email=req.user_email,
            plan_id=req.plan_id,
            exercise_names=exercises_str,
            notes=req.notes or "",
            completed_at=req.completed_at or "",
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return LogWorkoutResponse(
            success=result["success"],
            logged_count=result["logged_count"],
            exercises=result["exercises"],
            message=result["message"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{user_email}", response_model=ProgressResponse)
async def progress(user_email: str, days: int = 7):
    """
    Get workout completion stats for the last N days.
    Returns daily breakdown, streak, completion rate, and motivation message.
    """
    try:
        result = get_workout_progress(user_email=user_email, days=days)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("message"))
        return ProgressResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reminder")
async def reminder(req: ReminderRequest):
    """
    Send a workout reminder email to the user.
    Requires SMTP_USER and SMTP_PASS in .env.
    Without SMTP config, returns preview of the email HTML.
    """
    try:
        result = send_reminder(
            to_email=req.to_email,
            user_name=req.user_name,
            plan_day_label=req.plan_day_label,
            exercises=", ".join(req.exercises),
            reminder_type=req.reminder_type,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
