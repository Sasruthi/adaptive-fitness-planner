"""
FILE LOCATION: backend/app/schemas/models.py

All Pydantic request/response models for the FastAPI endpoints.
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Conversation ──────────────────────────────────────────────────────────────

class StartConversationResponse(BaseModel):
    thread_id: str
    stage:     str
    message:   str


class UserMessageRequest(BaseModel):
    thread_id:    str
    user_message: str


class ConversationResponse(BaseModel):
    thread_id:      str
    stage:          str
    message:        str
    slots_complete: bool = False
    profile:        Optional[Dict[str, Any]] = None
    plan:           Optional[Dict[str, Any]] = None
    sql_filters:    Optional[Dict[str, Any]] = None
    rag_filters:    Optional[Dict[str, Any]] = None
    # Semantically matched exercises (with gif_url/image_url) for chat media cards
    exercises:      Optional[List[Dict[str, Any]]] = None


# ── Plan ─────────────────────────────────────────────────────────────────────

class GeneratePlanRequest(BaseModel):
    profile:     Dict[str, Any]
    sql_filters: Optional[Dict[str, Any]] = None
    rag_filters: Optional[Dict[str, Any]] = None
    user_email:  str
    user_name:   str


class SavePlanRequest(BaseModel):
    user_email:  str
    user_name:   str
    goal:        str
    plan:        Dict[str, Any]


class SavePlanResponse(BaseModel):
    success:     bool
    plan_id:     Optional[int] = None
    user_id:     Optional[int] = None
    message:     str
    email_sent:  bool = False
    email_note:  Optional[str] = None


# ── Workout ───────────────────────────────────────────────────────────────────

class LogWorkoutRequest(BaseModel):
    user_email:      str
    plan_id:         int
    exercise_names:  List[str]
    notes:           Optional[str] = ""
    completed_at:    Optional[str] = None


class LogWorkoutResponse(BaseModel):
    success:       bool
    logged_count:  int
    exercises:     List[str]
    message:       str


class ProgressResponse(BaseModel):
    success:          bool
    period_days:      int
    total_sessions:   int
    total_exercises:  int
    completion_rate:  str
    current_streak:   int
    daily_breakdown:  Dict[str, List[str]]
    motivation:       str


# ── Reminder ──────────────────────────────────────────────────────────────────

class ReminderRequest(BaseModel):
    to_email:       str
    user_name:      str
    plan_day_label: str
    exercises:      List[str]
    reminder_type:  str = "workout"


# ── Exercises ─────────────────────────────────────────────────────────────────

class ExerciseSearchRequest(BaseModel):
    body_part:  Optional[str] = None
    equipment:  Optional[str] = None
    difficulty: Optional[str] = None
    query:      Optional[str] = None
    limit:      int = 10


class ExerciseResponse(BaseModel):
    id:             int
    name:           str
    description:    str
    category:       str
    body_part:      str
    target_muscle:  str
    equipment:      str
    difficulty:     str
    rating:         Optional[float]
    gif_url:        str
    image_url:      str = ""
    video_url:      str
    has_media:      bool


# ── Calories ─────────────────────────────────────────────────────────────────

class CalorieRequest(BaseModel):
    age:            int
    sex:            str
    weight_kg:      float
    height_cm:      float
    activity_level: str
    goal:           str


# ── Health check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:     str
    version:    str = "1.0.0"
    modules:    Dict[str, str]
    timestamp:  str
