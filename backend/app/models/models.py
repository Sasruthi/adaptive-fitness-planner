"""
Adaptive Fitness Planner — Database Models
============================================
SQLite for local dev (zero setup). Swap DATABASE_URL to Postgres for prod —
SQLAlchemy models are portable, no code changes needed elsewhere.
"""

from sqlalchemy import (
    Column, Integer, String, Text, Float, ForeignKey, DateTime, Boolean
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


class Exercise(Base):
    """Core exercise catalog — structured data, queried via SQL filters, NOT embedded/RAG."""
    __tablename__ = "exercises"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    category = Column(String(100), index=True)        # Strength, Cardio, Stretching, etc.
    body_part = Column(String(100), index=True)        # primary body part (taxonomy-controlled)
    target_muscle = Column(String(100), index=True)    # specific muscle if known
    equipment = Column(String(100), index=True)
    difficulty_level = Column(String(50), index=True)  # Beginner / Intermediate / Expert
    rating = Column(Float, nullable=True)
    gif_url = Column(String(500), nullable=True)
    image_url = Column(String(500), nullable=True)   # thumbnail (hasaneyldrm images/)
    video_url = Column(String(500), nullable=True)
    has_media = Column(Boolean, default=False)
    match_confidence = Column(Float, nullable=True)     # similarity score from merge step, for audit
    source = Column(String(100))                        # which dataset(s) this came from

    def __repr__(self):
        return f"<Exercise {self.id} {self.name}>"


class Taxonomy(Base):
    """Controlled vocabulary for body parts / equipment / muscles — used by conversational
    intake to present valid options and to resolve fuzzy user phrases to canonical terms."""
    __tablename__ = "taxonomy"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(50), index=True)   # 'body_part' | 'equipment' | 'muscle'
    name = Column(String(150), index=True)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    name = Column(String(150))
    age = Column(Integer)
    sex = Column(String(20))
    height_cm = Column(Float)
    weight_kg = Column(Float)
    activity_level = Column(String(50))   # sedentary/lightly/moderate/active/very_active
    health_flags = Column(Text)           # JSON-encoded list, e.g. ["high_bp","knee_injury"]
    created_at = Column(DateTime, default=datetime.utcnow)

    plans = relationship("Plan", back_populates="user")
    logs = relationship("WorkoutLog", back_populates="user")


class Plan(Base):
    """A generated workout plan — the directly-usable output of the conversational flow."""
    __tablename__ = "plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    goal = Column(String(150))            # e.g. "reduce arm fat", "strengthen calves"
    constraints_json = Column(Text)       # snapshot of slot-filled constraints at creation time
    plan_json = Column(Text)              # the structured day-by-day exercise plan
    rationale_json = Column(Text)         # per-item "why", citing RAG sources where used
    created_at = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)

    user = relationship("User", back_populates="plans")


class WorkoutLog(Base):
    """Tracks completion — feeds the progress/revisit module and reminder logic."""
    __tablename__ = "workout_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_id = Column(Integer, ForeignKey("plans.id"))
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    completed_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    user = relationship("User", back_populates="logs")
