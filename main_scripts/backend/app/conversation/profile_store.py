"""
FILE LOCATION: backend/app/conversation/profile_store.py

Thread-scoped state for the unified conversational agent.

Replaces the old intake_graph.py regex/keyword slot-filling stack.
The LLM (via the `update_profile` tool in agent.py) decides what it has
learned from the user; this module only validates + stores it and answers
"what's still missing" / "is it safe to generate a plan yet".

Kept intentionally dumb: no NLP here. All extraction judgement lives in
the agent's tool-calling, which is where it belongs for an agentic system.
"""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

GOALS = ["lose_fat", "build_muscle", "improve_strength", "improve_flexibility",
         "improve_endurance", "general_fitness", "rehabilitation", "stress_relief"]
BODY_PARTS = ["neck", "shoulders", "chest", "back", "upper arms", "lower arms",
              "waist", "upper legs", "lower legs", "cardio"]
EQUIPMENT_OPTIONS = ["body only", "dumbbell", "barbell", "kettlebells", "bands",
                     "cable", "machine", "exercise ball", "foam roll", "none"]
FITNESS_LEVELS = ["beginner", "intermediate", "expert"]
KNOWN_FLAGS = ["high_bp", "low_bp", "diabetes", "knee_injury", "back_injury",
               "shoulder_injury", "wrist_injury", "ankle_injury", "heart_condition",
               "asthma", "osteoporosis", "acidity", "pregnancy", "obesity", "none"]

# THE FIX for the "full body" deadlock: BODY_PARTS only lists specific
# regions (chest, back, upper legs, ...) — it never had a "full body"
# value, so every real user who says "full body" (an extremely common
# phrasing) got silently dropped by the old merge(), forever. Now it
# expands to every region instead of being rejected.
FULL_BODY_ALIASES = {"full body", "full_body", "whole body", "whole_body",
                     "total body", "total_body", "entire body", "everything", "all"}

# Natural language → canonical body_part. Users/agents say "core"; catalog uses "waist".
BODY_PART_ALIASES = {
    "core": "waist",
    "abs": "waist",
    "ab": "waist",
    "abdominals": "waist",
    "midsection": "waist",
    "biceps": "upper arms",
    "triceps": "upper arms",
    "arms": "upper arms",
    "legs": "upper legs",
    "thighs": "upper legs",
    "calves": "lower legs",
    "glutes": "upper legs",
    "shoulders": "shoulders",
}

# Equipment phrasing → catalog values. "without any equipment" must land on body only.
EQUIPMENT_ALIASES = {
    "no equipment": "body only",
    "without equipment": "body only",
    "without any equipment": "body only",
    "no equipments": "body only",
    "without equipments": "body only",
    "bodyweight": "body only",
    "body weight": "body only",
    "body-only": "body only",
    "bodyonly": "body only",
    "home": "body only",
    "none": "none",
}


def _strip_noise(s: str) -> str:
    """Lowercase, drop parentheticals/punctuation so 'full body (expands...)' → 'full body'."""
    s = (s or "").strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _expand_body_part_aliases(values: List[str]) -> List[str]:
    out = []
    for v in values:
        v_norm = _strip_noise(v)
        if not v_norm:
            continue
        # Exact full-body token, or phrase that clearly means full body
        if v_norm in FULL_BODY_ALIASES or any(
            alias == v_norm or v_norm.startswith(alias + " ") or f" {alias} " in f" {v_norm} "
            for alias in ("full body", "whole body", "total body", "entire body")
        ):
            out.extend(BODY_PARTS)
            continue
        mapped = BODY_PART_ALIASES.get(v_norm, v_norm)
        out.append(mapped)
    return list(dict.fromkeys(out))


def _normalize_equipment(values: List[str]) -> List[str]:
    out = []
    for v in values:
        v_norm = _strip_noise(v)
        if not v_norm:
            continue
        if v_norm in EQUIPMENT_ALIASES:
            out.append(EQUIPMENT_ALIASES[v_norm])
            continue
        # Substring catch for "without any equipment please"
        if "no equipment" in v_norm or ("without" in v_norm and "equipment" in v_norm):
            out.append("body only")
            continue
        if "bodyweight" in v_norm.replace(" ", "") or v_norm in ("body only", "body weight"):
            out.append("body only")
            continue
        out.append(v_norm)
    return list(dict.fromkeys(out))

REQUIRED_FIELDS = [
    "goal", "target_body_parts", "age", "gender",
    "health_flags",  # safety-critical — see note below
    "available_equipment", "fitness_level", "time_per_day_minutes",
]

# Fields only needed for workout plans — skipped when plan_mode == "diet_only"
EXERCISE_ONLY_FIELDS = {
    "target_body_parts", "available_equipment", "fitness_level", "time_per_day_minutes",
}

PLAN_MODES = {"full", "diet_only"}


class Profile(BaseModel):
    goal: Optional[str] = None
    target_body_parts: List[str] = Field(default_factory=list)
    age: Optional[int] = None
    gender: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    health_flags: List[str] = Field(default_factory=list)      # ["none"] means "confirmed, no flags"
    custom_health_notes: List[str] = Field(default_factory=list)
    available_equipment: List[str] = Field(default_factory=list)
    fitness_level: Optional[str] = None
    time_per_day_minutes: Optional[int] = None
    # "full" = workout + diet; "diet_only" = meals/nutrition without exercise slots
    plan_mode: str = "full"

    @field_validator("goal")
    @classmethod
    def _valid_goal(cls, v):
        if v is not None and v not in GOALS:
            return None
        return v

    @field_validator("fitness_level")
    @classmethod
    def _valid_level(cls, v):
        if v is not None and v not in FITNESS_LEVELS:
            return None
        return v

    @field_validator("plan_mode")
    @classmethod
    def _valid_plan_mode(cls, v):
        if v is None:
            return "full"
        v = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "diet": "diet_only",
            "dietonly": "diet_only",
            "nutrition": "diet_only",
            "nutrition_only": "diet_only",
            "meals_only": "diet_only",
            "workout": "full",
            "exercise": "full",
            "both": "full",
        }
        v = aliases.get(v, v)
        return v if v in PLAN_MODES else "full"

    def merge(self, **updates) -> Dict[str, list]:
        """
        Apply only non-empty updates.
        Returns {"changed": [...], "rejected": {field: {"got": ..., "valid_options": [...]}}}
        — the caller (update_profile tool) surfaces `rejected` back to the LLM
        so it can self-correct instead of silently retrying the same bad value
        forever (this was the exact cause of the "full body" deadlock: the
        old version dropped invalid values with zero explanation).
        """
        changed = []
        rejected = {}
        for key, value in updates.items():
            # Allow explicit 0 for time_per_day_minutes (diet-only / no exercise)
            if key != "time_per_day_minutes" and value in (None, [], ""):
                continue
            if key == "time_per_day_minutes" and value in (None, "", []):
                continue

            if key == "plan_mode":
                raw_mode = value
                value = Profile._valid_plan_mode(value)
                if str(raw_mode).strip().lower().replace("-", "_").replace(" ", "_") not in (
                    "full", "diet_only", "diet", "dietonly", "nutrition", "nutrition_only",
                    "meals_only", "workout", "exercise", "both",
                ) and value == "full" and str(raw_mode).lower() not in ("full",):
                    rejected["plan_mode"] = {
                        "got": raw_mode,
                        "valid_options": list(PLAN_MODES),
                    }
                    continue

            if key == "goal":
                if value not in GOALS:
                    rejected["goal"] = {"got": value, "valid_options": GOALS}
                    continue

            if key == "fitness_level":
                if value not in FITNESS_LEVELS:
                    rejected["fitness_level"] = {
                        "got": value,
                        "valid_options": FITNESS_LEVELS,
                    }
                    continue

            if key == "target_body_parts":
                normalized = _expand_body_part_aliases(value)
                valid = [p for p in normalized if p in BODY_PARTS]
                invalid = [p for p in normalized if p not in BODY_PARTS]
                if invalid:
                    rejected["target_body_parts"] = {
                        "got": invalid,
                        "valid_options": BODY_PARTS + ["full body"],
                        "hint": "Map user words yourself: core/abs→waist, full body→all regions. Do not ask the user to type the exact token.",
                    }
                if not valid:
                    continue
                value = valid

            if key == "available_equipment":
                normalized = _normalize_equipment(value)
                valid = [e for e in normalized if e in EQUIPMENT_OPTIONS]
                invalid = [e for e in normalized if e not in EQUIPMENT_OPTIONS]
                if invalid:
                    rejected["available_equipment"] = {
                        "got": invalid,
                        "valid_options": EQUIPMENT_OPTIONS,
                        "hint": "Map 'no/without equipment' or 'bodyweight' → 'body only'. Do not re-ask for the exact phrase.",
                    }
                if not valid:
                    continue
                value = valid

            if key == "health_flags":
                valid = [f for f in value if f in KNOWN_FLAGS]
                invalid = [f for f in value if f not in KNOWN_FLAGS]
                if invalid:
                    rejected["health_flags"] = {"got": invalid, "valid_options": KNOWN_FLAGS}
                if not valid:
                    continue
                # Explicit "none" clears prior flags (user confirmed healthy)
                if "none" in valid and len(valid) == 1:
                    value = ["none"]
                else:
                    existing = [f for f in self.health_flags if f != "none"]
                    merged = list(dict.fromkeys(existing + [f for f in valid if f != "none"]))
                    value = merged if merged else (["none"] if "none" in valid else [])

            if key == "custom_health_notes":
                value = list(dict.fromkeys(self.custom_health_notes + value))

            if key == "age":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    rejected["age"] = {"got": value, "valid_options": ["10–100"]}
                    continue
                if value < 10 or value > 100:
                    rejected["age"] = {"got": value, "valid_options": ["10–100"]}
                    continue

            if key == "gender":
                g = str(value).strip().lower()
                if g not in ("male", "female"):
                    rejected["gender"] = {"got": value, "valid_options": ["male", "female"]}
                    continue
                value = g

            if key == "height_cm":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    rejected["height_cm"] = {"got": value, "valid_options": ["100–250 cm"]}
                    continue
                if value < 100 or value > 250:
                    rejected["height_cm"] = {"got": value, "valid_options": ["100–250 cm"]}
                    continue

            if key == "weight_kg":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    rejected["weight_kg"] = {"got": value, "valid_options": ["25–300 kg"]}
                    continue
                if value < 25 or value > 300:
                    rejected["weight_kg"] = {"got": value, "valid_options": ["25–300 kg"]}
                    continue

            if key == "time_per_day_minutes":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    rejected["time_per_day_minutes"] = {
                        "got": value,
                        "valid_options": ["0 (diet-only / no exercise)", "15", "30", "45", "60"],
                    }
                    continue
                if value < 0 or value > 180:
                    rejected["time_per_day_minutes"] = {
                        "got": value,
                        "valid_options": ["0–180 minutes"],
                    }
                    continue

            setattr(self, key, value)
            changed.append(key)
        return {"changed": changed, "rejected": rejected}

    def missing_fields(self) -> List[str]:
        missing = []
        if not self.goal:
            missing.append("goal")
        if not self.age:
            missing.append("age")
        if not self.gender:
            missing.append("gender")
        # Required for Mifflin–St Jeor BMR / honest calorie targets
        if self.weight_kg is None:
            missing.append("weight_kg")
        if self.height_cm is None:
            missing.append("height_cm")
        if not self.health_flags:
            # custom_health_notes alone do NOT satisfy the safety gate
            missing.append("health_flags")

        # Diet-only: do NOT require exercise slots (body parts, equipment, level, time)
        if self.plan_mode == "diet_only":
            return missing

        if not self.target_body_parts:
            missing.append("target_body_parts")
        if not self.available_equipment:
            missing.append("available_equipment")
        if not self.fitness_level:
            missing.append("fitness_level")
        # 0 minutes is valid (explicit no-exercise); only None is missing
        if self.time_per_day_minutes is None:
            missing.append("time_per_day_minutes")
        return missing

    def is_safe_to_plan(self) -> bool:
        """
        Hard, code-level safety gate — NOT left to the LLM's judgement.
        Requires an explicit health_flags answer (real flags or ["none"]).
        custom_health_notes enrich RAG but do not skip the flag confirmation.
        """
        has_health_answer = bool(self.health_flags)
        return not self.missing_fields() and has_health_answer

    def to_dict(self) -> Dict:
        d = self.model_dump()
        d["sex"] = d.get("gender")
        return d


class SessionStore:
    """In-memory per-process store keyed by thread_id.

    NOTE for the hosting/deployment pass: this dict is lost on every
    redeploy/restart and won't work across multiple worker processes.
    Fine for a single-instance free-tier deployment; swap for a Redis-
    or Postgres-backed store (keyed by thread_id) before scaling out.
    """
    def __init__(self):
        self._profiles: Dict[str, Profile] = {}
        self._plans: Dict[str, dict] = {}
        self._plan_confirmed: Dict[str, bool] = {}
        self._exercises: Dict[str, list] = {}

    def get_profile(self, thread_id: str) -> Profile:
        if thread_id not in self._profiles:
            self._profiles[thread_id] = Profile()
        return self._profiles[thread_id]

    def set_plan(self, thread_id: str, plan: dict):
        self._plans[thread_id] = plan

    def get_plan(self, thread_id: str) -> Optional[dict]:
        return self._plans.get(thread_id)

    def set_exercises(self, thread_id: str, exercises: list):
        self._exercises[thread_id] = exercises or []

    def get_exercises(self, thread_id: str) -> list:
        return self._exercises.get(thread_id) or []

    def clear_exercises(self, thread_id: str):
        self._exercises.pop(thread_id, None)


session_store = SessionStore()
