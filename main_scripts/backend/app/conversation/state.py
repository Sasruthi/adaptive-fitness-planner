"""
FILE LOCATION: backend/app/conversation/state.py

Conversation state schema for the Adaptive Fitness Planner intake flow.
Handles out-of-vocabulary inputs gracefully — no silent defaults.
"""
from typing import Optional, List, TypedDict

# ── Controlled taxonomies ─────────────────────────────────────────────────────

GOALS = [
    "lose_fat", "build_muscle", "improve_strength", "improve_flexibility",
    "improve_endurance", "general_fitness", "rehabilitation", "stress_relief",
]

BODY_PARTS = [
    "neck", "shoulders", "chest", "back", "upper arms",
    "lower arms", "waist", "upper legs", "lower legs", "cardio",
]

EQUIPMENT_OPTIONS = [
    "body only", "dumbbell", "barbell", "kettlebells", "bands",
    "cable", "machine", "exercise ball", "foam roll", "none",
]

FITNESS_LEVELS = ["beginner", "intermediate", "expert"]

ACTIVITY_LEVELS = [
    "sedentary", "lightly_active", "moderately_active", "very_active",
]

# Recognised health flags — drive hard SQL safety filters
KNOWN_HEALTH_FLAGS = [
    "high_bp", "low_bp", "diabetes", "knee_injury", "back_injury",
    "shoulder_injury", "wrist_injury", "ankle_injury", "heart_condition",
    "asthma", "osteoporosis", "acidity", "pregnancy", "obesity", "none",
]

# Equipment synonyms — maps user vocab to controlled terms
EQUIPMENT_SYNONYMS = {
    "trx": "bands", "resistance band": "bands", "resistance bands": "bands",
    "elastic band": "bands", "theraband": "bands",
    "foam roller": "foam roll", "roller": "foam roll",
    "swiss ball": "exercise ball", "stability ball": "exercise ball",
    "ez bar": "barbell", "ez curl bar": "barbell",
    "pull up bar": "body only", "chin up bar": "body only",
    "bodyweight": "body only", "no equipment": "none",
    "gym": "dumbbell", "full gym": "dumbbell",
    "mat": "body only", "yoga mat": "body only",
}


# ── ConversationState TypedDict ───────────────────────────────────────────────

class ConversationState(TypedDict, total=False):
    # Conversation control
    stage:              str
    chat_history:       List[dict]      # [{"role":..,"content":..}]
    attempts:           dict
    profile_confirmed:  bool
    error:              Optional[str]

    # Canonical slot values (controlled vocab)
    goal:                   Optional[str]   # from GOALS
    target_body_parts:      List[str]       # subset of BODY_PARTS
    age:                    Optional[int]
    sex:                    Optional[str]   # male|female|prefer_not_to_say
    height_cm:              Optional[float]
    weight_kg:              Optional[float]
    activity_level:         Optional[str]
    health_flags:           List[str]       # subset of KNOWN_HEALTH_FLAGS
    available_equipment:    List[str]       # subset of EQUIPMENT_OPTIONS
    fitness_level:          Optional[str]   # from FITNESS_LEVELS
    time_per_day_minutes:   Optional[int]   # 15|30|45|60

    # OUT-OF-VOCAB captures — stored for RAG enrichment, never silently dropped
    raw_goal_phrase:        Optional[str]   # original user phrasing, e.g. "marathon training"
    raw_body_parts:         List[str]       # original phrases, e.g. ["deltoids", "love handles"]
    custom_health_notes:    List[str]       # unrecognised flags, e.g. ["spondylitis", "PCOD"]
    raw_equipment:          List[str]       # unrecognised equipment for context

    # Post-slot-fill
    sql_filters:    dict
    rag_filters:    dict
    plan:           Optional[dict]


def initial_state() -> ConversationState:
    return ConversationState(
        stage="greeting",
        chat_history=[], attempts={},
        profile_confirmed=False, error=None,
        goal=None, target_body_parts=[], age=None, sex=None,
        height_cm=None, weight_kg=None, activity_level=None,
        health_flags=[], available_equipment=[], fitness_level=None,
        time_per_day_minutes=None,
        # Out-of-vocab captures
        raw_goal_phrase=None, raw_body_parts=[],
        custom_health_notes=[], raw_equipment=[],
        sql_filters={}, rag_filters={}, plan=None,
    )


def slots_complete(state: ConversationState) -> bool:
    required = [
        "goal", "target_body_parts", "age", "sex",
        "health_flags", "available_equipment", "fitness_level",
        "time_per_day_minutes",
    ]
    return all(state.get(s) not in (None, [], "") for s in required)


def resolve_equipment(raw: str) -> Optional[str]:
    """Map synonyms to controlled vocab. Returns None if truly unrecognised."""
    cleaned = raw.lower().strip()
    if cleaned in EQUIPMENT_OPTIONS:
        return cleaned
    if cleaned in EQUIPMENT_SYNONYMS:
        return EQUIPMENT_SYNONYMS[cleaned]
    # Fuzzy: if any known term is a substring
    for known in EQUIPMENT_OPTIONS:
        if known in cleaned or cleaned in known:
            return known
    return None   # genuinely unrecognised — caller must handle


# Injury → body parts that must be excluded from workouts
INJURY_BODY_PART_EXCLUSIONS = {
    "knee_injury": ["lower legs", "upper legs"],
    "ankle_injury": ["lower legs"],
    "shoulder_injury": ["shoulders", "upper arms", "lower arms"],
    "wrist_injury": ["lower arms"],
    "back_injury": ["back"],
}


def injury_excluded_body_parts(health_flags: List[str]) -> List[str]:
    exclusions: List[str] = []
    for flag in health_flags or []:
        exclusions.extend(INJURY_BODY_PART_EXCLUSIONS.get(flag, []))
    return list(dict.fromkeys(exclusions))


def build_sql_filters(state: ConversationState) -> dict:
    filters = {}

    if state.get("target_body_parts"):
        filters["body_part__in"] = state["target_body_parts"]

    equip = state.get("available_equipment", [])
    if "none" in [e.lower() for e in equip] or not equip:
        # No equipment — only bodyweight exercises
        filters["equipment__in"] = [
            "body only", "Body Only", "bodyweight", "Bodyweight",
            "None", "none", "Body Weight", "body weight"
        ]
    else:
        # Include their equipment + bodyweight always
        filters["equipment__in"] = equip + [
            "body only", "Body Only", "bodyweight", "Bodyweight"
        ]

    if state.get("fitness_level"):
        filters["difficulty_level"] = state["fitness_level"].title()

    # Hard body-part exclusions from health flags (single source of truth)
    exclusions = injury_excluded_body_parts(state.get("health_flags", []))
    if exclusions:
        filters["exclude_body_parts"] = exclusions

    return filters


def build_rag_filters(state: ConversationState) -> dict:
    """
    RAG filters are intentionally broad — custom_health_notes and
    raw phrases are passed as free-text queries to Qdrant, not as
    hard filters, so genuinely novel conditions still retrieve context.
    """
    filters = {"trust_tier__in": ["Tier 1", "Tier 2"]}

    flags = state.get("health_flags", [])
    has_critical = any(f in flags for f in
                       ["high_bp", "heart_condition", "diabetes", "obesity", "pregnancy"])

    if has_critical or state.get("custom_health_notes"):
        filters["content_type__in"] = ["safety_medical", "guideline", "exercise", "nutrition"]
    else:
        filters["content_type__in"] = ["nutrition", "guideline", "exercise", "lifestyle"]

    goal = state.get("goal", "")
    if goal in ("lose_fat", "general_fitness"):
        filters["category__in"] = ["nutrition", "exercise"]
    elif goal in ("build_muscle", "improve_strength"):
        filters["category__in"] = ["exercise", "nutrition"]
    elif goal == "improve_flexibility":
        filters["category__in"] = ["exercise", "lifestyle"]
    else:
        filters["category__in"] = ["nutrition", "exercise", "lifestyle"]

    return filters