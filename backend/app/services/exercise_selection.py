"""
FILE LOCATION: backend/app/services/exercise_selection.py

Semantic-first exercise selection for PLAN GENERATION.
=========================================================
Replaces two problems in the old plan_generator.py:

  1. ROUND-ROBIN SLICING: the old select_and_schedule() sliced a flat
     retrieved pool by index per day (day1=ex[0:5], day2=ex[5:10]...),
     so every day got different exercises with no program structure.
     Fix: resolve ONE fixed exercise set per "day type" (a specific
     body part, or "Full Body"), ONCE. That same set is reused every
     time that day type recurs in the week — this is what makes it a
     real program instead of a shuffled grab-bag.

  2. STRING-MATCH RETRIEVAL: exercise_retrieval.py matches body_part
     via `.ilike('%term%')` against however the catalog happened to
     tag a row. Canonicalized profile values (e.g. "lower legs") can
     still miss real rows tagged inconsistently (e.g. "calves").
     Fix: match by meaning, using the existing embedding index in
     exercise_rag.py, with equipment/health exclusions applied as
     hard post-filters (semantic finds candidates, rules prune them —
     never the other way around).

No goal-based "split override" table. The split is derived directly
from target_body_parts:
  - target_body_parts expands to every region (incl. "full body" alias,
    already handled upstream in profile_store.py)  -> one Full Body day type
  - otherwise                                        -> the split IS the
                                                         specific parts chosen
Goal only affects sets/reps/rest elsewhere; it never decides which
muscles get trained.
"""

from typing import Dict, List, Optional

from app.services.exercise_rag import retrieve_exercise_semantic
from app.services.exercise_retrieval import get_exercises  # fallback only


# Canonical taxonomy -> natural-language query. Embeddings need real
# phrases to match meaningfully against exercise descriptions; the bare
# taxonomy label ("waist") is too generic to encode useful direction.
BODY_PART_QUERY_HINTS: Dict[str, str] = {
    "neck":       "neck mobility and neck strengthening exercises",
    "shoulders":  "shoulder press, lateral raise, and rotator cuff exercises",
    "chest":      "chest press and chest fly exercises for pectoral muscles",
    "back":       "back rows, pulldowns, and lat exercises for back strength",
    "upper arms": "biceps curl and triceps extension exercises for upper arm strength",
    "lower arms": "forearm and wrist strengthening exercises",
    "waist":      "core, abs, and oblique exercises for waist and midsection",
    "upper legs": "quadriceps and hamstring exercises for thigh strength",
    "lower legs": "calf raise and lower leg strengthening exercises",
    "cardio":     "cardiovascular conditioning and endurance exercises",
}

ALL_BODY_PARTS = set(BODY_PART_QUERY_HINTS.keys())

FITNESS_LEVEL_HINT = {
    "beginner":     "beginner friendly, simple technique, push-ups squats planks walking — no gymnastics",
    "intermediate": "intermediate level",
    "expert":       "advanced, high intensity",
}


def resolve_split(target_body_parts: List[str]) -> Dict:
    """
    Derive the split directly from target_body_parts. No override table.
    """
    parts = [p for p in (target_body_parts or []) if p in ALL_BODY_PARTS]
    if not parts or set(parts) == ALL_BODY_PARTS:
        return {"split_type": "full_body", "day_types": ["Full Body"]}
    return {"split_type": "part_focus", "day_types": parts}


def _build_day_type_query(
    day_type: str,
    fitness_level: str,
    equipment_note: str,
    *,
    yoga_mode: bool = False,
) -> str:
    if yoga_mode:
        if day_type == "Full Body":
            hint = (
                "yoga asanas stretches mobility: sun salutation, downward dog, "
                "cobra, cat-cow, child's pose, warrior, bridge, forward fold — "
                "bodyweight yoga practice"
            )
        else:
            base = BODY_PART_QUERY_HINTS.get(day_type, day_type)
            hint = f"yoga asana and stretch variations for {base}"
    elif day_type == "Full Body":
        if (fitness_level or "").lower() == "beginner":
            hint = (
                "beginner full body: push-ups, bodyweight squats, glute bridges, "
                "plank, walking or marching — easy floor moves only"
            )
        else:
            hint = "full body toning and conditioning covering all major muscle groups"
    else:
        hint = BODY_PART_QUERY_HINTS.get(day_type, day_type)
    level_hint = FITNESS_LEVEL_HINT.get(fitness_level, "")
    return f"{hint}, {level_hint}, {equipment_note}".strip(", ")


def resolve_exercise_sets(
    day_types: List[str],
    *,
    fitness_level: str,
    equipment: Optional[List[str]],
    exclude_body_parts: Optional[List[str]],
    exercises_per_session: int,
    yoga_mode: bool = False,
) -> Dict[str, List[Dict]]:
    """
    Resolve one fixed, semantically-matched exercise set per day type.
    Returns {day_type: [exercise_dict, ...]} — callers reuse the same
    list every time that day type appears in the week; never re-slice.
    """
    from app.services.exercise_rag import _is_advanced_skill

    has_no_equip = (
        not equipment
        or all(
            (e or "").lower().strip() in {"none", "body only", "bodyweight", "body weight", ""}
            for e in equipment
        )
        or yoga_mode
    )
    equipment_note = (
        "bodyweight yoga mat only, no gym machines" if yoga_mode or has_no_equip
        else f"using {', '.join(equipment)}"
    )
    level = (fitness_level or "beginner").lower()
    equip_filter = ["body only", "none"] if (yoga_mode or has_no_equip) else (equipment or ["body only", "none"])

    resolved: Dict[str, List[Dict]] = {}
    for day_type in day_types:
        query = _build_day_type_query(day_type, level, equipment_note, yoga_mode=yoga_mode)
        wanted_parts = None if day_type == "Full Body" else [day_type]

        hits = retrieve_exercise_semantic(
            query,
            top_k=max(exercises_per_session * 4, 12),
            equipment=equip_filter,
            body_parts=wanted_parts,
            prefer_media=True,
            prefer_difficulty=level,
        )

        if exclude_body_parts:
            excl = {e.lower() for e in exclude_body_parts}
            hits = [h for h in hits if (h.get("body_part") or "").lower() not in excl]

        if level == "beginner":
            hits = [
                h for h in hits
                if not _is_advanced_skill(h.get("name", ""), h.get("description", ""))
                and "expert" not in (h.get("difficulty") or "").lower()
            ]

        # Prefer rows that have GIF/image so the Plan UI is usable
        with_media = [h for h in hits if h.get("has_media") or h.get("gif_url") or h.get("image_url")]
        if len(with_media) >= max(3, exercises_per_session // 2):
            # Keep media first; fill remainder from non-media only if needed
            rest = [h for h in hits if h not in with_media]
            hits = with_media + rest

        if not hits:
            sql_filters: Dict = {}
            if equip_filter:
                sql_filters["equipment__in"] = equip_filter
            if day_type != "Full Body":
                sql_filters["body_part__in"] = [day_type]
            if exclude_body_parts:
                sql_filters["exclude_body_parts"] = exclude_body_parts
            if level == "beginner":
                sql_filters["difficulty_level"] = "Beginner"
            hits = get_exercises(sql_filters, limit=exercises_per_session * 3)
            if level == "beginner":
                hits = [
                    h for h in hits
                    if not _is_advanced_skill(h.get("name", ""), h.get("description", ""))
                ]
            for h in hits:
                h["score"] = None

        resolved[day_type] = hits[:exercises_per_session]

    return resolved
