"""
Adaptive Fitness Planner — Plan Generation Engine (Module 4)
=============================================================
Combines:
  1. Exercise retrieval  (SQL  → matched exercises with gif_urls)
  2. Guideline retrieval (RAG  → relevant India-first passages with citations)
  3. Plan synthesis      (GPT-4o → structured day-by-day plan as JSON)

Output contract (what the frontend receives):
{
  "plan_id": "uuid",
  "generated_at": "ISO datetime",
  "profile_summary": {...},
  "week_plan": [
    {
      "day": 1,
      "label": "Monday — Upper Arms",
      "focus": "upper arms",
      "exercises": [
        {
          "name": "...",
          "sets": 3,
          "reps": "12-15",
          "rest_seconds": 45,
          "gif_url": "...",
          "target_muscle": "...",
          "instructions": "...",
          "modification": "..."  // for health flags
        }
      ],
      "duration_minutes": 30,
      "notes": "..."
    }
  ],
  "diet_plan": {
    "daily_calories_estimate": 1600,
    "meals": [
      {"meal": "breakfast", "suggestions": [...], "notes": "..."},
      ...
    ],
    "india_specific_tips": [...],
    "foods_to_avoid": [...]
  },
  "safety_notes": [...],  // grounded in RAG, specific to health flags
  "citations": [
    {"source": "NIN Dietary Guidelines 2024", "page": 10, "note": "..."}
  ]
}
"""

import os, json, uuid, re
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage

from app.llm import get_llm
from app.services.rag_retrieval import retrieve_multi_query
from app.services.exercise_selection import resolve_split, resolve_exercise_sets
from app.services.anthropometrics import resolve_height_weight
from app.services.nutrition_lookup import get_verified_macros
from app.mcp_server.server import calculate_calories


# ── Query builder for RAG ─────────────────────────────────────────────────────
def build_rag_queries(profile: Dict) -> List[str]:
    """Build multiple targeted semantic queries based on user profile."""
    goal    = profile.get("goal", "").replace("_", " ")
    parts   = ", ".join(profile.get("target_body_parts", []))
    flags   = profile.get("health_flags", ["none"])
    notes   = profile.get("custom_health_notes", []) or []
    sex     = profile.get("sex", "") or profile.get("gender", "")
    age     = profile.get("age", "")
    mode = profile.get("plan_mode") or "full"
    diet_only = mode == "diet_only"
    yoga_only = mode == "yoga_only"

    queries = [
        f"{goal} diet and nutrition for {sex} age {age} India",
        f"healthy Indian meal pattern {goal}",
    ]
    if yoga_only:
        queries = [
            "Common Yoga Protocol asanas pranayama daily practice sequence",
            f"yoga practice guidelines for {sex} age {age} India",
            "yoga breathing techniques benefits and precautions",
            f"healthy Indian meal pattern for yoga lifestyle {goal}",
        ]
        if parts:
            queries.append(f"yoga stretches and asanas for {parts}")
    elif not diet_only:
        queries.insert(0, f"{goal} exercise recommendations for {sex} age {age}")
        if parts:
            queries.append(f"physical activity guidelines for {parts}")

    # Add safety queries for health flags / notes
    flag_blob = " ".join(list(flags) + list(notes)).lower()
    if "high_bp" in flags or "heart_condition" in flags or "cholesterol" in flag_blob:
        queries.append("diet recommendations high cholesterol heart health India ICMR")
        queries.append("foods to lower cholesterol fibre saturated fat India")
    if "diabetes" in flags:
        queries.append("exercise and diet for diabetes blood sugar India")
        queries.append("carbohydrate intake diabetes Indian diet")
    if "knee_injury" in flags or "back_injury" in flags:
        if not diet_only:
            queries.append(
                "yoga modifications joint injury"
                if yoga_only else
                "exercise modifications joint injury rehabilitation"
            )
    if "obesity" in flags:
        queries.append("physical activity obesity weight management India guidelines")

    return queries


# ── Exercise selection and scheduling ─────────────────────────────────────────
def _resolve_workout_days(profile: Dict) -> int:
    """
    Routine-first: 6 workout days + 1 recovery day.
    Users repeatedly rejected sparse 3–5 day weeks with multiple rest days.
    """
    if profile.get("workout_days_per_week"):
        return max(5, min(6, int(profile["workout_days_per_week"])))
    return 6


def _build_workout_rest_pattern(workout_days: int) -> List[str]:
    """Place `workout_days` sessions across 7 calendar days, rest the rest.

    Prefer spreading workouts (e.g. 5 → W R W R W W W) rather than
    stacking the week front-loaded then dumping four rest days at the end.
    """
    workout_days = max(1, min(7, int(workout_days)))
    if workout_days >= 7:
        return ["workout"] * 7
    # Evenly space workout indices across 0..6
    schedule = ["rest"] * 7
    for i in range(workout_days):
        idx = round(i * 6 / max(workout_days - 1, 1)) if workout_days > 1 else 0
        # resolve collisions by walking forward
        while schedule[idx % 7] == "workout":
            idx += 1
        schedule[idx % 7] = "workout"
    # Guarantee exact count
    while schedule.count("workout") < workout_days:
        for i in range(7):
            if schedule[i] == "rest":
                schedule[i] = "workout"
                break
    while schedule.count("workout") > workout_days:
        for i in range(6, -1, -1):
            if schedule[i] == "workout":
                schedule[i] = "rest"
                break
    return schedule


def select_and_schedule(profile: Dict) -> List[Dict]:
    """
    Organise a 7-day week plan built on a real split:
      - split (day types) is derived directly from target_body_parts —
        full body if it expands to every region, otherwise the split IS
        the specific parts chosen (see exercise_selection.resolve_split)
      - each day type's exercise set is resolved ONCE, semantically, and
        REUSED every time that day type recurs — never re-sliced from a
        shared pool, which is what caused a different workout every day
    Goal only tunes sets/reps/rest/intensity, never which muscles train.
    """
    time_min    = profile.get("time_per_day_minutes", 30)
    goal        = profile.get("goal", "general_fitness")
    fitness_lvl = profile.get("fitness_level", "beginner")
    equipment   = profile.get("available_equipment", [])
    health_flags= profile.get("health_flags", [])
    yoga_only   = (profile.get("plan_mode") or "full") == "yoga_only"
    if yoga_only:
        equipment = ["body only"]

    from app.conversation.state import injury_excluded_body_parts

    # Prefer sql_filters exclusions (caller-built); else derive from health flags
    sql_excl = (profile.get("_sql_exclude_body_parts")
                or (profile.get("_sql_filters") or {}).get("exclude_body_parts")
                or [])
    exclude_body_parts = list(sql_excl) if sql_excl else injury_excluded_body_parts(health_flags)

    exercises_per_session = {15: 3, 30: 5, 45: 7, 60: 9}.get(time_min, 5)

    if yoga_only or goal == "improve_flexibility":
        sets, reps, rest = 2, "30–45s hold / 5 breaths", 15
    elif goal in ("build_muscle", "improve_strength"):
        sets, reps, rest = 4, "8-10", 90
    elif goal == "lose_fat":
        sets, reps, rest = 3, "15-20", 30
    else:
        sets, reps, rest = 3, "12-15", 45
    if fitness_lvl == "beginner":
        sets = max(2, sets - 1)
        rest = int(rest * 1.5)

    workout_days = _resolve_workout_days(profile)

    # ── Derive the split directly from target_body_parts ─────────────────
    split = resolve_split(profile.get("target_body_parts", []))
    day_types = split["day_types"]

    # ── Resolve ONE fixed exercise set per day type, semantically ────────
    resolved_sets = resolve_exercise_sets(
        day_types,
        fitness_level=fitness_lvl,
        equipment=equipment,
        exclude_body_parts=exclude_body_parts,
        exercises_per_session=exercises_per_session,
        yoga_mode=yoga_only,
    )

    # Cycle through day types across the week so each recurs the right
    # number of times (e.g. 2 parts + 5 workout days -> parts rotate)
    day_type_cycle = [day_types[i % len(day_types)] for i in range(workout_days)]

    day_labels = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    schedule = _build_workout_rest_pattern(workout_days)

    days = []
    cycle_idx = 0
    for day_num, (label, stype) in enumerate(zip(day_labels, schedule), start=1):
        if stype == "rest":
            days.append({
                "day": day_num,
                "label": f"{label} — Rest & Recovery",
                "type": "rest",
                "focus": "recovery",
                "exercises": [],
                "duration_minutes": 0,
                "notes": (
                    "Active rest: gentle stretch, pranayama, or short walk. Stay hydrated."
                    if yoga_only else
                    "Active rest: light walking, stretching, or yoga. Stay hydrated."
                )
            })
            continue

        day_type = day_type_cycle[cycle_idx]
        cycle_idx += 1
        ex_list = resolved_sets.get(day_type, [])

        formatted = []
        for ex in ex_list:
            formatted.append({
                "name":          ex["name"],
                "sets":          sets,
                "reps":          reps,
                "rest_seconds":  rest,
                "gif_url":       ex.get("gif_url",""),
                "image_url":     ex.get("image_url",""),
                "video_url":     ex.get("video_url",""),
                "target_muscle": ex.get("target_muscle", ex.get("body_part","")),
                "body_part":     ex.get("body_part",""),
                "equipment":     ex.get("equipment",""),
                "difficulty":    ex.get("difficulty",""),
                "instructions":  (ex.get("description") or "")[:900] if ex.get("description") else "",
                "modification":  ""
            })

        days.append({
            "day":              day_num,
            "label":            f"{label} — {day_type}",
            "type":             "workout",
            "focus":            day_type.lower(),
            "exercises":        formatted,
            "duration_minutes": time_min,
            "notes":            ""
        })

    return days


# ── GPT-4o plan synthesis ─────────────────────────────────────────────────────

def _resolve_activity_level(activity_level: Optional[str], fitness_level: Optional[str]) -> str:
    """
    Map profile fields to calculate_calories keys:
      sedentary | lightly_active | moderately_active | very_active
    """
    aliases = {
        "sedentary": "sedentary",
        "light": "lightly_active",
        "lightly_active": "lightly_active",
        "moderate": "moderately_active",
        "moderately_active": "moderately_active",
        "active": "very_active",
        "very_active": "very_active",
        "very active": "very_active",
        # fitness_level sometimes lands in activity_level by mistake
        "beginner": "lightly_active",
        "intermediate": "moderately_active",
        "expert": "very_active",
    }
    if activity_level:
        key = aliases.get(str(activity_level).strip().lower().replace(" ", "_"))
        if key:
            return key
    fitness_map = {
        "beginner": "lightly_active",
        "intermediate": "moderately_active",
        "expert": "very_active",
    }
    return fitness_map.get((fitness_level or "").lower(), "lightly_active")


def _as_safety_note(note, flag: str = "general") -> Dict:
    """PlanPage expects {flag, note, citation?} — never bare strings."""
    if isinstance(note, dict):
        text = note.get("note") or note.get("text") or note.get("message") or ""
        if not text and note.get("flag"):
            text = str(note.get("flag"))
        return {
            "flag": note.get("flag") or flag,
            "note": text or str(note),
            "citation": note.get("citation"),
        }
    return {"flag": flag, "note": str(note), "citation": None}


def _normalize_safety_notes(notes: Optional[List]) -> List[Dict]:
    return [_as_safety_note(n) for n in (notes or [])]


def _calorie_target_for_profile(profile: Dict) -> Optional[Dict]:
    """
    Mifflin–St Jeor target from profile (with India age/sex size fallback).
    Independent of the LLM enrichment call — safe to keep on enrichment failure.
    """
    sex = profile.get("sex") or profile.get("gender") or ""
    age = profile.get("age")
    anthro = resolve_height_weight(
        age=age,
        sex=sex,
        height_cm=profile.get("height_cm"),
        weight_kg=profile.get("weight_kg"),
    )
    weight = anthro.get("weight_kg")
    height = anthro.get("height_cm")
    size_estimated = bool(anthro.get("height_estimated") or anthro.get("weight_estimated"))
    if not (age and weight and height and sex):
        return None
    activity = _resolve_activity_level(
        profile.get("activity_level"),
        profile.get("fitness_level"),
    )
    try:
        calorie_target = calculate_calories(
            age=age, sex=sex, weight_kg=weight, height_cm=height,
            activity_level=activity,
            goal=profile.get("goal", "general_fitness"),
        )
    except Exception as e:
        print(f"[Plan] calorie target calc skipped ({e})")
        return None
    if size_estimated and calorie_target:
        calorie_target["estimated_from_age_sex"] = True
        calorie_target["anthropometrics_source"] = anthro.get("anthropometrics_source")
        calorie_target["assumed_height_cm"] = height
        calorie_target["assumed_weight_kg"] = weight
        calorie_target["height_estimated"] = anthro.get("height_estimated")
        calorie_target["weight_estimated"] = anthro.get("weight_estimated")
    return calorie_target


def synthesize_plan(
    profile: Dict,
    week_plan: List[Dict],
    guidelines: List[Dict],
    llm,
) -> Dict:
    """
    GPT-4o enriches the plan with:
    - Exercise modifications for health flags
    - Day-level notes
    - Full diet plan grounded in RAG passages
    - Safety notes with citations
    - India-specific tips
    """
    # Format guidelines context for the prompt
    guideline_context = "\n\n".join([
        f"[{g['source_name']}, p.{g['page_number']}, {g['trust_tier']}]\n{g['text'][:400]}"
        for g in guidelines[:8]
    ])

    # Format exercise list for prompt
    exercise_names = []
    for day in week_plan:
        for ex in day.get("exercises", []):
            exercise_names.append(ex["name"])

    flags   = profile.get("health_flags", ["none"])
    goal    = (profile.get("goal") or "").replace("_", " ")
    sex     = profile.get("sex") or profile.get("gender") or ""
    age     = profile.get("age", "")
    anthro = resolve_height_weight(
        age=age,
        sex=sex,
        height_cm=profile.get("height_cm"),
        weight_kg=profile.get("weight_kg"),
    )
    weight = anthro.get("weight_kg")
    height = anthro.get("height_cm")
    size_estimated = bool(anthro.get("height_estimated") or anthro.get("weight_estimated"))

    calorie_target = _calorie_target_for_profile(profile)
    if calorie_target:
        est_note = ""
        if size_estimated:
            bits = []
            if anthro.get("height_estimated"):
                bits.append(f"height≈{height} cm")
            if anthro.get("weight_estimated"):
                bits.append(f"weight≈{weight} kg")
            est_note = (
                f" [ESTIMATED from India age/sex averages: {', '.join(bits)} "
                "— ask user for real height/weight when possible]"
            )
        calorie_target_note = f"{calorie_target['target_calories']} kcal/day{est_note}"
    else:
        calorie_target_note = "not available — insufficient profile data"

    size_line = f"{weight}kg / {height}cm" if weight and height else "unknown"
    if size_estimated and weight and height:
        size_line += " (partially/fully estimated from India age–sex midpoints)"

    prompt = f"""You are an expert India-focused fitness and nutrition planner.
Generate a complete personalised plan. Use the provided guidelines for citations.

USER PROFILE:
- Goal: {goal}
- Age/Sex: {age} / {sex}
- Weight/Height: {size_line}
- Health conditions: {', '.join(flags)}
- Fitness level: {profile.get('fitness_level')}
- Time per day: {profile.get('time_per_day_minutes')} minutes
- Equipment: {', '.join(profile.get('available_equipment', ['none']))}
- Daily calorie target (already calculated via Mifflin-St Jeor, do not recompute): {calorie_target_note}

EXERCISES IN PLAN:
{', '.join(set(exercise_names))}

GUIDELINE PASSAGES (use these to ground advice):
{guideline_context}

IMPORTANT — DO NOT invent calories or macro grams anywhere in your response.
Those numbers are computed separately from a verified nutrient database
after you respond. Your job here is choosing WHICH real Indian dishes fit
each meal slot and writing guidance text — not stating nutrition numbers.

Return a single valid JSON object:
{{
  "exercise_modifications": {{
    "<exercise_name>": "<short form tip for a health flag if useful, else empty string>"
  }},
  "day_notes": {{
    "<day_label>": "<1-sentence note for that day>"
  }},
  "diet_plan": {{
    "meals": [
      {{
        "meal": "breakfast",
        "suggestions": ["<specific Indian dish name, e.g. 'Idli with sambar'>", "<alternative dish>"],
        "timing": "7:00 - 8:00 AM",
        "notes": "<brief tip e.g. avoid sugar in chai>"
      }},
      {{ "meal": "mid_morning_snack", "suggestions": ["<dish>", "<dish>"], "timing": "10:30 AM", "notes": "" }},
      {{ "meal": "lunch", "suggestions": ["<dish>", "<dish>", "<dish>"], "timing": "1:00 - 2:00 PM", "notes": "" }},
      {{ "meal": "evening_snack", "suggestions": ["<dish>", "<dish>"], "timing": "5:00 - 6:00 PM", "notes": "" }},
      {{ "meal": "dinner", "suggestions": ["<dish>", "<dish>", "<dish>"], "timing": "7:00 - 8:00 PM", "notes": "" }}
    ],
    "india_specific_tips": ["<actionable tip>", "<tip>", "<tip>"],
    "foods_to_avoid": ["<food — reason>", "<food — reason>"],
    "hydration": "<daily water recommendation>"
  }},
  "safety_notes": [
    {{
      "flag": "<health_flag>",
      "note": "<specific safe guidance>",
      "citation": "<source name and page number>"
    }}
  ],
  "citations": [
    {{
      "source": "<source_name>",
      "page": <page_number or null>,
      "used_for": "<what this citation supports>"
    }}
  ],
  "weekly_tips": ["<practical tip>", "<tip>", "<tip>"]
}}

RULES:
- The FIRST item in each meal's "suggestions" list must be a real, specific
  Indian dish name (e.g. "Idli with sambar", "Missi roti", "Paneer bhurji")
  written plainly — no invented quantities/grams, that's looked up separately.
- If high_bp: avoid isometric holds, recommend low-sodium foods
- Citations must only reference the provided guideline passages
- exercise_modifications: ONLY optional short form cues (e.g. "keep knees soft").
  NEVER write "Avoid", "Not recommended", "do not perform", or "replace with…"
  for any exercise — every listed move is already filtered to be appropriate.
  If you have nothing useful to add, use "".
- Return ONLY valid JSON, no markdown fences"""

    response = llm.invoke([
        SystemMessage(content="You are a certified fitness and nutrition expert. Return only valid JSON."),
        HumanMessage(content=prompt),
    ])

    raw = response.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        enrichment = json.loads(raw)
    except json.JSONDecodeError:
        # Partial recovery — return empty enrichment rather than crash
        enrichment = {
            "exercise_modifications": {},
            "day_notes": {},
            "diet_plan": {"meals": [], "india_specific_tips": [], "foods_to_avoid": []},
            "safety_notes": [],
            "citations": [],
            "weekly_tips": [],
        }

    # Attach the code-computed calorie target (never the LLM's) under a
    # reserved key so ground_diet_plan_in_nutrition_db() can use it below.
    enrichment["_calorie_target"] = calorie_target
    return enrichment


def apply_enrichment(week_plan: List[Dict], enrichment: Dict) -> List[Dict]:
    """Merge GPT-4o enrichment back into the structured week plan."""
    mods      = enrichment.get("exercise_modifications", {})
    day_notes = enrichment.get("day_notes", {})
    ban = ("avoid", "not recommended", "do not perform", "don't perform",
           "replace with", "substitute with", "skip this")

    for day in week_plan:
        day["notes"] = day_notes.get(day["label"], day.get("notes", ""))

        for ex in day.get("exercises", []):
            tip = (mods.get(ex["name"], "") or "").strip()
            # Drop "avoid / not recommended" noise — those moves should not
            # have been in the plan; never show a red-flag that contradicts the card.
            if tip and any(b in tip.lower() for b in ban):
                tip = ""
            ex["modification"] = tip

    return week_plan


def _clean_dish_query(suggestion: str) -> str:
    """Strip quantities/parentheticals so the semantic lookup query is just
    the dish itself — e.g. '2 idli (180g) with sambar' -> 'idli with sambar'."""
    s = re.sub(r'\([^)]*\)', '', suggestion)
    s = re.sub(r'^\s*\d+[\.\d]*\s*', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def _as_str_list(value) -> List:
    """Normalize LLM list/string/dict tip fields into a flat list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        t = value.strip()
        return [t] if t else []
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
            elif isinstance(v, dict):
                name = v.get("name") or v.get("dish") or v.get("text")
                if isinstance(name, str) and name.strip():
                    out.append(name.strip())
        return out
    if isinstance(value, dict):
        return [s for v in value.values() for s in _as_str_list(v)]
    return []


def _normalize_meals(meals) -> List[Dict]:
    """Accept list-or-dict meal payloads from the LLM without crashing grounding."""
    if not meals:
        return []
    if isinstance(meals, dict):
        normalized = []
        for key, val in meals.items():
            if isinstance(val, dict):
                row = dict(val)
                row.setdefault("meal", key)
                normalized.append(row)
            else:
                normalized.append({"meal": key, "suggestions": _as_str_list(val)})
        meals = normalized
    if not isinstance(meals, list):
        return []
    out = []
    for meal in meals:
        if not isinstance(meal, dict):
            continue
        row = dict(meal)
        row["suggestions"] = _as_str_list(row.get("suggestions"))
        out.append(row)
    return out


def ground_diet_plan_in_nutrition_db(diet_plan: Dict, calorie_target: Optional[Dict]) -> Dict:
    """
    Replace LLM-guessed meal/day nutrition numbers with real values looked
    up from the Indian Nutrient Databank (INDB — see nutrition_lookup.py).
    The LLM only chose WHICH dishes go in each meal slot; every number here
    is either a verified DB value or explicitly marked unverified — nothing
    is silently invented, mirroring the same "verify, don't trust" approach
    used for the exercise/financial-statement accuracy checks elsewhere.
    """
    if not isinstance(diet_plan, dict):
        diet_plan = {}
    meals = _normalize_meals(diet_plan.get("meals"))
    diet_plan["india_specific_tips"] = _as_str_list(diet_plan.get("india_specific_tips"))
    diet_plan["foods_to_avoid"] = _as_str_list(diet_plan.get("foods_to_avoid"))
    total_kcal = total_protein = total_carb = total_fat = 0.0
    any_verified = False

    for meal in meals:
        suggestions = meal.get("suggestions") or []
        primary = suggestions[0] if suggestions else None
        match = get_verified_macros(_clean_dish_query(primary)) if primary else None

        if match:
            meal["calories"]      = match["calories"]
            meal["protein_g"]     = match["protein_g"]
            meal["carb_g"]        = match["carb_g"]
            meal["carbs_g"]       = match["carb_g"]  # UI reads carbs_g
            meal["fat_g"]         = match["fat_g"]
            meal["matched_food"]  = match["matched_food"]
            meal["match_score"]   = match["match_score"]
            meal["nutrient_unit"] = match["unit"]
            meal["verified"]      = True
            any_verified = True
            total_kcal    += match["calories"]
            total_protein += match["protein_g"]
            total_carb    += match["carb_g"]
            total_fat     += match["fat_g"]
        else:
            meal["verified"] = False
            meal["nutrient_note"] = (
                "No confident match in the nutrient database — this meal's "
                "figures are unverified estimates, not sourced values."
            )

    diet_plan["meals"] = meals
    unverified_n = sum(1 for m in meals if not m.get("verified"))

    # Expose BMR/TDEE target for the Nutrition UI (not only meal-sum estimate)
    if calorie_target:
        diet_plan["bmr"] = calorie_target.get("bmr")
        diet_plan["tdee"] = calorie_target.get("tdee")
        diet_plan["target_calories"] = calorie_target.get("target_calories")
        diet_plan["calorie_target"] = {
            "bmr": calorie_target.get("bmr"),
            "tdee": calorie_target.get("tdee"),
            "target_calories": calorie_target.get("target_calories"),
            "macros": calorie_target.get("macros"),
        }
        if calorie_target.get("estimated_from_age_sex"):
            diet_plan["calories_estimated"] = True
            diet_plan["anthropometrics_source"] = calorie_target.get("anthropometrics_source")
            diet_plan["assumed_height_cm"] = calorie_target.get("assumed_height_cm")
            diet_plan["assumed_weight_kg"] = calorie_target.get("assumed_weight_kg")
            diet_plan["calorie_estimate_note"] = (
                "BMR/TDEE used India age/sex average height and/or weight because "
                "the user did not provide measured values. Update height/weight for accuracy."
            )

    if any_verified:
        diet_plan["daily_calories_estimate"] = round(total_kcal)
        diet_plan["macros"] = {
            "protein_g": round(total_protein),
            "carbs_g":   round(total_carb),
            "fat_g":     round(total_fat),
        }
        note = (
            "Sum of verified per-meal values from the Indian Nutrient Databank (INDB)."
        )
        if unverified_n:
            note += (
                f" {unverified_n} meal(s) unmatched — daily total undercounts those meals."
            )
        diet_plan["calorie_note"] = note
        diet_plan["unverified_meal_count"] = unverified_n

    # Compare the ACTUAL sourced total against the BMR-based TARGET — same
    # expected-vs-actual discipline as the accuracy_engine checks: flag any
    # meaningful mismatch instead of silently presenting both as agreeing.
    if calorie_target and any_verified:
        target = calorie_target.get("target_calories")
        actual = diet_plan["daily_calories_estimate"]
        if target:
            delta = actual - target
            diet_plan["calorie_target_check"] = {
                "target_calories":  target,
                "actual_calories_from_meals": actual,
                "delta": delta,
                "within_tolerance": abs(delta) <= max(150, 0.1 * target),
            }

    diet_plan["nutrient_source"] = (
        "Indian Nutrient Databank (INDB), derived from ICMR-NIN Indian Food "
        "Composition Tables 2017/2004 — github.com/lindsayjaacks/"
        "Indian-Nutrient-Databank-INDB-"
    )
    return diet_plan


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_plan(
    profile: Dict,
    sql_filters: Dict,
    rag_filters: Dict,
) -> Dict:
    """
    Full plan generation pipeline.
    Supports plan_mode="diet_only" (meals + safety only, empty week_plan).
    """
    plan_id = str(uuid.uuid4())
    mode = profile.get("plan_mode") or "full"
    diet_only = mode == "diet_only"
    yoga_only = mode == "yoga_only"
    sql_filters = sql_filters or {}

    # Stash exclusions so select_and_schedule / resolve_exercise_sets honour them
    profile = dict(profile)
    profile["_sql_filters"] = sql_filters
    if sql_filters.get("exclude_body_parts"):
        profile["_sql_exclude_body_parts"] = sql_filters["exclude_body_parts"]
    if yoga_only and not profile.get("available_equipment"):
        profile["available_equipment"] = ["body only"]

    rag_queries = build_rag_queries(profile)
    guidelines = retrieve_multi_query(rag_queries, rag_filters or {}, top_k_per_query=4)
    print(f"[Plan] Retrieved {len(guidelines)} guideline chunks from Qdrant")

    if diet_only:
        week_plan = []
        unique_exercise_names = set()
        print("[Plan] diet_only mode — skipping workout schedule")
    else:
        week_plan = select_and_schedule(profile)
        unique_exercise_names = {
            ex["name"] for d in week_plan for ex in d.get("exercises", [])
        }
        print(
            f"[Plan] Scheduled {sum(1 for d in week_plan if d['type']=='workout')} workout days, "
            f"{len(unique_exercise_names)} distinct exercises across the split"
        )

    enrichment_failed = False
    # Compute BMR outside the LLM try — keep it if enrichment crashes
    fallback_calorie_target = _calorie_target_for_profile(profile)
    try:
        llm = get_llm(temperature=0.4, max_tokens=3000)
        enrichment = synthesize_plan(profile, week_plan, guidelines, llm)
        if not diet_only:
            week_plan = apply_enrichment(week_plan, enrichment)
        print("[Plan] LLM enrichment complete")
    except Exception as e:
        print(f"[Plan] LLM enrichment skipped ({e}) — returning scheduled week_plan only")
        enrichment_failed = True
        enrichment = {
            "diet_plan": {},
            "safety_notes": [
                _as_safety_note(
                    "Diet enrichment failed — meal plan is incomplete. Retry generate_plan.",
                    flag="enrichment",
                )
            ],
            "citations": [],
            "weekly_tips": [],
            # Preserve Mifflin targets even when meal enrichment fails
            "_calorie_target": fallback_calorie_target,
        }

    try:
        diet_plan = ground_diet_plan_in_nutrition_db(
            enrichment.get("diet_plan", {}) or {},
            enrichment.get("_calorie_target"),
        )
        enrichment["diet_plan"] = diet_plan
        print("[Plan] Diet plan grounded in INDB nutrient data")
    except Exception as e:
        print(f"[Plan] Nutrition grounding skipped ({e}) — diet numbers are unverified LLM output")
        enrichment_failed = True

    diet_plan = enrichment.get("diet_plan") or {}
    if diet_only and enrichment_failed and not (diet_plan.get("meals") or []):
        diet_plan["generation_error"] = (
            "Diet plan could not be generated. Please try again."
        )
        enrichment["diet_plan"] = diet_plan

    thin_program = (not diet_only) and len(unique_exercise_names) < 3
    if thin_program:
        enrichment.setdefault("safety_notes", [])
        enrichment["safety_notes"] = list(enrichment.get("safety_notes") or []) + [
            _as_safety_note(
                f"Only {len(unique_exercise_names)} distinct exercise(s) matched filters — "
                "consider widening equipment or body parts and regenerating.",
                flag="thin_program",
            )
        ]

    plan_mode_out = "diet_only" if diet_only else ("yoga_only" if yoga_only else "full")
    plan = {
        "plan_id": plan_id,
        "plan_mode": plan_mode_out,
        "generated_at": datetime.utcnow().isoformat(),
        "profile_summary": {
            "goal": profile.get("goal"),
            "target_body_parts": profile.get("target_body_parts"),
            "age": profile.get("age"),
            "sex": profile.get("sex") or profile.get("gender"),
            "weight_kg": profile.get("weight_kg"),
            "height_cm": profile.get("height_cm"),
            "health_flags": profile.get("health_flags"),
            "custom_health_notes": profile.get("custom_health_notes"),
            "available_equipment": profile.get("available_equipment"),
            "fitness_level": profile.get("fitness_level"),
            "activity_level": _resolve_activity_level(
                profile.get("activity_level"),
                profile.get("fitness_level"),
            ),
            "time_per_day_minutes": profile.get("time_per_day_minutes"),
            "plan_mode": plan_mode_out,
        },
        "week_plan": week_plan,
        "diet_plan": diet_plan,
        "safety_notes": _normalize_safety_notes(enrichment.get("safety_notes", [])),
        "citations": enrichment.get("citations", []),
        "weekly_tips": enrichment.get("weekly_tips", []),
        "stats": {
            "distinct_exercises_in_split": len(unique_exercise_names),
            "guideline_chunks_used": len(guidelines),
            "workout_days": sum(1 for d in week_plan if d.get("type") == "workout"),
            "rest_days": sum(1 for d in week_plan if d.get("type") == "rest"),
            "diet_only": diet_only,
            "yoga_only": yoga_only,
            "enrichment_failed": enrichment_failed,
            "thin_program": thin_program,
            "has_bmr": bool(diet_plan.get("bmr") or diet_plan.get("calorie_target", {}).get("bmr")),
            "calories_estimated": bool(diet_plan.get("calories_estimated")),
        },
    }

    return plan
