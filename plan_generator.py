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
from app.services.nutrition_lookup import get_verified_macros
from app.mcp_server.server import calculate_calories


# ── Query builder for RAG ─────────────────────────────────────────────────────
def build_rag_queries(profile: Dict) -> List[str]:
    """Build multiple targeted semantic queries based on user profile."""
    goal    = profile.get("goal", "").replace("_", " ")
    parts   = ", ".join(profile.get("target_body_parts", []))
    flags   = profile.get("health_flags", ["none"])
    sex     = profile.get("sex", "")
    age     = profile.get("age", "")

    queries = [
        f"{goal} exercise recommendations for {sex} age {age}",
        f"diet and nutrition for {goal} India",
        f"physical activity guidelines for {parts}",
    ]

    # Add safety queries for health flags
    if "high_bp" in flags or "heart_condition" in flags:
        queries.append("exercise safety guidelines high blood pressure hypertension")
        queries.append("diet recommendations high blood pressure India ICMR")
    if "diabetes" in flags:
        queries.append("exercise and diet for diabetes blood sugar India")
    if "knee_injury" in flags or "back_injury" in flags:
        queries.append("exercise modifications joint injury rehabilitation")
    if "obesity" in flags:
        queries.append("physical activity obesity weight management India guidelines")

    return queries


# ── Exercise selection and scheduling ─────────────────────────────────────────
def _build_workout_rest_pattern(workout_days: int) -> List[str]:
    """Interleave workout/rest across 7 days (unchanged from before)."""
    schedule = []
    count = 0
    for i in range(7):
        if count < workout_days and (i % 2 == 0 or count == 0):
            schedule.append("workout")
            count += 1
        else:
            schedule.append("rest")
    for i in range(7):
        if schedule.count("workout") >= workout_days:
            break
        if schedule[i] == "rest":
            schedule[i] = "workout"
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

    exclude_body_parts = []
    injury_to_part = {
        "knee_injury": "upper legs", "back_injury": "back",
        "shoulder_injury": "shoulders", "wrist_injury": "lower arms",
    }
    for flag in health_flags:
        if flag in injury_to_part:
            exclude_body_parts.append(injury_to_part[flag])

    exercises_per_session = {15: 3, 30: 5, 45: 7, 60: 9}.get(time_min, 5)

    if goal in ("build_muscle", "improve_strength"):
        sets, reps, rest = 4, "8-10", 90
    elif goal == "lose_fat":
        sets, reps, rest = 3, "15-20", 30
    elif goal == "improve_flexibility":
        sets, reps, rest = 2, "30s hold", 15
    else:
        sets, reps, rest = 3, "12-15", 45
    if fitness_lvl == "beginner":
        sets = max(2, sets - 1)
        rest = int(rest * 1.5)

    workout_days = 3 if time_min <= 30 else 4

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
    )

    # Cycle through day types across the week so each recurs the right
    # number of times (e.g. 2 parts + 4 workout days -> each part 2x/week)
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
                "notes": "Active rest: light walking, stretching, or yoga. Stay hydrated."
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
                "instructions":  ex.get("description","")[:300] if ex.get("description") else "",
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
    goal    = profile.get("goal", "").replace("_", " ")
    sex     = profile.get("sex", "")
    age     = profile.get("age", "")
    weight  = profile.get("weight_kg")
    height  = profile.get("height_cm")

    # Real BMR-based target (Mifflin-St Jeor) — computed here, never by the LLM.
    calorie_target = None
    if age and weight and height and sex:
        try:
            calorie_target = calculate_calories(
                age=age, sex=sex, weight_kg=weight, height_cm=height,
                activity_level=profile.get("activity_level", "moderate"),
                goal=profile.get("goal", "general_fitness"),
            )
        except Exception as e:
            print(f"[Plan] calorie target calc skipped ({e})")
    calorie_target_note = (
        f"{calorie_target['target_calories']} kcal/day" if calorie_target else "not available — insufficient profile data"
    )

    prompt = f"""You are an expert India-focused fitness and nutrition planner.
Generate a complete personalised plan. Use the provided guidelines for citations.

USER PROFILE:
- Goal: {goal}
- Age/Sex: {age} / {sex}
- Weight/Height: {weight}kg / {height}cm
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
    "<exercise_name>": "<modification for health condition, else empty string>"
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

    for day in week_plan:
        # Add day note
        day["notes"] = day_notes.get(day["label"], day.get("notes", ""))

        # Add exercise modifications
        for ex in day.get("exercises", []):
            ex["modification"] = mods.get(ex["name"], "")

    return week_plan


def _clean_dish_query(suggestion: str) -> str:
    """Strip quantities/parentheticals so the semantic lookup query is just
    the dish itself — e.g. '2 idli (180g) with sambar' -> 'idli with sambar'."""
    s = re.sub(r'\([^)]*\)', '', suggestion)
    s = re.sub(r'^\s*\d+[\.\d]*\s*', '', s)
    return re.sub(r'\s+', ' ', s).strip()


def ground_diet_plan_in_nutrition_db(diet_plan: Dict, calorie_target: Optional[Dict]) -> Dict:
    """
    Replace LLM-guessed meal/day nutrition numbers with real values looked
    up from the Indian Nutrient Databank (INDB — see nutrition_lookup.py).
    The LLM only chose WHICH dishes go in each meal slot; every number here
    is either a verified DB value or explicitly marked unverified — nothing
    is silently invented, mirroring the same "verify, don't trust" approach
    used for the exercise/financial-statement accuracy checks elsewhere.
    """
    meals = diet_plan.get("meals", [])
    total_kcal = total_protein = total_carb = total_fat = 0.0
    any_verified = False

    for meal in meals:
        suggestions = meal.get("suggestions", [])
        primary = suggestions[0] if suggestions else None
        match = get_verified_macros(_clean_dish_query(primary)) if primary else None

        if match:
            meal["calories"]      = match["calories"]
            meal["protein_g"]     = match["protein_g"]
            meal["carb_g"]        = match["carb_g"]
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

    if any_verified:
        diet_plan["daily_calories_estimate"] = round(total_kcal)
        diet_plan["macros"] = {
            "protein_g": round(total_protein),
            "carbs_g":   round(total_carb),
            "fat_g":     round(total_fat),
        }
        diet_plan["calorie_note"] = (
            "Calculated by summing verified per-meal values from the Indian "
            "Nutrient Databank (INDB) — not LLM-estimated."
        )

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
    Called by FastAPI after conversational intake is complete.

    Returns the complete plan JSON ready for frontend rendering.
    """
    plan_id = str(uuid.uuid4())

    # ── Step 1: Retrieve relevant guidelines (RAG) ────────────────────────────
    rag_queries  = build_rag_queries(profile)
    guidelines   = retrieve_multi_query(rag_queries, rag_filters, top_k_per_query=4)
    print(f"[Plan] Retrieved {len(guidelines)} guideline chunks from Qdrant")

    # ── Step 2: Derive split + resolve exercises semantically, schedule week ──
    # (exercise retrieval now happens per day-type inside select_and_schedule,
    #  via exercise_selection.py — semantic match, not a flat SQL pool sliced
    #  by index. sql_filters is still accepted for signature compatibility
    #  with callers but is no longer used to build a shared pool here.)
    week_plan = select_and_schedule(profile)
    unique_exercise_names = {ex["name"] for d in week_plan for ex in d.get("exercises", [])}
    print(f"[Plan] Scheduled {sum(1 for d in week_plan if d['type']=='workout')} workout days, "
          f"{len(unique_exercise_names)} distinct exercises across the split")

    # ── Step 4: LLM enrichment (dish choices, safety text, citations) ─────────
    try:
        llm        = get_llm(temperature=0.4, max_tokens=3000)
        enrichment = synthesize_plan(profile, week_plan, guidelines, llm)
        week_plan  = apply_enrichment(week_plan, enrichment)
        print("[Plan] LLM enrichment complete")
    except Exception as e:
        print(f"[Plan] LLM enrichment skipped ({e}) — returning scheduled week_plan only")
        enrichment = {
            "diet_plan": {},
            "safety_notes": [],
            "citations": [],
            "weekly_tips": [],
            "_calorie_target": None,
        }

    # ── Step 5: Ground diet numbers in the real nutrient DB (INDB) ────────────
    # The LLM only picked dish names above; every calorie/macro figure comes
    # from here now, or is explicitly marked unverified if unmatched.
    try:
        diet_plan = ground_diet_plan_in_nutrition_db(
            enrichment.get("diet_plan", {}),
            enrichment.get("_calorie_target"),
        )
        enrichment["diet_plan"] = diet_plan
        print("[Plan] Diet plan grounded in INDB nutrient data")
    except Exception as e:
        print(f"[Plan] Nutrition grounding skipped ({e}) — diet numbers are unverified LLM output")

    # ── Step 6: Assemble final plan ───────────────────────────────────────────
    plan = {
        "plan_id":        plan_id,
        "generated_at":   datetime.utcnow().isoformat(),
        "profile_summary": {
            "goal":                 profile.get("goal"),
            "target_body_parts":    profile.get("target_body_parts"),
            "age":                  profile.get("age"),
            "sex":                  profile.get("sex"),
            "health_flags":         profile.get("health_flags"),
            "available_equipment":  profile.get("available_equipment"),
            "fitness_level":        profile.get("fitness_level"),
            "time_per_day_minutes": profile.get("time_per_day_minutes"),
        },
        "week_plan":    week_plan,
        "diet_plan":    enrichment.get("diet_plan", {}),
        "safety_notes": enrichment.get("safety_notes", []),
        "citations":    enrichment.get("citations", []),
        "weekly_tips":  enrichment.get("weekly_tips", []),
        "stats": {
            "distinct_exercises_in_split": len(unique_exercise_names),
            "guideline_chunks_used":       len(guidelines),
            "workout_days":                sum(1 for d in week_plan if d["type"] == "workout"),
            "rest_days":                   sum(1 for d in week_plan if d["type"] == "rest"),
        }
    }

    return plan