"""
Semantic NLU — embedding prototypes for soft classification.

Roles:
  1. Fallback Q&A routing when answer_fitness_question is called without
     intent / with media="auto" (agent normally passes those explicitly).
  2. Soft profile slot hints (goal, body parts, health flags, plan_mode)
     used by auto_ingest_profile_semantic.

Inputs: free-text user utterances.
Outputs: intent labels (info | exercise_qa | plan), media preference
(yoga demo vs gym), body-part lists, optional plan_mode.

Structured demographics / health with negation use LLM extract when
SEMANTIC_LLM_EXTRACT=1 (see extract helpers below).
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def _embed_model():
    from app.services.embedder import get_shared_embed_model
    return get_shared_embed_model()


def embed_texts(texts: List[str]) -> np.ndarray:
    from app.services.embedder import encode_texts
    return encode_texts(texts, normalize=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def best_prototype_score(query_vec: np.ndarray, proto_vecs: np.ndarray) -> float:
    if proto_vecs.ndim == 1:
        return cosine_sim(query_vec, proto_vecs)
    return float(np.max(proto_vecs @ query_vec))


DIET_PROTOTYPES = [
    "How many carbohydrates should a diabetic patient eat per day?",
    "What protein and nutrient intake is right for me?",
    "I feel bloated, what foods help digestion?",
    "Suggest an Indian meal plan and calorie targets.",
    "Nutrition advice for cholesterol and blood sugar.",
    "What should I eat for weight loss diet only?",
    "How much water should I drink daily for hydration?",
    "What is the recommended daily water intake?",
    "How much fluid or water to drink regularly?",
]

# Pure factual Q&A — guideline text only (no exercise cards / yoga photos)
INFO_PROTOTYPES = DIET_PROTOTYPES + [
    "What are the WHO physical activity guidelines for adults?",
    "Is running safe with high blood pressure according to guidelines?",
    "How many minutes of moderate activity per week are recommended?",
    "What does ICMR say about protein for vegetarians?",
]

# Topic Q&A that needs a short answer + targeted demos (gym GIFs or yoga photos)
EXERCISE_QA_PROTOTYPES = [
    "How do I lose arm fat? Show exercises for arms and toning.",
    "What are good exercises for lower back pain?",
    "Show me beginner bodyweight moves for biceps with form tips.",
    "How do I do a push-up correctly?",
    "How do I do a squat correctly? Show squat form.",
    "Arm fat reducing exercises with demos.",
    "Exercises to reduce belly fat and strengthen core.",
    "Recommend stretches for tight shoulders and neck.",
    "Best movements to tone thighs without equipment.",
    "Give me squatting workouts and lower body moves.",
    "How do I practice alternate nostril breathing pranayama?",
    "What are the steps for Kapalabhati in Common Yoga Protocol?",
    "How to do tadasana mountain pose technique?",
]

EXERCISE_PROTOTYPES = EXERCISE_QA_PROTOTYPES  # alias used by older callers

PLAN_PROTOTYPES = [
    "Create a personalised weekly workout and diet plan for me.",
    "I want a full fitness plan with training days and meals.",
    "Generate a beginner exercise plan for the whole week.",
    "Build me a customised routine after asking my details.",
    "I need a structured 7-day workout plan.",
    "Make me a diet-only meal plan for the week.",
]

DIET_ONLY_PLAN_PROTOTYPES = [
    "I only want a diet plan without any exercise.",
    "Just give me nutrition and meal plan, no workouts.",
    "Diet only plan please, skip exercise.",
    "A healthy diet plan is required",
]

FULL_PLAN_PROTOTYPES = PLAN_PROTOTYPES

YOGA_ONLY_PLAN_PROTOTYPES = [
    "I want a yoga plan for the week",
    "Create a yoga-only routine with asanas and pranayama",
    "Give me a weekly yoga practice plan, not gym workouts",
    "Build a Common Yoga Protocol style yoga schedule",
    "I only want yoga asanas stretching breathing practice plan",
]

# When True, retrieve CLIP photos from yoga / Fit India protocol PDFs.
# Gym form questions (squat, arm fat, push-up) must stay False so Free
# Exercise DB GIFs are shown instead of random booklet cartoons.
GUIDELINE_DEMO_PROTOTYPES = [
    "How do I do this yoga asana from the Common Yoga Protocol?",
    "Show step-by-step pranayama alternate nostril breathing technique",
    "Yoga protocol demonstration photo for this pose",
    "How to perform Surya Namaskar from the yoga booklet",
    "Anulom Vilom Bhramari Kapalabhati technique from yoga protocol",
    "What is pranayama and how is it practiced?",
    "Explain this yoga posture with demonstration steps",
    "Boat pose naukasana muscular strength how to perform",
    "Mountain pose standing asana technique from yoga booklet",
    "Tree pose balancing yoga asana demonstration",
    "Alternate nostril breathing nadi shodhana technique",
]

GYM_EXERCISE_MEDIA_PROTOTYPES = [
    "How do I do a squat correctly gym form",
    "Arm fat reducing exercises with workout GIFs",
    "Show me push-up and tricep dip exercise demos",
    "Best gym or bodyweight moves to tone arms",
    "Squatting workouts and strength training form tips",
]


BODYWEIGHT_PROTOTYPES = [
    "I have no equipment, only bodyweight at home.",
    "Without any equipment, body only floor exercises.",
    "Bodyweight training with nothing at home.",
]

FULL_BODY_PROTOTYPES = [
    "full body workout covering all muscle regions",
    "whole body total body entire body training",
    "I want to train full body all regions",
]

GOAL_PROTOTYPES = {
    "lose_fat": ["I want to lose fat and reduce belly weight", "weight loss and fat loss goal"],
    "build_muscle": ["I want to build muscle and gain mass", "hypertrophy muscle building"],
    "improve_strength": ["I want to get stronger and lift heavier", "strength training goal"],
    "improve_flexibility": ["I want better flexibility and mobility", "stretching and yoga flexibility"],
    "improve_endurance": ["I want better stamina and cardio endurance", "running endurance training"],
    "general_fitness": ["I want general fitness and to stay healthy", "overall fitness wellness"],
    "rehabilitation": ["I am recovering from injury and need rehab", "rehabilitation physiotherapy"],
    "stress_relief": ["I want stress relief and relaxation", "calm mental wellness exercise"],
}

FITNESS_PROTOTYPES = {
    "beginner": ["I am a beginner new to exercise never trained", "just starting fitness"],
    "intermediate": ["I am intermediate train regularly sometimes", "moderate experience gym"],
    "expert": ["I am advanced expert athlete competitive", "very experienced lifter"],
}

HEALTH_FLAG_PROTOTYPES = {
    "diabetes": ["I have diabetes", "I am a diabetic patient with high blood sugar"],
    "high_bp": ["I have high blood pressure or hypertension"],
    "low_bp": ["I have low blood pressure or hypotension"],
    "knee_injury": ["I have a knee injury or knee pain or ACL tear"],
    "back_injury": ["I have back pain disc injury or spondylitis"],
    "shoulder_injury": ["I have shoulder injury or rotator cuff pain"],
    "wrist_injury": ["I have wrist injury or carpal tunnel pain"],
    "ankle_injury": ["I have ankle injury or plantar fasciitis pain"],
    "heart_condition": ["I have a heart condition or cardiac issue"],
    "asthma": ["I have asthma or breathing problems"],
    "osteoporosis": ["I have osteoporosis or low bone density"],
    "acidity": ["I have acidity GERD or gastritis"],
    "pregnancy": ["I am pregnant"],
    "obesity": ["I am obese or overweight with high BMI"],
    "none": ["I have no health issues no injuries I am healthy"],
}

HEALTH_FLAG_NEGATIONS = {
    "diabetes": ["I am not diabetic", "I do not have diabetes", "non-diabetic"],
    "high_bp": ["I do not have high blood pressure", "blood pressure is normal"],
    "low_bp": ["I do not have low blood pressure"],
    "knee_injury": ["my knees are fine no knee injury"],
    "back_injury": ["no back pain no back injury"],
    "shoulder_injury": ["no shoulder injury"],
    "wrist_injury": ["no wrist injury"],
    "ankle_injury": ["no ankle injury"],
    "heart_condition": ["no heart condition healthy heart"],
    "asthma": ["I do not have asthma"],
    "osteoporosis": ["no osteoporosis"],
    "acidity": ["no acidity no GERD"],
    "pregnancy": ["I am not pregnant"],
    "obesity": ["I am not obese"],
    "none": ["I have several health conditions"],
}

BODY_PART_PROTOTYPES = {
    "neck": ["neck mobility strengthening"],
    "shoulders": ["shoulders deltoids rotator cuff"],
    "chest": ["chest pectorals press fly"],
    "back": ["back lats rows lower back"],
    "upper arms": ["biceps triceps upper arms"],
    "lower arms": ["forearms wrists grip"],
    "waist": ["core abs waist midsection obliques"],
    "upper legs": ["quads hamstrings thighs glutes"],
    "lower legs": ["calves lower legs"],
    "cardio": ["cardio conditioning endurance"],
}


@lru_cache(maxsize=1)
def _proto_bank() -> Dict[str, np.ndarray]:
    bank: Dict[str, np.ndarray] = {}
    groups = {
        "diet": DIET_PROTOTYPES,
        "info": INFO_PROTOTYPES,
        "exercise": EXERCISE_QA_PROTOTYPES,
        "exercise_qa": EXERCISE_QA_PROTOTYPES,
        "plan": PLAN_PROTOTYPES,
        "diet_only_plan": DIET_ONLY_PLAN_PROTOTYPES,
        "full_plan": FULL_PLAN_PROTOTYPES,
        "yoga_only_plan": YOGA_ONLY_PLAN_PROTOTYPES,
        "guideline_demo": GUIDELINE_DEMO_PROTOTYPES,
        "gym_exercise_media": GYM_EXERCISE_MEDIA_PROTOTYPES,
        "bodyweight": BODYWEIGHT_PROTOTYPES,
        "full_body": FULL_BODY_PROTOTYPES,
    }
    for key, phrases in groups.items():
        bank[key] = embed_texts(phrases)
    for g, phrases in GOAL_PROTOTYPES.items():
        bank[f"goal:{g}"] = embed_texts(phrases)
    for lvl, phrases in FITNESS_PROTOTYPES.items():
        bank[f"fitness:{lvl}"] = embed_texts(phrases)
    for flag, phrases in HEALTH_FLAG_PROTOTYPES.items():
        bank[f"health:{flag}"] = embed_texts(phrases)
    for flag, phrases in HEALTH_FLAG_NEGATIONS.items():
        bank[f"health_neg:{flag}"] = embed_texts(phrases)
    for bp, phrases in BODY_PART_PROTOTYPES.items():
        bank[f"body:{bp}"] = embed_texts(phrases)
    return bank


def _qvec(text: str) -> np.ndarray:
    return embed_texts([text or ""])[0]


def query_is_diet_focused(query: str, threshold: float = 0.42) -> bool:
    """True when the turn should be guidelines-only (no exercise cards)."""
    return classify_turn_intent(query) == "info"


def classify_turn_intent(query: str) -> str:
    """
    Three product intents (mutually exclusive for routing):

      info        — answer from document chunks only (water, macros, guidelines)
      exercise_qa — short answer + a few targeted exercise demos (e.g. lose arm fat)
      plan        — collect profile slots and/or generate a customised week plan

    Returns one of: "info" | "exercise_qa" | "plan"
    """
    if not (query or "").strip():
        return "info"
    v = _qvec(query)
    bank = _proto_bank()
    scores = {
        "info": best_prototype_score(v, bank["info"]),
        "exercise_qa": best_prototype_score(v, bank["exercise_qa"]),
        "plan": max(
            best_prototype_score(v, bank["plan"]),
            best_prototype_score(v, bank["diet_only_plan"]),
            best_prototype_score(v, bank["full_plan"]),
            # yoga_only_plan is ONLY for plan_mode — including it here makes
            # "how to do pranayama" look like "I want a yoga plan".
        ),
    }
    # Technique/how-to with booklet demos → exercise_qa (not plan intake)
    if query_wants_guideline_demo(query):
        return "exercise_qa"

    # Plan wins only when clearly ahead (avoid treating "arm fat tips" as plan)
    winner = max(scores, key=scores.get)
    if winner == "plan" and scores["plan"] < 0.48:
        # Weak plan signal — fall back to info vs exercise
        return "exercise_qa" if scores["exercise_qa"] >= scores["info"] else "info"
    if winner == "plan" and scores["plan"] < scores["exercise_qa"] + 0.03:
        return "exercise_qa"
    if winner == "info" and scores["exercise_qa"] >= scores["info"] + 0.04:
        return "exercise_qa"
    return winner


def query_wants_bodyweight(query: str, threshold: float = 0.45) -> bool:
    if not (query or "").strip():
        return False
    return best_prototype_score(_qvec(query), _proto_bank()["bodyweight"]) >= threshold


def classify_plan_mode(query: str) -> Optional[str]:
    if not (query or "").strip():
        return None
    v = _qvec(query)
    bank = _proto_bank()
    diet_only = best_prototype_score(v, bank["diet_only_plan"])
    full = best_prototype_score(v, bank["full_plan"])
    yoga_only = best_prototype_score(v, bank["yoga_only_plan"])
    # Keyword shortcuts (embedding alone can miss short "yoga plan")
    low = query.lower()
    if any(k in low for k in ("yoga plan", "yoga-only", "yoga only", "only yoga", "yogic plan")):
        return "yoga_only"
    if yoga_only >= 0.46 and yoga_only >= full - 0.02 and yoga_only >= diet_only:
        return "yoga_only"
    if full >= 0.48 and full >= diet_only + 0.02 and full >= yoga_only:
        return "full"
    if diet_only >= 0.48 and diet_only > full and diet_only > yoga_only:
        return "diet_only"
    return None


def query_wants_guideline_demo(query: str, threshold: float = 0.42) -> bool:
    """
    Fallback only: should booklet demos (CYP / Fit India) be retrieved?

    Primary decision is the agent's media= arg. This uses embedding prototypes
    (guideline_demo vs gym_exercise_media) — not a hardcoded pose-name list.
    Age-band protocol LIST asks stay False (text tables, not pose photos).
    """
    if not (query or "").strip():
        return False
    low = query.lower()
    listing = any(w in low for w in (
        "for 50", "for 35", "for 18", "year old", "years of age",
        "age group", "protocol for",
    ))
    how_to = any(w in low for w in (
        "how", "technique", "steps", "show", "do i", "perform", "practice",
        "demonstrate", "what is", "explain",
    ))
    if listing and not how_to:
        return False

    v = _qvec(query)
    bank = _proto_bank()
    demo = best_prototype_score(v, bank["guideline_demo"])
    gym = best_prototype_score(v, bank["gym_exercise_media"])

    # Clear booklet-demo signal
    if demo >= threshold and demo >= gym - 0.02:
        return True
    # Yoga-domain phrasing with a competitive demo score
    yogaish = any(t in low for t in ("yoga", "asana", "āsana", "pranayam", "prāṇāyām"))
    if yogaish and demo >= 0.36 and demo >= gym - 0.05:
        return True
    # Short technique names often score mid-demo / near-zero gym (Sanskrit).
    # Prefer booklet when gym signal is weak and demo clearly leads — still
    # embedding-relative, not a pose-name list.
    if gym < 0.28 and demo >= 0.18 and (demo - gym) >= 0.12:
        return True
    return False


def match_body_parts(query: str, threshold: float = 0.40, top_n: int = 3) -> List[str]:
    if not (query or "").strip():
        return []
    v = _qvec(query)
    bank = _proto_bank()
    scored = []
    for bp in BODY_PART_PROTOTYPES:
        s = best_prototype_score(v, bank[f"body:{bp}"])
        if s >= threshold:
            scored.append((s, bp))
    scored.sort(reverse=True)
    return [bp for _, bp in scored[:top_n]]


def match_goal(query: str, threshold: float = 0.45) -> Optional[str]:
    v = _qvec(query)
    bank = _proto_bank()
    best_g, best_s = None, 0.0
    for g in GOAL_PROTOTYPES:
        s = best_prototype_score(v, bank[f"goal:{g}"])
        if s > best_s:
            best_g, best_s = g, s
    return best_g if best_s >= threshold else None


def match_fitness_level(query: str, threshold: float = 0.45) -> Optional[str]:
    v = _qvec(query)
    bank = _proto_bank()
    best_l, best_s = None, 0.0
    for lvl in FITNESS_PROTOTYPES:
        s = best_prototype_score(v, bank[f"fitness:{lvl}"])
        if s > best_s:
            best_l, best_s = lvl, s
    return best_l if best_s >= threshold else None


def match_health_flags(query: str, threshold: float = 0.55) -> Tuple[List[str], List[str]]:
    """Conservative embedding fallback — prefer LLM extract for health."""
    if not (query or "").strip():
        return [], []
    v = _qvec(query)
    bank = _proto_bank()
    notes: List[str] = []

    chol = embed_texts(["need to lower high cholesterol cholestral"])[0]
    chol_neg = embed_texts(["cholesterol is normal fine"])[0]
    if cosine_sim(v, chol) >= 0.45 and cosine_sim(v, chol) > cosine_sim(v, chol_neg) + 0.05:
        notes.append("high cholesterol")

    scored = []
    for flag in HEALTH_FLAG_PROTOTYPES:
        if flag == "none":
            continue
        aff = best_prototype_score(v, bank[f"health:{flag}"])
        neg = best_prototype_score(v, bank[f"health_neg:{flag}"])
        if aff >= threshold and aff > neg + 0.05:
            scored.append((aff - neg, aff, flag))
    scored.sort(reverse=True)
    flags: List[str] = []
    if scored and (len(scored) == 1 or scored[0][0] >= scored[1][0] + 0.05):
        flags = [scored[0][2]]

    none_s = best_prototype_score(v, bank["health:none"])
    none_neg = best_prototype_score(v, bank["health_neg:none"])
    if none_s >= 0.55 and none_s > none_neg and not flags:
        flags = ["none"]
    return flags, notes


def _match_gender(query: str, threshold: float = 0.40) -> Optional[str]:
    v = _qvec(query)
    male = best_prototype_score(v, embed_texts([
        "The speaker is male", "I am male", "year old male patient",
    ]))
    female = best_prototype_score(v, embed_texts([
        "The speaker is female", "I am a woman", "year old woman patient",
    ]))
    if male >= threshold and male >= female + 0.05:
        return "male"
    if female >= threshold and female >= male + 0.05:
        return "female"
    return None


_EXTRACT_PROMPT = """Extract fitness profile facts from the user message.
Return ONLY valid JSON. Respect negation ("not diabetic" must NOT set diabetes).
Unset keys must be null or [].

Schema:
{{
  "age": int|null,
  "gender": "male"|"female"|null,
  "weight_kg": number|null,
  "height_cm": number|null,
  "goal": one of {goals} | null,
  "fitness_level": "beginner"|"intermediate"|"expert"|null,
  "health_flags": list from {flags},
  "custom_health_notes": string list,
  "available_equipment": list from {equip},
  "target_body_parts": list from {parts} or ["full body"],
  "plan_mode": "full"|"diet_only"|"yoga_only"|null,
  "time_per_day_minutes": int|null
}}

User message: {message}
"""


def llm_extract_profile(message: str) -> Dict[str, Any]:
    if not (message or "").strip():
        return {}
    try:
        from app.llm import get_llm
        from app.conversation.profile_store import (
            GOALS, BODY_PARTS, EQUIPMENT_OPTIONS, KNOWN_FLAGS,
        )
        llm = get_llm(temperature=0.0, max_tokens=400, with_azure_fallback=True)
        prompt = _EXTRACT_PROMPT.format(
            goals=GOALS, flags=KNOWN_FLAGS, equip=EQUIPMENT_OPTIONS,
            parts=BODY_PARTS, message=message,
        )
        raw = llm.invoke(prompt)
        text = getattr(raw, "content", None) or str(raw)
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[SemanticNLU] LLM extract skipped: {e}")
        return {}


def auto_ingest_profile_semantic(user_message: str, profile) -> Dict[str, Any]:
    """
    Embeddings: plan mode, equipment, goal, fitness, body parts.
    LLM extract (opt-in via SEMANTIC_LLM_EXTRACT=1): age, gender, height,
    health flags with negation. Default is off — prefer update_profile tool.
    """
    text = user_message or ""
    updates: Dict[str, Any] = {}

    mode = classify_plan_mode(text)
    if mode == "diet_only":
        updates["plan_mode"] = "diet_only"
        updates["time_per_day_minutes"] = 0
        if not profile.goal:
            updates["goal"] = match_goal(text) or "general_fitness"
    elif mode == "yoga_only":
        updates["plan_mode"] = "yoga_only"
        if not profile.available_equipment:
            updates["available_equipment"] = ["body only"]
        if not profile.goal:
            updates["goal"] = match_goal(text) or "improve_flexibility"
        if not profile.fitness_level:
            lvl = match_fitness_level(text)
            if lvl:
                updates["fitness_level"] = lvl
    elif mode == "full":
        updates["plan_mode"] = "full"
        lvl = match_fitness_level(text)
        if lvl:
            updates["fitness_level"] = lvl

    if query_wants_bodyweight(text) and not profile.available_equipment:
        updates["available_equipment"] = ["body only"]

    if not profile.goal:
        g = match_goal(text)
        if g:
            updates["goal"] = g

    if not profile.fitness_level:
        fl = match_fitness_level(text)
        if fl:
            updates["fitness_level"] = fl

    if not profile.target_body_parts:
        fb = best_prototype_score(_qvec(text), _proto_bank()["full_body"])
        if fb >= 0.45:
            from app.conversation.profile_store import BODY_PARTS
            updates["target_body_parts"] = list(BODY_PARTS)
        else:
            parts = match_body_parts(text)
            if parts:
                updates["target_body_parts"] = parts

    use_llm = os.getenv("SEMANTIC_LLM_EXTRACT", "0") == "1"
    extracted = llm_extract_profile(text) if use_llm else {}

    if extracted:
        if profile.age is None and isinstance(extracted.get("age"), int):
            if 10 <= extracted["age"] <= 100:
                updates["age"] = extracted["age"]
        if not profile.gender and extracted.get("gender") in ("male", "female"):
            updates["gender"] = extracted["gender"]
        if profile.weight_kg is None and extracted.get("weight_kg") is not None:
            try:
                updates["weight_kg"] = float(extracted["weight_kg"])
            except (TypeError, ValueError):
                pass
        if profile.height_cm is None and extracted.get("height_cm") is not None:
            try:
                updates["height_cm"] = float(extracted["height_cm"])
            except (TypeError, ValueError):
                pass
        # Health flags: only from LLM extract when explicitly enabled — never
        # auto-unlock planning from noisy embedding matches alone.
        if not profile.health_flags and extracted.get("health_flags") is not None:
            hf = extracted.get("health_flags") or []
            if "none" in hf and len([x for x in hf if x != "none"]) == 0:
                updates["health_flags"] = ["none"]
            elif hf:
                updates["health_flags"] = [f for f in hf if f != "none"]
        if not profile.custom_health_notes and extracted.get("custom_health_notes"):
            updates["custom_health_notes"] = extracted["custom_health_notes"]
        if extracted.get("plan_mode") in ("full", "diet_only", "yoga_only") and "plan_mode" not in updates:
            updates["plan_mode"] = extracted["plan_mode"]

    if "gender" not in updates and not profile.gender:
        gender = _match_gender(text)
        if gender:
            updates["gender"] = gender
    # Do NOT auto-set health_flags from embeddings (false positives unlock plans).
    # Still capture cholesterol-style notes for RAG enrichment only.
    if not profile.custom_health_notes and "custom_health_notes" not in updates:
        _, notes = match_health_flags(text)
        if notes:
            updates["custom_health_notes"] = notes

    if "age" not in updates and profile.age is None:
        m = re.search(r"\b(\d{1,2})\s*(?:years?\s*old|yrs?\s*old|y/?o|years?\b)", text, re.I)
        if not m:
            m = re.search(r"\bage[:\s]+(\d{1,2})\b", text, re.I)
        if m:
            age = int(m.group(1))
            if 10 <= age <= 100:
                updates["age"] = age
    if "weight_kg" not in updates and profile.weight_kg is None:
        wm = re.search(r"\b(\d{2,3})\s*kgs?\b", text, re.I)
        if wm:
            updates["weight_kg"] = float(wm.group(1))
    if "height_cm" not in updates and profile.height_cm is None:
        hm = re.search(r"\b(1\d{2}|2[0-4]\d)\s*cm\b", text, re.I)
        if not hm:
            hm = re.search(r"\bheight[:\s]+(1\d{2}|2[0-4]\d)\b", text, re.I)
        if hm:
            updates["height_cm"] = float(hm.group(1))

    if updates:
        profile.merge(**updates)
    if updates.get("plan_mode") == "full" and profile.time_per_day_minutes == 0:
        profile.time_per_day_minutes = None
    return updates
