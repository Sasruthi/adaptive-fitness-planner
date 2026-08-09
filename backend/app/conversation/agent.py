"""
FILE LOCATION: backend/app/conversation/agent.py

The Adaptive Fitness Planner — Unified Conversational Agent
=============================================================
This REPLACES the old split architecture:
  - intake_graph.py's regex/keyword slot-filling state machine
  - plan_agent.py being a separate, only-reachable-after-intake agent

Instead: ONE LangGraph ReAct agent owns the whole conversation. It
decides autonomously, per turn, whether to:
  - answer a fitness/nutrition/exercise question directly (RAG)
  - ask for missing profile info (conversationally, not a checklist)
  - generate a plan (once the profile is actually complete + safe)
  - save a plan / send a reminder / log a workout / show progress

Autonomy lives in the LLM's tool-calling. Safety does NOT: generating
a plan without a confirmed health-flag answer is blocked in code
(Profile.is_safe_to_plan), not left to the model's discretion.

plan_agent.py's LangGraph tool-loop (fetch_exercises -> fetch_guidelines
-> calculate_calories -> validate_plan -> retry-if-unsafe) is unchanged
and reused here as the `generate_plan` tool's implementation — that part
of the architecture was already correct.
"""

import contextvars
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.conversation.profile_store import session_store, Profile
from app.conversation.state import build_sql_filters, build_rag_filters, injury_excluded_body_parts
from app.services.rag_retrieval import (
    retrieve_multi_query,
    retrieve_guideline_images,
    retrieve_guidelines_by_pages,
    match_protocol_age_band,
    rerank_guideline_chunks,
    images_from_matching_chunks,
)
from app.services.exercise_rag import retrieve_exercise_semantic
from app.services.plan_agent import generate_plan_agentic
from app.tools.actions import (
    save_plan, send_reminder, log_workout, get_workout_progress,
)
from app.llm import (
    get_llm as _shared_get_llm,
    azure_configured,
    is_failover_error,
    resolve_provider,
)

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# Current thread_id, set right before invoking the agent so tool closures
# (which the LLM calls with only the args it decided on) know which
# session's profile/plan they're touching.
_current_thread: contextvars.ContextVar[str] = contextvars.ContextVar("thread_id")


def get_llm(provider: Optional[str] = None):
    # Failover is handled in process_user_message (create_react_agent binds tools itself).
    return _shared_get_llm(
        temperature=0.2,
        max_tokens=1500,
        provider=provider,
        with_azure_fallback=False,
    )


# ══════════════════════════════════════════════════════════════════════════
# TOOLS — the agent decides which of these to call, and when
# ══════════════════════════════════════════════════════════════════════════

def _auto_ingest_profile_hints(user_message: str, thread_id: str) -> None:
    """
    Semantically capture profile facts from free text (embeddings + LLM extract)
    so the agent does not re-ask for facts already stated.
    """
    from app.services.semantic_nlu import auto_ingest_profile_semantic
    profile = session_store.get_profile(thread_id)
    auto_ingest_profile_semantic(user_message, profile)

def _normalize_exercise_for_ui(e: dict) -> dict:
    """Ensure chat cards get instructions (description) + media fields."""
    desc = (e.get("description") or e.get("instructions") or "").strip()
    return {
        **e,
        "description": desc,
        "instructions": desc,
    }


_VALID_QA_INTENTS = frozenset({"info", "exercise_qa", "plan"})
_VALID_QA_MEDIA = frozenset({"none", "yoga_protocol", "gym_catalog", "auto"})
_VALID_BODY_PARTS = frozenset({
    "neck", "shoulders", "chest", "back", "upper arms", "lower arms",
    "waist", "upper legs", "lower legs", "cardio",
})


def _resolve_qa_intent(intent: Optional[str], query: str) -> str:
    """Prefer the agent's intent arg; fall back to prototype NLU only if omitted."""
    raw = (intent or "").strip().lower().replace("-", "_")
    if raw in _VALID_QA_INTENTS:
        return raw
    from app.services.semantic_nlu import classify_turn_intent
    return classify_turn_intent(query)


def _resolve_qa_media(media: Optional[str], query: str, intent: str) -> str:
    """
    Resolve media mode. Agent-passed media wins (agentic path).

    Fallback when media is missing/"auto": embedding NLU only — no pose-name
    whitelist. Broad yoga+gym dumps → none; booklet demo → yoga_protocol;
    exercise_qa → gym_catalog; info → none.
    """
    raw = (media or "auto").strip().lower().replace("-", "_")
    if raw not in _VALID_QA_MEDIA:
        raw = "auto"
    if raw != "auto":
        return raw
    if _is_broad_mixed_workout_ask(query):
        return "none"
    if intent == "info":
        return "none"
    from app.services.semantic_nlu import query_wants_guideline_demo
    if query_wants_guideline_demo(query):
        return "yoga_protocol"
    if intent == "exercise_qa":
        return "gym_catalog"
    return "none"


def _is_broad_mixed_workout_ask(query: str) -> bool:
    """
    Underspecified mixed ask: yoga lane AND gym/workout lane with a mixing cue.

    No pose-name list — structure of the ask is enough. "Yoga exercises for
    beginners" is not broad; "yoga and gym workouts" is.
    """
    low = (query or "").lower()
    if not low.strip():
        return False
    has_yoga = any(t in low for t in ("yoga", "asana", "pranayam"))
    has_gym_lane = any(
        t in low for t in ("gym", "workout", "workouts", "weight training", "strength training")
    )
    mixed = any(
        w in low for w in (
            " and ", " & ", " plus ", " as well as ", " both ",
            "combine", "mixed", "along with",
        )
    )
    return bool(has_yoga and has_gym_lane and mixed)


@tool
def answer_fitness_question(
    query: str,
    intent: Optional[str] = None,
    media: Optional[str] = None,
    focus_body_parts: Optional[List[str]] = None,
) -> str:
    """
    Answer a factual fitness/nutrition/exercise/yoga question (NOT a full weekly plan).

    YOU choose intent + media (agentic routing). Pass them every time:

      intent:
        info         — nutrition, hydration, WHO/ICMR facts, guideline tables
        exercise_qa  — form tips / "exercises for X" that need demos
        plan         — user wants a customised week plan (do NOT use this tool;
                       call update_profile / generate_plan instead). If you still
                       call with intent=plan, this tool redirects you.

      media:
        none           — text guidelines only (diet, water, age-band protocol lists)
        yoga_protocol  — Common Yoga Protocol / Fit India DEMO PHOTOS for a
                         SPECIFIC asana/pranayama (name it in query). Never for
                         vague "show me yoga and workouts" — clarify first.
        gym_catalog    — Free Exercise DB GIF cards (squat, arm fat, push-ups)
        auto           — only if unsure; server falls back to a small NLU heuristic

      focus_body_parts (optional): neck, shoulders, chest, back, upper arms,
        lower arms, waist, upper legs, lower legs, cardio — for gym_catalog search.
    """
    thread_id = _current_thread.get()
    profile = session_store.get_profile(thread_id)
    from app.services.semantic_nlu import match_body_parts, query_wants_bodyweight

    intent = _resolve_qa_intent(intent, query)
    media_mode = _resolve_qa_media(media, query, intent)

    # Plan flow is profile → generate_plan; this tool must not invent a week.
    if intent == "plan":
        session_store.set_exercises(thread_id, [])
        session_store.set_guideline_images(thread_id, [])
        return (
            "INTENT=plan. The user wants a customised plan, not a one-off tip. "
            "Do NOT list random exercises. Call get_profile_status, ask only for "
            "missing slots via update_profile, then generate_plan when ready. "
            "If they only want diet, set plan_mode=diet_only. "
            "If they want a yoga plan, set plan_mode=yoga_only."
        )

    # Mixed yoga+gym dump with no named technique — clarify channels, skip RAG.
    if _is_broad_mixed_workout_ask(query):
        session_store.set_exercises(thread_id, [])
        session_store.set_guideline_images(thread_id, [])
        return (
            "BROAD_REQUEST. The user mixed yoga and gym/workouts without naming "
            "a specific asana, pranayama, or move.\n"
            "Do NOT invent topics or call media=yoga_protocol until they name one.\n"
            "Offer briefly:\n"
            "  (A) Named yoga technique → intent=exercise_qa media=yoga_protocol\n"
            "  (B) Gym/bodyweight for a body region → intent=exercise_qa "
            "media=gym_catalog + focus_body_parts\n"
            "  (C) Full week plan → update_profile / generate_plan\n"
        )

    known_flags = [f for f in profile.health_flags if f != "none"] + profile.custom_health_notes
    # Soft demographic context for retrieval (age/gender do shape queries + calories)
    demo_bits = []
    if profile.age:
        demo_bits.append(f"age {profile.age}")
        if profile.age >= 60:
            demo_bits.append("older adult")
        elif profile.age >= 50:
            demo_bits.append("middle-aged adult")
    if profile.gender:
        demo_bits.append(profile.gender)
    demo_ctx = f" ({', '.join(demo_bits)})" if demo_bits else ""
    flag_ctx = f" (context: user has {', '.join(known_flags)})" if known_flags else ""
    augmented_query = f"{query}{demo_ctx}{flag_ctx}"
    rag_queries = [augmented_query]
    age_band = match_protocol_age_band(query)
    if age_band:
        # Explicit Fit India section title — lifts the correct protocol table
        # above nearby TOC / video-link pages.
        rag_queries.append(
            f"Yoga Protocol for {age_band} Years of Age Fit India practices rounds duration"
        )
    # Yoga technique expansions (OCR/phrasing bridges, e.g. Naukasana / boat pose).
    if media_mode == "yoga_protocol":
        from app.services.rag_retrieval import _clip_query_variants
        for variant in _clip_query_variants(query)[1:]:
            if variant not in rag_queries:
                rag_queries.append(variant)

    guideline_chunks = retrieve_multi_query(
        rag_queries, {"trust_tier__in": ["Tier 1", "Tier 2"]}, top_k_per_query=4
    )
    guideline_chunks = rerank_guideline_chunks(query, guideline_chunks)

    # ── Media: prefer SAME-PAGE images as grounded text, then CLIP hybrid ───
    guideline_images: List[dict] = []
    if media_mode == "yoga_protocol":
        try:
            # 1) Text-aligned: Naukasana lives on Fit India SRC003 p.18 — not CYP.
            guideline_images = images_from_matching_chunks(
                query, guideline_chunks, top_k=2,
            )
            if not guideline_images:
                # 2) CLIP + text-page anchors across CYP + Fit India demos.
                guideline_images = retrieve_guideline_images(
                    query, top_k=2, source_ids=["SRC009", "SRC003"],
                )
        except Exception as e:
            print(f"[RAG] guideline image retrieval failed: {e}")
            guideline_images = []
    session_store.set_guideline_images(thread_id, guideline_images)

    # When CLIP matched yoga/protocol photos, pull Technique text from those pages.
    if guideline_images:
        by_source: Dict[str, List[int]] = {}
        for img in guideline_images:
            sid = img.get("source_id")
            pg = img.get("page_number")
            if sid and pg is not None:
                by_source.setdefault(sid, []).append(int(pg))
        page_chunks = []
        for sid, pages in by_source.items():
            page_chunks.extend(retrieve_guidelines_by_pages(sid, pages, limit=6))
        if page_chunks:
            seen = {(c.get("source_id"), c.get("page_number"), (c.get("text") or "")[:80])
                    for c in guideline_chunks}
            for c in page_chunks:
                key = (c.get("source_id"), c.get("page_number"), (c.get("text") or "")[:80])
                if key not in seen:
                    guideline_chunks.append(c)
                    seen.add(key)

    # Gym GIFs only when the agent asked for gym_catalog (never mix with yoga photos).
    want_exercises = media_mode == "gym_catalog" and not guideline_images

    equipment = list(profile.available_equipment) if profile.available_equipment else None
    if query_wants_bodyweight(query):
        equipment = ["body only", "none"]

    # Agent-supplied body focus first; NLU match as fallback for gym search.
    mentioned: List[str] = []
    if focus_body_parts:
        for bp in focus_body_parts:
            key = (bp or "").strip().lower()
            if key == "core" or key == "abs":
                key = "waist"
            if key in _VALID_BODY_PARTS and key not in mentioned:
                mentioned.append(key)
    if not mentioned:
        mentioned = match_body_parts(query, threshold=0.40, top_n=2)
    body_parts = mentioned or None

    # Age soft-gates difficulty for exercise demos
    prefer_diff = profile.fitness_level or None
    if profile.age and profile.age >= 55 and not prefer_diff:
        prefer_diff = "beginner"
    elif profile.age and profile.age >= 55 and (prefer_diff or "").lower() == "expert":
        prefer_diff = "intermediate"

    exercise_hits: List[dict] = []
    if want_exercises:
        region = ", ".join(mentioned) if mentioned else ""
        ex_query = (
            f"{query}. Focus on practical exercises"
            + (f" for {region}" if region else "")
            + demo_ctx
        )
        # Hard injury filters — same map as plan SQL (never show contraindicated cards)
        injury_excl = injury_excluded_body_parts(profile.health_flags or [])
        flags_for_ban = [f for f in (profile.health_flags or []) if f != "none"]
        safe_body_parts = body_parts
        if body_parts and injury_excl:
            excl_l = {e.lower() for e in injury_excl}
            safe_body_parts = [b for b in body_parts if b.lower() not in excl_l] or None
        exercise_hits = [
            _normalize_exercise_for_ui(e)
            for e in retrieve_exercise_semantic(
                ex_query,
                top_k=4,
                equipment=equipment,
                body_parts=safe_body_parts,
                exclude_body_parts=injury_excl or None,
                health_flags=flags_for_ban or None,
                prefer_media=True,
                prefer_difficulty=prefer_diff,
                relax_filters_on_empty=bool(safe_body_parts),
            )
        ]
    session_store.set_exercises(thread_id, exercise_hits)

    parts = []
    parts.append(
        f"TURN_INTENT={intent}  MEDIA={media_mode}  "
        f"(info+none=guidelines; exercise_qa+gym_catalog=GIF demos; "
        f"yoga_protocol=booklet photos)"
    )
    if known_flags:
        parts.append(
            f"KNOWN USER HEALTH CONTEXT: {known_flags}. You MUST actively screen "
            f"out or clearly flag any exercise/activity below that's commonly "
            f"contraindicated for these conditions."
        )
        if want_exercises:
            excl = injury_excluded_body_parts(profile.health_flags or [])
            if excl:
                parts.append(
                    f"HARD FILTER APPLIED: excluded body parts {excl} and injury-related "
                    "move names from the demo list. Do not recommend those regions/moves."
                )
    known_bits = []
    if profile.age: known_bits.append(f"age={profile.age}")
    if profile.gender: known_bits.append(f"gender={profile.gender}")
    if profile.goal: known_bits.append(f"goal={profile.goal}")
    if profile.health_flags: known_bits.append(f"health_flags={profile.health_flags}")
    if profile.available_equipment: known_bits.append(f"equipment={profile.available_equipment}")
    if known_bits:
        parts.append("ALREADY KNOWN PROFILE (do not re-ask): " + "; ".join(known_bits))
    if profile.age or profile.gender:
        parts.append(
            "DEMOGRAPHICS: Weave age/sex into advice when relevant "
            "(e.g. older adults → lower impact; calorie needs differ by sex). "
            "Do not invent different exercise lists solely by gender."
        )

    if guideline_chunks:
        parts.append("GUIDELINE PASSAGES:")
        for c in guideline_chunks[:5]:
            body = (c.get("text") or "")[:1600]
            parts.append(f"- ({c['source_name']}, p.{c.get('page_number')}): {body}")
        parts.append(
            "If a passage is a yoga/fitness PROTOCOL TABLE (practices, rounds, "
            "durations), reproduce the full sequence from that passage — do not "
            "summarize it away as 'other practices'."
        )
    else:
        parts.append(
            "GUIDELINE PASSAGES: none retrieved for this query. Say you lack a "
            "grounded source; do not invent citations or page numbers."
        )

    if guideline_images:
        parts.append(
            "\nMATCHED DEMONSTRATION PHOTO(S) (the app will render these below your "
            "reply — do not describe them as missing, do not ask the user to imagine "
            "the pose, and do NOT suggest unrelated catalog exercises this turn):"
        )
        for img in guideline_images:
            parts.append(
                f"- {img['source_name']} p.{img['page_number']} "
                f"(match score {img['score']}): {img['caption'][:150]}"
            )
        parts.append(
            "When answering, give the step-by-step technique from the GUIDELINE "
            "PASSAGES above (Sthiti / Technique / Benefits). Cite the source and "
            "page. The photos below are from that protocol — tie your steps to them."
        )

    if guideline_images:
        parts.append(
            "\nYOGA/PROTOCOL DEMO TURN: teach the technique from passages + photos. "
            "If passages do not cover the named pose, say so and do not invent steps "
            "from another asana. Do not switch to gym catalog this turn."
        )
    elif intent == "info" or media_mode == "none":
        parts.append(
            "\nFACTUAL / NUTRITION / GUIDELINE QUESTION. Answer from passages only. "
            "Cite sources. Do NOT invent a workout or attach gym GIFs."
        )
    elif intent == "exercise_qa" or media_mode == "gym_catalog":
        parts.append(
            "\nTOPIC Q&A (form / body-region tips). Short grounded answer, then "
            "recommend ONLY listed exercises with one form cue each. Not a weekly plan."
        )

    if media_mode == "yoga_protocol" and not guideline_images:
        parts.append(
            "\nNo protocol demo photo matched confidently. Answer from text passages "
            "only; say if the named pose is not in the retrieved Common Yoga Protocol "
            "material. Do not invent photos or substitute gym GIFs."
        )

    if exercise_hits:
        parts.append("\nRELEVANT EXERCISES (UI shows GIF + instructions for each):")
        for e in exercise_hits:
            instr = (e.get("instructions") or e.get("description") or "")[:280]
            line = (
                f"- {e['name']} [{e.get('body_part')}/{e.get('equipment')}] "
                f"target={e.get('target_muscle') or 'n/a'}: {instr}"
            )
            if e.get("gif_url"):
                line += f" | gif={e['gif_url']}"
            if e.get("image_url"):
                line += f" | image={e['image_url']}"
            parts.append(line)
        parts.append(
            "Recommend these by name with one short form cue. Do NOT invent exercises "
            "that aren't listed. The app displays their GIFs and how-to text."
        )
    elif media_mode == "gym_catalog":
        parts.append(
            "\nNo catalog exercises matched filters — answer with guideline advice only "
            "and say you can widen search or build a plan if they share equipment."
        )

    if not parts:
        return "No grounded information found for this query. Answer cautiously."
    return "\n".join(parts)


@tool
def update_profile(
    goal: Optional[str] = None,
    target_body_parts: Optional[List[str]] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    health_flags: Optional[List[str]] = None,
    custom_health_notes: Optional[List[str]] = None,
    available_equipment: Optional[List[str]] = None,
    fitness_level: Optional[str] = None,
    time_per_day_minutes: Optional[int] = None,
    plan_mode: Optional[str] = None,
) -> str:
    """
    Store any profile facts the user just revealed, in this or any past
    message — including facts implied by HOW they asked a question, not
    just direct answers (e.g. if they ask "best exercises for women",
    call this with gender="female" right away; don't wait to ask separately).

    Valid values (using anything outside these lists gets rejected — check
    the tool's return message for exactly what was rejected and why):
      goal: lose_fat, build_muscle, improve_strength, improve_flexibility,
            improve_endurance, general_fitness, rehabilitation, stress_relief
      plan_mode: "full" (workout+diet), "diet_only" (meals only), or
            "yoga_only" (yoga/asana week + light diet guidance).
            If the user only wants a diet plan / no exercise, set
            plan_mode="diet_only" immediately — then body parts, equipment,
            fitness_level, and time_per_day are NOT required.
            If they ask for a yoga plan, set plan_mode="yoga_only"
            (equipment defaults to body only / mat).
            height_cm and weight_kg are preferred for accurate BMR. If the user
            does not know them, leave unset — the plan will use India age/sex
            average midpoints and mark calories as estimated.
      age: 10–100; gender: male|female; height_cm: 100–250; weight_kg: 25–300.
      target_body_parts: neck, shoulders, chest, back, upper arms, lower arms,
            waist, upper legs, lower legs, cardio — OR "full body" (auto-expands).
            Also map: core/abs → waist. Never ask the user to type exact tokens.
            SKIP asking for this when plan_mode=diet_only.
      available_equipment: body only, dumbbell, barbell, kettlebells, bands,
            cable, machine, exercise ball, foam roll, none
            SKIP when plan_mode=diet_only.
      fitness_level: beginner, intermediate, expert
      time_per_day_minutes: integer minutes; 0 is valid when diet_only / no exercise
      health_flags: high_bp, low_bp, diabetes, knee_injury, back_injury,
            shoulder_injury, wrist_injury, ankle_injury, heart_condition,
            asthma, osteoporosis, acidity, pregnancy, obesity, none
            (anything not on this list — e.g. "high cholesterol" — put
            in custom_health_notes as free text instead)

    If the user says they have no health conditions, pass
    health_flags=["none"] explicitly — this field can never be inferred
    as empty, it must be set.
    """
    profile = session_store.get_profile(_current_thread.get())
    result = profile.merge(
        goal=goal, target_body_parts=target_body_parts, age=age, gender=gender,
        height_cm=height_cm, weight_kg=weight_kg, health_flags=health_flags,
        custom_health_notes=custom_health_notes, available_equipment=available_equipment,
        fitness_level=fitness_level, time_per_day_minutes=time_per_day_minutes,
        plan_mode=plan_mode,
    )
    missing = profile.missing_fields()
    mode = profile.plan_mode
    msg = (
        f"Stored: {result['changed'] or 'nothing new'}. plan_mode={mode}. "
        f"Still missing: {missing or 'nothing — profile complete'}."
    )
    if mode == "diet_only":
        msg += " Diet-only mode: do NOT ask for body parts / equipment / fitness level / workout time."
    elif mode == "yoga_only":
        msg += (
            " Yoga-only mode: keep equipment as body only/mat; prefer flexibility/"
            "stress-relief goals; ask age, gender, health, fitness level, time, body focus."
        )
    if result["rejected"]:
        msg += (
            f" REJECTED (these did not match the valid schema — tell the user "
            f"in plain language and offer the actual valid options, don't just "
            f"repeat the same value): {result['rejected']}"
        )
    return msg


@tool
def get_profile_status() -> str:
    """
    Check what profile info you already have and what's still missing
    before deciding whether to ask another question or generate a plan.
    Call this before offering to generate a plan.
    """
    profile = session_store.get_profile(_current_thread.get())
    missing = profile.missing_fields()
    return f"Current profile: {profile.model_dump()}. Missing: {missing or 'none'}."


@tool
def generate_plan() -> str:
    """
    Generate a personalised plan from the profile collected so far.
    - plan_mode=full → 7-day workout + diet
    - plan_mode=diet_only → diet/nutrition plan only (no workouts)
    - plan_mode=yoga_only → 7-day yoga/asana practice + light diet guidance
    Only call once get_profile_status shows nothing missing AND the user
    wants a plan. Health flags/notes must be confirmed first.
    """
    thread_id = _current_thread.get()
    profile = session_store.get_profile(thread_id)
    if not profile.is_safe_to_plan():
        return (
            f"BLOCKED: profile is not complete/safe yet. Missing: {profile.missing_fields()}. "
            f"Ask the user for these before calling generate_plan again. "
            f"If they only want diet, set plan_mode=diet_only first (skips exercise slots). "
            f"If they want yoga, set plan_mode=yoga_only."
        )
    p = profile.to_dict()
    sql_filters = build_sql_filters(p) if profile.plan_mode != "diet_only" else {}
    rag_filters = build_rag_filters(p)
    plan = generate_plan_agentic(p, sql_filters, rag_filters)
    session_store.set_plan(thread_id, plan)
    stats = plan.get("stats") or {}
    enrichment_failed = bool(stats.get("enrichment_failed"))
    thin_program = bool(stats.get("thin_program"))
    n_ex = stats.get("distinct_exercises_in_split", 0)
    diet = plan.get("diet_plan") or {}
    diet_empty = not (diet.get("meals") or [])
    if profile.plan_mode == "diet_only":
        if enrichment_failed and diet_empty:
            return (
                "Diet-only plan FAILED: meal enrichment did not produce meals. "
                "Tell the user honestly and offer to retry generate_plan. "
                "Do NOT claim a diet plan was created."
            )
        warn = " (enrichment had issues — note any gaps)" if enrichment_failed else ""
        est = ""
        if stats.get("calories_estimated") or diet.get("calories_estimated"):
            est = (
                " Disclose that calorie targets used India age/sex average height/weight "
                "because measured size was not given."
            )
        return (
            f"Diet-only plan generated{warn} (no workout days).{est} Summarize the "
            "diet_plan, safety_notes and citations for the user. Ask if they'd like it "
            "saved and emailed (need name + email for save_generated_plan)."
        )
    if thin_program:
        return (
            f"Plan held in session but THIN PROGRAM: only {n_ex} distinct exercise(s) "
            "matched equipment/injury filters. Tell the user honestly — do NOT claim a "
            "full rich program. Summarize what exists, cite safety_notes, and offer to "
            "widen equipment or body parts then retry generate_plan. "
            "Ask about save/email only after they accept this limited plan."
        )
    est = ""
    if stats.get("calories_estimated") or (diet.get("calories_estimated")):
        ah = diet.get("assumed_height_cm")
        aw = diet.get("assumed_weight_kg")
        est = (
            f" Note: calorie targets used India age/sex averages"
            f"{f' (≈{ah} cm / ≈{aw} kg)' if ah and aw else ''} because height/weight "
            "were not provided — tell the user these are approximate and they can "
            "update measured height/weight anytime for a more accurate diet target."
        )
    warn = " Note: diet enrichment had issues." if enrichment_failed else ""
    return (
        f"Plan generated successfully ({n_ex} distinct exercises) and held in session.{warn}{est} "
        "Summarize the week_plan, diet_plan and safety_notes for the user in a friendly way, "
        "then ask if they'd like it saved and emailed (you'll need their name + email to "
        "call save_generated_plan)."
    )


@tool
def save_generated_plan(user_email: str, user_name: str) -> str:
    """
    Persist the most recently generated plan and email it to the user.
    Only call after generate_plan has succeeded and the user has given
    you their name and email and confirmed they want it saved.
    """
    thread_id = _current_thread.get()
    plan = session_store.get_plan(thread_id)
    if not plan:
        return "No plan has been generated yet in this session — call generate_plan first."
    profile = session_store.get_profile(thread_id)
    import json as _json
    result = save_plan(
        user_email=user_email, user_name=user_name,
        goal=profile.goal or "general_fitness",
        plan_json=_json.dumps(plan), constraints_json="{}",
    )
    if isinstance(result, dict) and result.get("success") and result.get("plan_id") is not None:
        session_store.set_saved_plan_id(thread_id, result["plan_id"])
    return str(result)


@tool
def send_reminder_email(to_email: str, user_name: str, plan_day_label: str,
                         exercises: List[str], reminder_type: str = "workout") -> str:
    """Send a one-off workout/diet reminder email. Confirm with the user first."""
    return str(send_reminder(
        to_email=to_email, user_name=user_name, plan_day_label=plan_day_label,
        exercises=", ".join(exercises), reminder_type=reminder_type,
    ))


@tool
def log_completed_workout(user_email: str, plan_id: int, exercise_names: List[str],
                           notes: str = "") -> str:
    """Log exercises the user says they completed."""
    return str(log_workout(
        user_email=user_email, plan_id=plan_id,
        exercise_names=", ".join(exercise_names), notes=notes, completed_at="",
    ))


@tool
def get_progress(user_email: str, days: int = 7) -> str:
    """Get the user's workout completion stats/streak for the last N days."""
    return str(get_workout_progress(user_email=user_email, days=days))


TOOLS = [
    answer_fitness_question, update_profile, get_profile_status,
    generate_plan, save_generated_plan, send_reminder_email,
    log_completed_workout, get_progress,
]

SYSTEM_PROMPT = """You are the Adaptive Fitness Planner, a friendly India-focused fitness assistant.

You choose tools AND (for Q&A) intent + media. Always pass intent and media
explicitly on answer_fitness_question — that is the agentic loop. Never use
media="auto" if you can decide. Never mix yoga_protocol with gym_catalog.

Routing — pick ONE lane per user message:

1) INFO — nutrition, hydration, WHO/ICMR facts, age-band protocol TABLES only
   → intent="info", media="none" (text guidelines; no photos/GIFs).

2) EXERCISE_QA — any technique / demo / "exercises for X"
   Call answer_fitness_question FIRST (do not open plan intake for tips).
   → Yoga / asana / pranayama / breathing technique (even a single word like
     "pranayama" or "tadasana", or "what is Naukasana"):
        intent="exercise_qa", media="yoga_protocol"
        Teach only from returned passages/photos; if missing, say so.
   → Gym / bodyweight ("squat", "arm fat", push-ups):
        intent="exercise_qa", media="gym_catalog", focus_body_parts when obvious.
   → Vague "yoga for beginners" with no technique → intent="info", media="none"
     OR ask which technique they want.
   → Mixed "yoga AND gym workouts" with no focus → clarify A/B/C or follow
     BROAD_REQUEST. Never invent body goals they did not mention.

3) PLAN — customised week / diet-only / yoga-only
   → Do NOT call answer_fitness_question for the plan itself.
   → update_profile + get_profile_status; ask 1–2 missing slots at a time.
   → generate_plan when slots complete; plan_mode=diet_only|yoga_only|full.

You may update_profile in parallel when Q&A reveals demographics, but still
answer via answer_fitness_question in the same turn.
When user says "yes" to demos, call the tool with yoga_protocol or gym_catalog.
INFER profile facts; never re-ask stored facts.
AGE & GENDER required before generate_plan; height/weight preferred for BMR.
IF update_profile rejects a value: remap (core→waist, full body, no equipment→body only).
NEVER fabricate a plan without successful generate_plan.
After generate_plan succeeds, summarize, then ask before save/email.
"""

_checkpointer = MemorySaver()
_agents: dict = {}


def get_agent(provider: Optional[str] = None):
    """Cache one ReAct agent per provider so Groq→Azure failover can retry cleanly."""
    p = resolve_provider(provider)
    if p not in _agents:
        _agents[p] = create_react_agent(
            get_llm(provider=p), TOOLS, checkpointer=_checkpointer, prompt=SYSTEM_PROMPT,
        )
    return _agents[p]


# ── Public API (kept compatible with routers/conversation.py) ────────────────

def start_conversation(thread_id: str) -> dict:
    greeting = (
        "Hi! I'm your Adaptive Fitness Planner. Ask me anything about fitness "
        "or nutrition, or tell me you'd like a personalised plan and I'll build "
        "one with you. What's on your mind?"
    )
    return {"thread_id": thread_id, "stage": "chatting", "message": greeting}


def process_user_message(user_message: str, thread_id: str) -> dict:
    _current_thread.set(thread_id)
    session_store.clear_exercises(thread_id)
    session_store.clear_guideline_images(thread_id)
    # Capture age/gender/diabetes from free text before the LLM turn so we
    # never re-ask for facts the user already stated.
    _auto_ingest_profile_hints(user_message, thread_id)
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"messages": [HumanMessage(content=user_message)]}

    primary = resolve_provider()
    try:
        result = get_agent(primary).invoke(payload, config=config)
    except Exception as e:
        if primary == "groq" and is_failover_error(e) and azure_configured():
            print(
                f"[LLM] Groq failed ({type(e).__name__}); "
                "retrying this turn with Azure OpenAI"
            )
            result = get_agent("azure").invoke(payload, config=config)
        else:
            raise

    last_ai = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content), None)
    reply = last_ai.content if last_ai else "Sorry, I didn't quite catch that — could you rephrase?"

    profile = session_store.get_profile(thread_id)
    plan = session_store.get_plan(thread_id)
    sql_filters = build_sql_filters(profile.to_dict()) if not profile.missing_fields() else None
    rag_filters = build_rag_filters(profile.to_dict()) if not profile.missing_fields() else None

    # Only show exercise cards when THIS turn retrieved them (via
    # answer_fitness_question / generate_plan). Never re-attach week_plan
    # day-0 GIFs onto unrelated Q&A (e.g. water → abs cards).
    exercises = [_normalize_exercise_for_ui(e) for e in session_store.get_exercises(thread_id)]
    guideline_images = session_store.get_guideline_images(thread_id)

    return {
        "thread_id": thread_id,
        "stage": "plan_ready" if plan else ("collecting" if profile.missing_fields() else "confirming"),
        "message": reply,
        "slots_complete": not profile.missing_fields(),
        "profile": profile.to_dict(),
        "plan": plan,
        "plan_id": session_store.get_saved_plan_id(thread_id),
        "sql_filters": sql_filters,
        "rag_filters": rag_filters,
        "exercises": exercises,
        "guideline_images": guideline_images,
    }
