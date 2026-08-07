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
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.conversation.profile_store import session_store, Profile
from app.conversation.state import build_sql_filters, build_rag_filters
from app.services.rag_retrieval import retrieve_multi_query
from app.services.exercise_rag import retrieve_exercise_semantic
from app.services.plan_agent import generate_plan_agentic
from app.mcp_server.server import (
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

def _query_wants_bodyweight(q: str) -> bool:
    from app.services.semantic_nlu import query_wants_bodyweight
    return query_wants_bodyweight(q)


def _query_is_diet_focused(q: str) -> bool:
    """True when the user is asking about food/nutrition — not exercise demos."""
    from app.services.semantic_nlu import query_is_diet_focused
    return query_is_diet_focused(q)


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


@tool
def answer_fitness_question(query: str) -> str:
    """
    Answer a factual fitness/nutrition/exercise question (NOT a full weekly plan).

    Routing (done in code — you do not choose):
      info        → guideline passages only (water, macros, WHO/ICMR facts)
      exercise_qa → guidelines + a few targeted exercise demos (e.g. lose arm fat)
      plan        → do NOT use this tool; use update_profile / generate_plan instead

    For diet/nutrition/hydration: never attach random workouts.
    """
    thread_id = _current_thread.get()
    profile = session_store.get_profile(thread_id)
    from app.services.semantic_nlu import classify_turn_intent, match_body_parts

    intent = classify_turn_intent(query)
    # Plan intent should not be answered as a one-shot Q&A with exercise cards
    if intent == "plan":
        session_store.set_exercises(thread_id, [])
        return (
            "INTENT=plan. The user wants a customised plan, not a one-off tip. "
            "Do NOT list random exercises. Call get_profile_status, ask only for "
            "missing slots via update_profile, then generate_plan when ready. "
            "If they only want diet, set plan_mode=diet_only."
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

    guideline_chunks = retrieve_multi_query(
        [augmented_query], {"trust_tier__in": ["Tier 1", "Tier 2"]}, top_k_per_query=4
    )

    want_exercises = intent == "exercise_qa"

    equipment = list(profile.available_equipment) if profile.available_equipment else None
    if _query_wants_bodyweight(query):
        equipment = ["body only", "none"]

    # Prefer body parts named IN THIS question (e.g. arm fat → upper arms)
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
        # Build a focused exercise query from the user ask + body region
        region = ", ".join(mentioned) if mentioned else ""
        ex_query = (
            f"{query}. Focus on practical exercises"
            + (f" for {region}" if region else "")
            + demo_ctx
        )
        exercise_hits = [
            _normalize_exercise_for_ui(e)
            for e in retrieve_exercise_semantic(
                ex_query,
                top_k=4,
                equipment=equipment,
                body_parts=body_parts,
                prefer_media=True,
                prefer_difficulty=prefer_diff,
                relax_filters_on_empty=bool(body_parts),  # if region filter empty, widen once
            )
        ]
    session_store.set_exercises(thread_id, exercise_hits)

    parts = []
    parts.append(f"TURN_INTENT={intent}  (info=guidelines only; exercise_qa=guidelines+demos)")
    if known_flags:
        parts.append(
            f"KNOWN USER HEALTH CONTEXT: {known_flags}. You MUST actively screen "
            f"out or clearly flag any exercise/activity below that's commonly "
            f"contraindicated for these conditions."
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
            parts.append(f"- ({c['source_name']}, p.{c.get('page_number')}): {c['text'][:350]}")
    else:
        parts.append(
            "GUIDELINE PASSAGES: none retrieved for this query. Say you lack a "
            "grounded source; do not invent citations or page numbers."
        )

    if intent == "info":
        parts.append(
            "\nUSER ASKED A FACTUAL / NUTRITION / GUIDELINE QUESTION. Answer from "
            "passages only. Cite sources. Do NOT invent a workout, do NOT list "
            "exercise names, do NOT attach GIFs as a substitute for the answer."
        )
    elif intent == "exercise_qa":
        parts.append(
            "\nUSER ASKED A SPECIFIC FITNESS TOPIC (e.g. lose arm fat, form tips). "
            "Give a short grounded answer from passages, then recommend ONLY the "
            "listed exercises below with one form cue each. This is NOT a weekly plan — "
            "do not invent a 7-day schedule. Offer to build a full plan if they want one."
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
    elif intent == "exercise_qa":
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
      plan_mode: "full" (workout+diet) OR "diet_only" (meals/nutrition only).
            If the user says they only want a diet plan / no exercise, set
            plan_mode="diet_only" immediately — then body parts, equipment,
            fitness_level, and time_per_day are NOT required.
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
    Only call once get_profile_status shows nothing missing AND the user
    wants a plan. Health flags/notes must be confirmed first.
    """
    thread_id = _current_thread.get()
    profile = session_store.get_profile(thread_id)
    if not profile.is_safe_to_plan():
        return (
            f"BLOCKED: profile is not complete/safe yet. Missing: {profile.missing_fields()}. "
            f"Ask the user for these before calling generate_plan again. "
            f"If they only want diet, set plan_mode=diet_only first (skips exercise slots)."
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

Every user message is ONE of three intents — treat them differently:

1) INFO (fact / nutrition / hydration / guideline question)
   → Call answer_fitness_question. Answer from document passages only.
   → Do NOT show workouts, do NOT start plan intake, do NOT invent a schedule.

2) EXERCISE_QA (topic tip, e.g. "how to lose arm fat", "exercises for back pain")
   → Call answer_fitness_question. It returns guidelines + a few targeted demos.
   → Summarize the tip and name those demos. This is NOT a weekly plan.
   → Offer a full plan only if they ask.

3) PLAN (wants a customised week plan / diet plan / workout plan)
   → Do NOT dump exercise GIFs as the answer.
   → update_profile + get_profile_status; ask 1–2 missing slots at a time.
   → generate_plan only when slots are complete and health flags confirmed.
   → plan_mode=diet_only if they only want meals (skips exercise slots).

INFER, DON'T JUST WAIT TO BE ASKED: "exercises for women" → gender=female;
"my knee injury" → health_flags; "54 year old male diabetic" → age+gender+flags.
Never re-ask facts already stored.

AGE & GENDER & SIZE: age + sex are required before generate_plan.
Ask for height_cm + weight_kg when possible (better BMR). If the user does
not know / skips size, proceed anyway — calories use India age/sex midpoints
and must be disclosed as approximate. Never invent a different formula.

IF update_profile REJECTS A VALUE: map it yourself (core→waist, full body,
no equipment→body only). Never make the user type exact system tokens.

NEVER FABRICATE A PLAN. If generate_plan hasn't succeeded, you have no plan —
don't invent a 7-day schedule in chat.

After generate_plan succeeds, summarize warmly, then ask before save/email.
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

    return {
        "thread_id": thread_id,
        "stage": "plan_ready" if plan else ("collecting" if profile.missing_fields() else "confirming"),
        "message": reply,
        "slots_complete": not profile.missing_fields(),
        "profile": profile.to_dict(),
        "plan": plan,
        "sql_filters": sql_filters,
        "rag_filters": rag_filters,
        "exercises": exercises,
    }
