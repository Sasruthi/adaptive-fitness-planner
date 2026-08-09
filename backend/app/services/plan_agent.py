"""
FILE LOCATION: backend/app/services/plan_agent.py

LangGraph Plan Agent — the RIGHT use of LangGraph in this project.
=================================================================
This is where LangGraph belongs: an agentic loop where the LLM
decides which tools to call, evaluates the result, and loops
until the plan is complete and safe.

NOT used for turn-by-turn conversation (wrong tool for that).
USED for plan generation and revision (correct tool for this).

Agent loop:
  1. Receives user profile + filters
  2. Calls retrieve_exercises tool
  3. Calls retrieve_guidelines tool
  4. Calls calculate_calories tool
  5. Synthesizes a structured plan
  6. Calls validate_plan tool — checks exercises are safe
  7. If validation fails → loops back to retrieve better exercises
  8. If validation passes → returns final plan

This means:
  - If a health flag causes all exercises to be excluded → agent
    automatically widens the search and tries again
  - If guidelines retrieval returns nothing relevant → agent
    reformulates the query and retries
  - Plan generation is self-correcting, not brittle
"""

import os, json
from typing import TypedDict, List, Optional, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage

from app.services.exercise_retrieval import get_exercises
from app.services.exercise_rag import retrieve_exercise_semantic
from app.services.exercise_selection import BODY_PART_QUERY_HINTS, resolve_split
from app.services.rag_retrieval import retrieve_multi_query
from app.tools.actions import calculate_calories

load_dotenv()


# ── LLM with tool-calling (Groq → Azure failover when Azure is configured) ────
def get_llm_with_tools(tools):
    from app.llm import get_llm_with_tools as _shared_get_llm_with_tools
    return _shared_get_llm_with_tools(tools, temperature=0.3, max_tokens=4000)


# ══════════════════════════════════════════════════════════════════════════════
# TOOLS — what the agent can call
# ══════════════════════════════════════════════════════════════════════════════

@tool
def fetch_exercises_for_day_type(
    day_type: str,
    equipment: List[str],
    difficulty: str,
    exclude_body_parts: List[str] = [],
    limit: int = 8,
) -> str:
    """
    Retrieve ONE exercise set for a single day type — either a specific
    body part from target_body_parts (e.g. "waist", "lower legs") or
    "Full Body". Matches by MEANING (semantic search over exercise
    name/description/muscle), not exact tag matching — so it still finds
    the right exercises even if the catalog's own tagging is inconsistent
    (e.g. a row tagged "calves" still matches a "lower legs" day type).

    Call this ONCE PER DAY TYPE in the split, not once per calendar day.
    Reuse the exact same returned list every time that day type recurs
    in the week — do not call this again or substitute different
    exercises for a repeat occurrence of the same day type.

    Returns a JSON list of exercises with name, body_part, equipment, gif_url.
    """
    has_no_equip = not equipment or "none" in [e.lower() for e in equipment]
    equipment_note = "bodyweight only, no equipment" if has_no_equip else f"using {', '.join(equipment)}"
    hint = ("full body toning and conditioning covering all major muscle groups"
            if day_type.lower() == "full body"
            else BODY_PART_QUERY_HINTS.get(day_type.lower(), day_type))
    query = f"{hint}, {difficulty} level, {equipment_note}"

    hits = retrieve_exercise_semantic(
        query,
        top_k=limit * 2,
        equipment=equipment,
        prefer_media=True,
        prefer_difficulty=difficulty,
    )
    if exclude_body_parts:
        excl = {e.lower() for e in exclude_body_parts}
        hits = [h for h in hits if (h.get("body_part") or "").lower() not in excl]

    if not hits:
        # Safety-net SQL fallback — not the primary matcher
        sql_filters = {"equipment__in": equipment, "difficulty_level": difficulty}
        if day_type.lower() != "full body":
            sql_filters["body_part__in"] = [day_type]
        if exclude_body_parts:
            sql_filters["exclude_body_parts"] = exclude_body_parts
        hits = get_exercises(sql_filters, limit=limit)

    if not hits:
        return json.dumps({
            "exercises": [],
            "count": 0,
            "hint": (
                "No exercises matched. Retry with broader equipment "
                "(include 'body only'), fewer exclude_body_parts, or a different day_type."
            ),
        })

    compact = [{
        "name":         ex["name"],
        "body_part":    ex.get("body_part",""),
        "target_muscle":ex.get("target_muscle",""),
        "equipment":    ex.get("equipment",""),
        "difficulty":   ex.get("difficulty",""),
        "gif_url":      ex.get("gif_url",""),
        "image_url":    ex.get("image_url", ""),
        "video_url":    ex.get("video_url", ""),
        "has_media":    ex.get("has_media", False),
    } for ex in hits[:limit]]
    return json.dumps({"exercises": compact, "count": len(compact)})


@tool
def fetch_guidelines(
    queries: List[str],
    content_types: List[str] = ["nutrition", "guideline", "exercise", "safety_medical"],
    trust_tiers: List[str] = ["Tier 1", "Tier 2"],
) -> str:
    """
    Retrieve relevant guideline passages from India-first health documents
    (ICMR-NIN, FSSAI, Fit India, WHO) using semantic search.
    Use this to ground diet and safety recommendations in real sources.
    Returns relevant text passages with source citations.
    """
    rag_filters = {
        "trust_tier__in":   trust_tiers,
        "content_type__in": content_types,
    }
    chunks = retrieve_multi_query(queries, rag_filters, top_k_per_query=3)
    compact = [{
        "text":        c["text"][:400],
        "source":      c["source_name"],
        "page":        c["page_number"],
        "trust_tier":  c["trust_tier"],
        "content_type":c["content_type"],
    } for c in chunks[:8]]
    if not compact:
        return json.dumps({
            "passages": [],
            "count": 0,
            "hint": "No guideline hits — reformulate queries or widen content_types/trust_tiers.",
        })
    return json.dumps({"passages": compact, "count": len(compact)})


@tool
def get_calorie_target(
    age: int,
    gender: str,
    weight_kg: float,
    height_cm: float,
    activity_level: str,
    goal: str,
) -> str:
    """
    Calculate daily calorie target and macronutrient split
    using Mifflin-St Jeor BMR adjusted for goal and activity level.
    activity_level: sedentary | lightly_active | moderately_active | very_active
    (aliases accepted: light, moderate, active, beginner, intermediate, expert).
    Returns calorie target, protein/carb/fat grams, and India meal guide.
    """
    from app.services.plan_generator import _resolve_activity_level
    mapped = _resolve_activity_level(activity_level, activity_level)
    result = calculate_calories(
        age=age, sex=gender,
        weight_kg=weight_kg, height_cm=height_cm,
        activity_level=mapped, goal=goal,
    )
    return json.dumps(result)


@tool
def validate_plan(
    exercises: List[str],
    health_flags: List[str],
    goal: str,
    time_per_day_minutes: int,
) -> str:
    """
    Validate that the selected exercises are safe and appropriate.
    Checks: health flag exclusions, exercise count vs time, goal alignment.
    Returns {"valid": true/false, "issues": ["..."], "suggestions": ["..."]}
    """
    issues = []
    suggestions = []

    # Health flag safety checks
    flag_exclusions = {
        "knee_injury":     ["squat","lunge","jump","running","leg press","step"],
        "back_injury":     ["deadlift","barbell row","good morning","hyperextension"],
        "shoulder_injury": ["overhead press","military press","upright row","dip"],
        "wrist_injury":    ["push-up","plank","dumbbell curl","barbell curl"],
        "high_bp":         ["heavy deadlift","max squat","isometric hold","valsalva"],
        "pregnancy":       ["crunch","sit-up","heavy compound","supine"],
    }

    exercise_names_lower = [e.lower() for e in exercises]
    for flag in health_flags:
        if flag in flag_exclusions:
            blocked = flag_exclusions[flag]
            for ex in exercise_names_lower:
                if any(b in ex for b in blocked):
                    issues.append(f"'{ex}' may not be safe for {flag.replace('_',' ')}")
                    suggestions.append(f"Replace '{ex}' with a {flag.replace('_',' ')}-safe alternative")

    # Exercise count check
    expected_count = {15: 3, 30: 5, 45: 7, 60: 9}.get(time_per_day_minutes, 5)
    if len(exercises) < expected_count - 1:
        issues.append(f"Only {len(exercises)} exercises for {time_per_day_minutes}-minute session (need ~{expected_count})")
        suggestions.append(f"Fetch {expected_count - len(exercises)} more exercises")

    return json.dumps({
        "valid":       len(issues) == 0,
        "issues":      issues,
        "suggestions": suggestions,
        "exercise_count": len(exercises),
        "expected_count": expected_count,
    })


# ── Agent state ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages:     Annotated[List[BaseMessage], lambda x, y: x + y]
    profile:      dict
    final_plan:   Optional[dict]
    iterations:   int


# ── Agent node — LLM decides what tools to call ───────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    """The LLM reasons about what tools to call next."""
    profile = state["profile"]
    tools   = [fetch_exercises_for_day_type, fetch_guidelines, get_calorie_target, validate_plan]
    llm     = get_llm_with_tools(tools)

    # Split is derived directly from target_body_parts — no goal override.
    # Uses the SAME resolve_split() as the deterministic fallback path so
    # both paths produce the same program shape for the same profile.
    split = resolve_split(profile.get("target_body_parts", []))
    day_types = split["day_types"]

    # Build explicit equipment constraint for the agent prompt
    equip = profile.get("available_equipment", ["none"])
    has_no_equip = "none" in [e.lower() for e in equip] or not equip
    equip_instruction = (
        "CRITICAL: User has NO equipment. You MUST pass equipment=['body only','Body Only','None','bodyweight'] "
        "to fetch_exercises_for_day_type. NEVER fetch barbell, dumbbell, cable, or machine exercises."
        if has_no_equip else
        f"User has: {equip}. Always include bodyweight options too."
    )

    if not state["messages"]:
        workout_days = 6
        day_type_cycle = [day_types[i % len(day_types)] for i in range(workout_days)]

        system = f"""You are an expert India-focused fitness planner.
Generate a complete, safe, personalised 7-day workout and diet plan.

User Profile:
- Goal: {profile.get('goal','').replace('_',' ')}
- Focus areas: {', '.join(profile.get('target_body_parts', []))}
- Age/Gender: {profile.get('age')} / {profile.get('gender','?')}
- Health flags: {', '.join(profile.get('health_flags', ['none']))}
- Equipment: {', '.join(equip)}
- Fitness level: {profile.get('fitness_level','beginner')}
- Time per day: {profile.get('time_per_day_minutes', 30)} minutes

{equip_instruction}

THE SPLIT (already derived from target_body_parts — do not change it):
Day types for this plan: {', '.join(day_types)}
This week has {workout_days} workout days and 1 rest day, cycling through day types in
this exact order: {' -> '.join(day_type_cycle)}
Prefer beginner-tagged (or untagged) exercises when fitness_level is beginner —
pass difficulty="{profile.get('fitness_level','beginner')}" to fetch_exercises_for_day_type.
NEVER include planche, muscle-up, front/back lever, or other advanced gymnastics for beginners.
NEVER list an exercise and then say to avoid it — only fetch appropriate moves.

CRITICAL — this is a real training program, not a random daily grab-bag:
- Call fetch_exercises_for_day_type EXACTLY ONCE per distinct day type
  listed above (not once per calendar day).
- When a day type recurs later in the week (per the cycle order), REUSE
  the exact same exercise list you already fetched for it. Do not
  substitute, add, or invent different exercises for a repeat occurrence
  of the same day type — a real program repeats its exercises across
  the week so the user can track progress on the same lifts.
- Only sets/reps/rest/intensity may vary by goal/level — never which
  exercises are assigned to a given day type.

Steps:
1. Call fetch_exercises_for_day_type once per day type above — MUST respect equipment constraint.
   Tool returns {{"exercises":[...],"count":N}} (or count=0 + hint — then widen and retry).
2. Call fetch_guidelines for diet + safety advice — returns {{"passages":[...],"count":N}}.
3. Call get_calorie_target if age/weight/height available (map activity: beginner→lightly_active).
4. Call validate_plan to check exercises are safe
5. If validation fails → re-fetch that day type with corrected parameters
6. Synthesize final plan as JSON with keys:
   week_plan, diet_plan, safety_notes, citations, weekly_tips, profile_summary
   — each workout day's "exercises" must be the reused list for its day type
   — safety_notes must be objects: {{"flag","note","citation"}}

All food must be India-appropriate (dal, roti, sabzi, idli, rice, etc.)
"""
        state["messages"] = [SystemMessage(content=system)]

    response = llm.invoke(state["messages"])
    return {"messages": [response], "iterations": state.get("iterations", 0) + 1}


def should_continue(state: AgentState) -> str:
    """Route: continue calling tools OR extract final plan."""
    last = state["messages"][-1]
    max_iter = 6  # prevent infinite loops

    if state.get("iterations", 0) >= max_iter:
        return "extract_plan"

    # If last message has tool calls → run tools
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    # Otherwise agent is done → extract plan
    return "extract_plan"


def extract_plan_node(state: AgentState) -> AgentState:
    """
    Parse agent draft JSON only — never call generate_plan here.
    generate_plan_agentic always re-grounds once via _build_fallback_plan;
    calling it here would double-run RAG + enrichment.
    """
    import re

    last_text = ""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and isinstance(msg.content, str) and len(msg.content) > 100:
            last_text = msg.content
            break

    plan = None
    text = re.sub(r'```json\s*', '', last_text)
    text = re.sub(r'```\s*', '', text)
    try:
        plan = json.loads(text.strip())
    except Exception:
        pass

    if not plan:
        matches = re.findall(r'\{.*\}', last_text, re.DOTALL)
        for m in sorted(matches, key=len, reverse=True):
            try:
                plan = json.loads(m)
                break
            except Exception:
                continue

    if not plan or "week_plan" not in (plan or {}):
        return {"final_plan": None}
    return {"final_plan": plan}


def _build_fallback_plan(profile: dict, sql_filters: dict = None, rag_filters: dict = None) -> dict:
    """
    Direct pipeline fallback — always respects sql_filters.
    Used when LangGraph agent fails or returns malformed JSON.
    """
    from app.services.plan_generator import generate_plan
    from app.conversation.state import build_sql_filters, build_rag_filters

    if sql_filters is None:
        sql_filters = build_sql_filters(profile)
    if rag_filters is None:
        rag_filters = build_rag_filters(profile)
    return generate_plan(profile, sql_filters, rag_filters)


# ── Build the graph ───────────────────────────────────────────────────────────
def build_plan_agent():
    tools    = [fetch_exercises_for_day_type, fetch_guidelines, get_calorie_target, validate_plan]
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("agent",        agent_node)
    graph.add_node("tools",        tool_node)
    graph.add_node("extract_plan", extract_plan_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools":        "tools",
        "extract_plan": "extract_plan",
    })
    graph.add_edge("tools", "agent")       # tools always return to agent
    graph.add_edge("extract_plan", END)

    return graph.compile()


_plan_agent = None

def get_plan_agent():
    global _plan_agent
    if _plan_agent is None:
        _plan_agent = build_plan_agent()
    return _plan_agent


# ── Public API ────────────────────────────────────────────────────────────────
def generate_plan_agentic(profile: dict, sql_filters: dict = None, rag_filters: dict = None) -> dict:
    """
    Public plan entry used by the conversation agent.

    Always returns the deterministic pipeline (sql_filters, injury exclusions,
    Mifflin BMR, INDB meal grounding) — never raw agent JSON.

    PLAN_AGENTIC=1 runs the LangGraph retrieve/validate loop first (tool-calling
    warm-up / logging). Draft JSON is discarded; generate_plan runs exactly once.
    """
    import os

    if os.getenv("PLAN_AGENTIC", "0") != "1":
        return _build_fallback_plan(profile, sql_filters, rag_filters)

    if (profile.get("plan_mode") or "full") == "diet_only":
        return _build_fallback_plan(profile, sql_filters, rag_filters)

    try:
        agent = get_plan_agent()
        result = agent.invoke({
            "messages": [],
            "profile": profile,
            "final_plan": None,
            "iterations": 0,
        })
        draft = result.get("final_plan") or {}
        if draft.get("week_plan"):
            n = sum(len(d.get("exercises") or []) for d in draft["week_plan"])
            print(f"[PlanAgent] Agentic draft parsed ({n} exercise slots) — re-grounding once via generate_plan")
        else:
            print("[PlanAgent] No usable agent draft — generate_plan once")
    except Exception as e:
        print(f"[PlanAgent] Agent failed: {e} — generate_plan once")

    return _build_fallback_plan(profile, sql_filters, rag_filters)
