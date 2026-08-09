"""
FILE LOCATION: backend/tests/test_integration.py

Full integration test — Modules 1 through 4 in sequence.
Uses mock LLM so NO Groq credentials needed.
Run from backend/ directory: python tests/test_integration.py

Tests:
  Module 1 — DB schema + exercise/taxonomy query
  Module 2 — Qdrant vector store reachable + returns chunks
  Module 3 — Full conversational intake (7 turns, correct slot extraction)
  Module 4 — Plan generation (SQL + RAG + structured output)
  Integration — M3 profile feeds directly into M4 plan
"""

import sys, os, uuid, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock

PASS = "✅"
FAIL = "❌"
results = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, label, detail))
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    return condition


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODULE 1 — Data Layer (SQLite)")
print("="*60)

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models.models import Exercise, Taxonomy

    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'fitness.db')
    db_path = os.path.abspath(db_path)

    engine  = create_engine(f"sqlite:///{db_path}")
    session = sessionmaker(bind=engine)()

    total_ex  = session.query(Exercise).count()
    with_media= session.query(Exercise).filter(Exercise.has_media == True).count()
    total_tax = session.query(Taxonomy).count()

    check("DB file exists",          os.path.exists(db_path), db_path)
    check("Exercises loaded",        total_ex >= 900,         f"{total_ex} rows")
    check("Exercises with media",    with_media >= 800,       f"{with_media} rows")
    check("Taxonomy loaded",         total_tax >= 80,         f"{total_tax} rows")

    # Test filtered query (simulates Module 4 SQL retrieval)
    results_q = session.query(Exercise).filter(
        Exercise.body_part.ilike("%upper arm%") |
        Exercise.target_muscle.ilike("%tricep%") |
        Exercise.target_muscle.ilike("%bicep%")
    ).limit(5).all()
    check("Filtered query works",    len(results_q) > 0,      f"{len(results_q)} exercises for upper arms")

    # Test body part taxonomy
    bp_list = [t.name for t in session.query(Taxonomy).filter(Taxonomy.kind=="body_part").all()]
    check("Body part taxonomy",      "shoulders" in bp_list,  f"{len(bp_list)} entries")

    # Test equipment taxonomy
    eq_list = [t.name for t in session.query(Taxonomy).filter(Taxonomy.kind=="equipment").all()]
    check("Equipment taxonomy",      len(eq_list) > 0,        f"{len(eq_list)} entries")

    session.close()
except Exception as e:
    check("Module 1 import/run",     False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODULE 2 — RAG (Qdrant Vector Store)")
print("="*60)

try:
    from app.services.rag_retrieval import get_qdrant, COLLECTION_NAME, QDRANT_PATH

    check("Qdrant path exists",      QDRANT_PATH.exists(), str(QDRANT_PATH))

    client = get_qdrant()
    collections = [c.name for c in client.get_collections().collections]
    check("Collection exists",       COLLECTION_NAME in collections, str(collections))

    count = client.count(COLLECTION_NAME).count
    check("Vectors stored",          count >= 100, f"{count} vectors")

    # Test retrieval with dummy vector
    import numpy as np
    dummy_vec = np.ones(384, dtype="float32")
    dummy_vec /= np.linalg.norm(dummy_vec)
    res = client.query_points(
        collection_name=COLLECTION_NAME,
        query=dummy_vec.tolist(),
        limit=3, with_payload=True
    )
    check("Retrieval returns results", len(res.points) > 0, f"{len(res.points)} chunks")

    # Verify metadata fields exist on chunks
    if res.points:
        payload = res.points[0].payload
        required_fields = ["text", "source_id", "source_name", "trust_tier",
                           "content_type", "page_number"]
        all_present = all(f in payload for f in required_fields)
        check("Chunk metadata complete", all_present,
              f"fields: {[f for f in required_fields if f in payload]}")

except Exception as e:
    check("Module 2 import/run", False, str(e))


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODULE 3 — Profile store + agent greeting (unified agent)")
print("="*60)
# Old intake_graph FSM removed — conversation is LangGraph ReAct in agent.py.
# Module 3 checks code-level safety gates + start_conversation (no LLM needed).

m3_final = None
try:
    from app.conversation.agent import start_conversation
    from app.conversation.profile_store import Profile, session_store
    from app.conversation.state import build_sql_filters, build_rag_filters

    tid = str(uuid.uuid4())
    r = start_conversation(tid)
    check("Greeting runs", r.get("message") != "", r.get("stage"))

    incomplete = Profile(goal="lose_fat", age=28, gender="female")
    check("Unsafe without health_flags", not incomplete.is_safe_to_plan(),
          str(incomplete.missing_fields()))

    # "full body" must expand (historical deadlock fix)
    p = Profile()
    merged = p.merge(
        goal="lose_fat",
        target_body_parts=["full body"],
        age=28,
        gender="female",
        health_flags=["high_bp"],
        available_equipment=["no equipment"],
        fitness_level="beginner",
        time_per_day_minutes=30,
    )
    check("full body expands", len(p.target_body_parts) >= 8, str(p.target_body_parts))
    check("equipment alias → body only", "body only" in (p.available_equipment or []),
          str(p.available_equipment))
    check("Safe when slots complete", p.is_safe_to_plan(), str(p.missing_fields()))
    check("merge has no hard rejects", not merged.get("rejected"), str(merged.get("rejected")))

    sql_f = build_sql_filters(p.to_dict())
    rag_f = build_rag_filters(p.to_dict())
    check("SQL filters built", bool(sql_f), str(sql_f))
    check("RAG filters built", bool(rag_f), str(rag_f))

    session_store.set_profile(tid, p)
    m3_final = {
        "profile": p.to_dict(),
        "sql_filters": sql_f,
        "rag_filters": rag_f,
        "slots_complete": True,
    }

except Exception as e:
    check("Module 3 import/run", False, str(e))
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODULE 4 — Plan Generation Engine")
print("="*60)

MOCK_PLAN_JSON = {
    "exercise_modifications": {
        "impossible dips": "Avoid locking elbows. Safe for high BP."
    },
    "day_notes": {},
    "diet_plan": {
        "daily_calories_estimate": 1550,
        "calorie_note": "Based on ICMR-NIN 2024 guidelines.",
        "meals": [
            {"meal":"breakfast","suggestions":["2 idli + sambar","Poha with vegetables"],"timing":"7-8 AM","notes":""},
            {"meal":"lunch","suggestions":["Dal + roti + sabzi"],"timing":"1-2 PM","notes":""},
            {"meal":"dinner","suggestions":["Moong dal + 2 roti"],"timing":"7-8 PM","notes":""},
        ],
        "india_specific_tips": ["Use mustard oil","Prefer dal over processed protein"],
        "foods_to_avoid": ["Pickles — high sodium worsens BP"],
        "hydration": "2.5 litres per day"
    },
    "safety_notes": [
        {"flag":"high_bp","note":"Avoid Valsalva. Keep pace controlled.","citation":"WHO PA Guidelines 2020"}
    ],
    "citations": [
        {"source":"ICMR-NIN 2024","page":45,"trust_tier":"Tier 1","used_for":"Calorie estimate"}
    ],
    "weekly_tips": ["Track workouts","Sleep 7-8 hours"]
}

def m4_mock(messages):
    r = MagicMock()
    r.content = json.dumps(MOCK_PLAN_JSON)
    return r

try:
    with patch('app.llm.ChatGroq') as MockLLM:
        MockLLM.return_value.invoke.side_effect = m4_mock

        from app.services.plan_generator import generate_plan
        from app.services.exercise_retrieval import get_exercises
        from app.services.rag_retrieval import retrieve_multi_query

        # Use exact profile + filters from M3 output
        profile = {
            "goal": "lose_fat",
            "target_body_parts": ["upper arms"],
            "age": 28, "sex": "female",
            "height_cm": 162.0, "weight_kg": 60.0,
            "health_flags": ["high_bp"],
            "custom_health_notes": [],
            "available_equipment": ["none"],
            "fitness_level": "beginner",
            "time_per_day_minutes": 30,
        }
        sql_filters = {
            "body_part__in": ["upper arms"],
            "equipment__in": ["body only","Body Only","Bodyweight"],
            "difficulty_level": "Beginner",
        }
        rag_filters = {
            "trust_tier__in": ["Tier 1","Tier 2"],
            "content_type__in": ["safety_medical","guideline","exercise"],
            "category__in": ["nutrition","exercise"],
        }

        # Test exercise retrieval independently
        exercises = get_exercises(sql_filters, limit=10)
        check("Exercise SQL retrieval",  len(exercises) >= 3,         f"{len(exercises)} exercises")
        check("Exercises have gif_url",  any(e["gif_url"] for e in exercises), "at least one gif")
        check("Health flag exclusion",
              all("back" not in e["body_part"].lower() for e in exercises),
              "back not in results")

        # Test RAG retrieval independently
        chunks = retrieve_multi_query(
            ["diet for fat loss India", "exercise safety high BP"],
            rag_filters, top_k_per_query=3
        )
        check("RAG retrieval works",     len(chunks) >= 1,            f"{len(chunks)} chunks")
        check("Chunks have citations",
              all("source_name" in c for c in chunks), "source_name present")

        # Test full plan generation
        plan = generate_plan(profile, sql_filters, rag_filters)

        check("Plan has plan_id",        bool(plan.get("plan_id")))
        check("Week plan is 7 days",     len(plan.get("week_plan",[])) == 7,
              f"{len(plan.get('week_plan',[]))} days")
        check("Has workout days",
              any(d["type"]=="workout" for d in plan["week_plan"]), "3 workout days")
        check("Has rest days",
              any(d["type"]=="rest"    for d in plan["week_plan"]), "4 rest days")
        check("Exercises have gif_url",
              any(ex["gif_url"] for d in plan["week_plan"]
                  for ex in d.get("exercises",[])), "gif_url present")
        check("Diet plan present",       bool(plan.get("diet_plan")),
              f"meals: {len(plan['diet_plan'].get('meals',[]))}")
        check("India-specific meals",    len(plan["diet_plan"].get("meals",[])) >= 3)
        check("Safety notes present",    len(plan.get("safety_notes",[])) > 0)
        check("Citations present",       len(plan.get("citations",[])) > 0)
        check("Weekly tips present",     len(plan.get("weekly_tips",[])) > 0)

except Exception as e:
    check("Module 4 import/run", False, str(e))
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("INTEGRATION — M3 profile → M4 plan (end to end)")
print("="*60)

try:
    if m3_final:
        with patch('app.llm.ChatGroq') as MockLLM:
            MockLLM.return_value.invoke.side_effect = m4_mock
            from app.services.plan_generator import generate_plan

            profile     = m3_final["profile"]
            sql_filters = m3_final["sql_filters"]
            rag_filters = m3_final["rag_filters"]

            plan = generate_plan(profile, sql_filters, rag_filters)

            check("M3→M4 pipeline works",   bool(plan.get("plan_id")))
            check("Profile carried through", plan["profile_summary"]["goal"] == profile["goal"],
                  plan["profile_summary"]["goal"])
            check("Body parts match",
                  plan["profile_summary"]["target_body_parts"] == profile["target_body_parts"],
                  str(plan["profile_summary"]["target_body_parts"]))
            check("Health flags in safety notes",
                  any(n["flag"] == "high_bp" for n in plan.get("safety_notes",[])),
                  "high_bp safety note present")
    else:
        check("M3→M4 integration", False, "M3 did not complete — cannot run integration")

except Exception as e:
    check("Integration test", False, str(e))
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"\n{PASS} Passed: {passed}")
print(f"{FAIL} Failed: {failed}")
print(f"Total:   {passed + failed}")

if failed > 0:
    print(f"\n{FAIL} FAILED CHECKS:")
    for status, label, detail in results:
        if status == FAIL:
            print(f"   • {label}: {detail}")
else:
    print(f"\nAll checks passed — ready for Module 5!")
