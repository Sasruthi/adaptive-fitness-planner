"""
Test Module 4 — Plan Generation Engine
Uses mock GPT-4o so no Azure credentials needed.
Tests: SQL retrieval, RAG retrieval, scheduling, enrichment, final plan structure.
"""
import sys, json
sys.path.insert(0, '/home/claude/fitness-app/backend')
from unittest.mock import patch, MagicMock

MOCK_ENRICHMENT = {
    "exercise_modifications": {
        "impossible dips": "Keep range of motion controlled. Avoid locking elbows at top. Safe for high BP.",
        "cable squatting curl": "Substitute with wall-supported curl to avoid Valsalva."
    },
    "day_notes": {
        "Monday — Upper Arms": "Focus on form over speed. Breathe out on exertion.",
        "Wednesday — Upper Arms": "Slightly increase reps if Monday felt easy.",
        "Friday — Upper Arms": "End of week push — maintain good posture throughout.",
    },
    "diet_plan": {
        "daily_calories_estimate": 1550,
        "calorie_note": "Based on age 28, female, sedentary-moderate activity for fat loss (ICMR-NIN 2024, p.45).",
        "meals": [
            {"meal":"breakfast","suggestions":["2 idli with sambar","Oats upma with vegetables","Poha with peanuts"],"timing":"7:00-8:00 AM","notes":"Avoid sugar in tea/coffee. Use skimmed milk."},
            {"meal":"mid_morning_snack","suggestions":["1 banana","Handful of roasted chana"],"timing":"10:30 AM","notes":""},
            {"meal":"lunch","suggestions":["2 roti + dal + sabzi + salad","Brown rice + rajma + cucumber raita","Khichdi with ghee + mixed vegetables"],"timing":"1:00-2:00 PM","notes":"Fill half plate with vegetables first."},
            {"meal":"evening_snack","suggestions":["Sprouts chaat","Coconut water"],"timing":"5:00-6:00 PM","notes":"Avoid packaged snacks."},
            {"meal":"dinner","suggestions":["2 roti + moong dal + palak sabzi","Vegetable soup + 1 roti","Daliya khichdi"],"timing":"7:00-8:00 PM","notes":"Eat at least 2 hours before sleep."}
        ],
        "india_specific_tips": [
            "Use pressure cooker to retain nutrients in dal and vegetables.",
            "Prefer sesame or mustard oil over refined oils — rich in PUFA (FSSAI Eat Right).",
            "Buttermilk (chaas) is an excellent low-calorie afternoon drink, recommended by NIN."
        ],
        "foods_to_avoid": ["Packaged namkeen — high sodium worsens BP","Deep fried snacks — excess saturated fat","Pickles and papad — high salt content"],
        "hydration": "2.5-3 litres of water per day. Coconut water counts toward hydration goal."
    },
    "safety_notes": [
        {"flag":"high_bp","note":"Avoid heavy isometric holds and Valsalva manoeuvre. Keep rep pace controlled. Monitor BP before and after workout.","citation":"WHO Physical Activity Guidelines 2020, p.24"},
        {"flag":"high_bp","note":"DASH-style diet recommended: low sodium, high potassium foods (banana, spinach, dal).","citation":"ICMR-NIN Dietary Guidelines 2024"}
    ],
    "citations": [
        {"source":"ICMR-NIN Dietary Guidelines 2024","page":45,"trust_tier":"Tier 1","used_for":"Calorie estimate and macronutrient ratios"},
        {"source":"FSSAI Do You Eat Right","page":77,"trust_tier":"Tier 1","used_for":"Oil type recommendation and salt reduction"},
        {"source":"WHO Physical Activity Guidelines 2020","page":24,"trust_tier":"Tier 2","used_for":"BP safety during exercise"}
    ],
    "weekly_tips": [
        "Track your workouts in a notebook — seeing progress is motivating.",
        "Sleep 7-8 hours. Growth and fat loss both happen during sleep.",
        "Weigh yourself once a week, same time, same conditions."
    ]
}

def mock_llm_invoke(messages):
    r = MagicMock()
    r.content = json.dumps(MOCK_ENRICHMENT)
    return r

print("="*60)
print("MODULE 4 — Plan Generation Engine Test")
print("="*60)

with patch('app.services.plan_generator.AzureChatOpenAI') as MockLLM:
    MockLLM.return_value.invoke.side_effect = mock_llm_invoke
    from app.services.plan_generator import generate_plan

    # Exact profile + filters that Module 3 produces
    profile = {
        "goal":                 "lose_fat",
        "target_body_parts":    ["upper arms"],
        "age":                  28,
        "sex":                  "female",
        "height_cm":            162.0,
        "weight_kg":            60.0,
        "health_flags":         ["high_bp"],
        "available_equipment":  ["none"],
        "fitness_level":        "beginner",
        "time_per_day_minutes": 30,
    }
    sql_filters = {
        "body_part__in":     ["upper arms"],
        "equipment__in":     ["body only","Body Only","Bodyweight"],
        "difficulty_level":  "Beginner",
    }
    rag_filters = {
        "trust_tier__in":   ["Tier 1","Tier 2"],
        "content_type__in": ["safety_medical","guideline","exercise"],
        "category__in":     ["nutrition","exercise"],
    }

    plan = generate_plan(profile, sql_filters, rag_filters)

# ── Verify output structure ───────────────────────────────────────────────────
print(f"\n✅ plan_id:       {plan['plan_id']}")
print(f"✅ generated_at:  {plan['generated_at']}")
print(f"✅ stats:         {plan['stats']}")

print(f"\n📅 WEEK PLAN ({len(plan['week_plan'])} days):")
for day in plan["week_plan"]:
    tag = "🏋️" if day["type"]=="workout" else "😴"
    excount = len(day["exercises"])
    print(f"  {tag} Day {day['day']}: {day['label']} — {excount} exercises — {day['duration_minutes']}min")
    if excount:
        ex = day["exercises"][0]
        print(f"     Sample: {ex['name']} | {ex['sets']}x{ex['reps']} | rest={ex['rest_seconds']}s | gif={bool(ex['gif_url'])}")
        if ex.get("modification"):
            print(f"     ⚠️  Mod: {ex['modification'][:80]}")

print(f"\n🥗 DIET PLAN:")
print(f"  Calories: {plan['diet_plan'].get('daily_calories_estimate')} kcal/day")
for meal in plan["diet_plan"].get("meals", []):
    print(f"  {meal['meal']:20s} {meal['timing']} → {meal['suggestions'][0]}")

print(f"\n⚠️  SAFETY NOTES ({len(plan['safety_notes'])}):")
for note in plan["safety_notes"]:
    print(f"  [{note['flag']}] {note['note'][:80]}...")
    print(f"  Citation: {note['citation']}")

print(f"\n📚 CITATIONS ({len(plan['citations'])}):")
for c in plan["citations"]:
    print(f"  {c['source']} | p.{c['page']} | {c['trust_tier']} | {c['used_for']}")

print(f"\n💡 WEEKLY TIPS:")
for t in plan["weekly_tips"]:
    print(f"  • {t}")

print(f"\n✅ Module 4 — Plan Generation Engine: COMPLETE")
print(f"   Total exercises in plan: {plan['stats']['exercises_retrieved']}")
print(f"   Guideline chunks used:   {plan['stats']['guideline_chunks_used']}")
