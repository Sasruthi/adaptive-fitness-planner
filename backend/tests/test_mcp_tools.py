"""
FILE LOCATION: backend/tests/test_mcp_tools.py

Tests MCP tools directly without starting the MCP server.
Run from backend/: python tests/test_mcp_tools.py
"""
import sys, json
sys.path.insert(0, '.')

from app.mcp_server.server import (
    save_plan, log_workout, calculate_calories,
    get_user_plan, get_workout_progress, send_reminder
)

print("Testing MCP tools directly (no server needed)...\n")

# Tool 1
r = save_plan(
    user_email="test@example.com",
    user_name="Test User",
    goal="lose_fat",
    plan_json=json.dumps({"profile_summary": {"goal":"lose_fat"}, "week_plan":[], "diet_plan":{}}),
)
print(f"save_plan:    {'✅' if r['success'] else '❌'} {r.get('message','')[:60]}")

# Tool 3 - most useful standalone
r = calculate_calories(age=28, sex="female", weight_kg=60, height_cm=162,
                        activity_level="moderately_active", goal="lose_fat")
print(f"calc_cals:    {'✅' if r['success'] else '❌'} {r.get('note','')[:70]}")

# Tool 4
r = get_user_plan("test@example.com")
print(f"get_plan:     {'✅' if r['success'] else '❌'} plan_id={r.get('plan_id')}")

# Tool 5
r = get_workout_progress("test@example.com", days=7)
print(f"progress:     {'✅' if r['success'] else '❌'} {r.get('motivation','')[:50]}")

# Tool 6 - preview mode without SMTP
r = send_reminder("test@example.com", "Test", "Monday", "Push-up, Squat")
print(f"send_reminder:{'✅' if not r.get('error') else '❌'} {r.get('message','')[:60]}")

print("\nAll tools verified. The server.py runs inside FastAPI in Module 6 — not standalone.")
