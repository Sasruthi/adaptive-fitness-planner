"""
FILE LOCATION: backend/app/mcp_server/server.py

Adaptive Fitness Planner — MCP Tool Server
===========================================
Exposes 6 tools the LangGraph agent can call to take real actions.
Built with FastMCP (Python MCP SDK).

Tools:
  1. save_plan          — persist generated plan to DB, return plan_id
  2. log_workout        — mark exercises done, update completion tracking
  3. calculate_calories — Mifflin-St Jeor BMR + goal-adjusted target
  4. get_user_plan      — retrieve user's active plan
  5. get_workout_progress — weekly completion stats
  6. send_reminder      — email workout reminder via SMTP

Run standalone:
  python app/mcp_server/server.py

Or import and mount into FastAPI:
  from app.mcp_server.server import mcp
  app.mount("/mcp", mcp.get_asgi_app())
"""

import os, json, smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.models.models import Exercise, Plan, User, WorkoutLog

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

# ── DB connection ─────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "fitness.db"
engine  = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)

# ── MCP server instance ───────────────────────────────────────────────────────
mcp = FastMCP(
    name="adaptive-fitness-planner",
    instructions="""
You are the Adaptive Fitness Planner tool server.
Use these tools to save plans, track workouts, calculate nutrition targets,
and send reminders. Always confirm actions with the user before saving.
""",
)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — save_plan
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def save_plan(
    user_email: str,
    user_name: str,
    goal: str,
    plan_json: str,
    constraints_json: str = "{}",
) -> dict:
    """
    Save a generated fitness plan to the database.
    Creates a User record if one doesn't exist for this email.
    Deactivates any previous active plan for this user.

    Args:
        user_email:       User's email address (unique identifier)
        user_name:        User's display name
        goal:             Primary fitness goal (e.g. lose_fat, build_muscle)
        plan_json:        Full plan as JSON string (from plan_generator output)
        constraints_json: Profile/filter snapshot as JSON string

    Returns:
        dict with plan_id, user_id, message
    """
    session = Session()
    try:
        # Upsert user
        user = session.query(User).filter(User.email == user_email).first()
        if not user:
            user = User(email=user_email, name=user_name)
            session.add(user)
            session.flush()

        # Parse plan to extract profile fields if present
        plan_data = json.loads(plan_json)
        try:
            profile   = plan_data.get("profile_summary", {})
            if profile.get("age"):        user.age             = profile["age"]
            if profile.get("sex"):        user.sex             = profile["sex"]
            if profile.get("height_cm"):  user.height_cm       = profile["height_cm"]
            if profile.get("weight_kg"):  user.weight_kg       = profile["weight_kg"]
            # Persist Mifflin activity keys, never raw fitness_level strings
            act = profile.get("activity_level")
            if not act and profile.get("fitness_level"):
                act = {
                    "beginner": "lightly_active",
                    "intermediate": "moderately_active",
                    "expert": "very_active",
                }.get(str(profile["fitness_level"]).lower())
            if act:
                user.activity_level = act
            if profile.get("health_flags"):
                user.health_flags = json.dumps(profile["health_flags"])
        except (json.JSONDecodeError, KeyError):
            pass

        # Deactivate previous active plans
        session.query(Plan).filter(
            Plan.user_id == user.id,
            Plan.active == True
        ).update({"active": False})

        # Create new plan
        plan = Plan(
            user_id=user.id,
            goal=goal,
            plan_json=plan_json,
            constraints_json=constraints_json,
            active=True,
        )
        session.add(plan)
        session.commit()

        email_result = send_plan_ready_email(
            user_name=user_name,
            to_email=user_email,
            goal=goal,
            plan_data=plan_data,
        )

        return {
            "success":     True,
            "plan_id":     plan.id,
            "user_id":     user.id,
            "message":     f"Plan saved successfully! Plan ID: {plan.id}.",
            "email_sent":  email_result.get("success", False),
            "email_note":  email_result.get("message") if not email_result.get("success") else None,
        }
    except Exception as e:
        session.rollback()
        return {"success": False, "error": str(e)}
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — log_workout
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def log_workout(
    user_email: str,
    plan_id: int,
    exercise_names: str,
    notes: str = "",
    completed_at: str = "",
) -> dict:
    """
    Log completed exercises for a workout session.
    Looks up exercise IDs by name, creates WorkoutLog entries.

    Args:
        user_email:      User's email
        plan_id:         Plan ID (from save_plan)
        exercise_names:  Comma-separated exercise names completed
        notes:           Optional session notes (how it felt, weight used, etc.)
        completed_at:    ISO datetime string, defaults to now

    Returns:
        dict with logged count, streak info, encouragement message
    """
    session = Session()
    try:
        user = session.query(User).filter(User.email == user_email).first()
        if not user:
            return {"success": False, "error": "User not found. Save a plan first."}

        completed_dt = datetime.fromisoformat(completed_at) if completed_at else datetime.utcnow()
        names        = [n.strip() for n in exercise_names.split(",") if n.strip()]
        logged       = 0

        for name in names:
            # Find exercise by name (case-insensitive)
            exercise = session.query(Exercise).filter(
                Exercise.name.ilike(f"%{name}%")
            ).first()

            log = WorkoutLog(
                user_id     = user.id,
                plan_id     = plan_id,
                exercise_id = exercise.id if exercise else None,
                completed_at= completed_dt,
                notes       = notes,
            )
            session.add(log)
            logged += 1

        session.commit()

        # Calculate streak
        today     = datetime.utcnow().date()
        yesterday = today - timedelta(days=1)
        yesterday_logs = session.query(WorkoutLog).filter(
            WorkoutLog.user_id == user.id,
            WorkoutLog.completed_at >= datetime.combine(yesterday, datetime.min.time()),
            WorkoutLog.completed_at <  datetime.combine(today, datetime.min.time()),
        ).count()

        streak_msg = "Keep it up!" if yesterday_logs > 0 else "New streak started!"

        return {
            "success":        True,
            "logged_count":   logged,
            "exercises":      names,
            "completed_at":   completed_dt.isoformat(),
            "message":        f"Logged {logged} exercise(s) — {streak_msg} 💪",
        }
    except Exception as e:
        session.rollback()
        return {"success": False, "error": str(e)}
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — calculate_calories
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def calculate_calories(
    age: int,
    sex: str,
    weight_kg: float,
    height_cm: float,
    activity_level: str,
    goal: str,
) -> dict:
    """
    Calculate daily calorie target using Mifflin-St Jeor BMR.
    Adjusted for activity level and goal (deficit/surplus/maintenance).
    India-appropriate macronutrient split included.

    Args:
        age:            Age in years
        sex:            male | female
        weight_kg:      Weight in kilograms
        height_cm:      Height in centimetres
        activity_level: sedentary | lightly_active | moderately_active | very_active
        goal:           lose_fat | build_muscle | general_fitness | improve_endurance
                        improve_strength | improve_flexibility | rehabilitation | stress_relief

    Returns:
        dict with BMR, TDEE, target calories, macros, and Indian meal calorie guide
    """
    # Mifflin-St Jeor BMR
    if sex.lower() in ("male", "m"):
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

    # Activity multiplier (+ common aliases)
    raw_act = (activity_level or "").strip().lower().replace(" ", "_")
    act_aliases = {
        "moderate": "moderately_active",
        "light": "lightly_active",
        "active": "very_active",
        "veryactive": "very_active",
        "beginner": "lightly_active",
        "intermediate": "moderately_active",
        "expert": "very_active",
    }
    raw_act = act_aliases.get(raw_act, raw_act)
    multipliers = {
        "sedentary":         1.2,
        "lightly_active":    1.375,
        "moderately_active": 1.55,
        "very_active":       1.725,
    }
    tdee = bmr * multipliers.get(raw_act, 1.375)

    # Goal-based adjustment
    goal_key = (goal or "").lower()
    goal_adjustments = {
        "lose_fat":           -500,   # 0.5kg/week deficit
        "build_muscle":       +300,   # lean bulk surplus
        "improve_strength":   +200,   # slight surplus
        "improve_endurance":  +100,   # maintenance+ for fuel
        "general_fitness":    0,      # maintenance
        "improve_flexibility":0,
        "rehabilitation":     0,
        "stress_relief":      0,
    }
    adjustment     = goal_adjustments.get(goal_key, 0)
    target_calories= round(tdee + adjustment)

    # Macronutrient split (India-appropriate: higher carb from dal/roti/rice)
    if goal_key in ("build_muscle", "improve_strength"):
        protein_pct, carb_pct, fat_pct = 0.30, 0.45, 0.25
    elif goal_key == "lose_fat":
        protein_pct, carb_pct, fat_pct = 0.35, 0.40, 0.25
    else:
        protein_pct, carb_pct, fat_pct = 0.25, 0.50, 0.25

    protein_g = round((target_calories * protein_pct) / 4)
    carbs_g   = round((target_calories * carb_pct)    / 4)
    fat_g     = round((target_calories * fat_pct)     / 9)

    return {
        "success":          True,
        "bmr":              round(bmr),
        "tdee":             round(tdee),
        "goal_adjustment":  adjustment,
        "target_calories":  target_calories,
        "macros": {
            "protein_g": protein_g,
            "carbs_g":   carbs_g,
            "fat_g":     fat_g,
        },
        "india_meal_guide": {
            "breakfast":     round(target_calories * 0.25),
            "mid_morning":   round(target_calories * 0.10),
            "lunch":         round(target_calories * 0.35),
            "evening_snack": round(target_calories * 0.10),
            "dinner":        round(target_calories * 0.20),
        },
        "note": (
            f"Target: {target_calories} kcal/day "
            f"({'deficit' if adjustment < 0 else 'surplus' if adjustment > 0 else 'maintenance'} "
            f"of {abs(adjustment)} kcal). "
            f"Protein: {protein_g}g | Carbs: {carbs_g}g | Fat: {fat_g}g"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — get_user_plan
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_user_plan(user_email: str) -> dict:
    """
    Retrieve the user's current active plan.
    Returns the full plan JSON + metadata.

    Args:
        user_email: User's email address

    Returns:
        dict with plan details or not-found message
    """
    session = Session()
    try:
        user = session.query(User).filter(User.email == user_email).first()
        if not user:
            return {
                "success": False,
                "message": "No user found with this email. Start a conversation to create a plan.",
            }

        plan = session.query(Plan).filter(
            Plan.user_id == user.id,
            Plan.active  == True,
        ).order_by(desc(Plan.created_at)).first()

        if not plan:
            return {
                "success": False,
                "message": "No active plan found. Complete the intake questionnaire to generate one.",
            }

        plan_data = json.loads(plan.plan_json) if plan.plan_json else {}

        return {
            "success":      True,
            "plan_id":      plan.id,
            "goal":         plan.goal,
            "created_at":   plan.created_at.isoformat(),
            "week_plan":    plan_data.get("week_plan", []),
            "diet_plan":    plan_data.get("diet_plan", {}),
            "safety_notes": plan_data.get("safety_notes", []),
            "weekly_tips":  plan_data.get("weekly_tips", []),
            "citations":    plan_data.get("citations", []),
        }
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 5 — get_workout_progress
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def get_workout_progress(
    user_email: str,
    days: int = 7,
) -> dict:
    """
    Get workout completion stats for the past N days.
    Returns daily breakdown, total sessions, and streak.

    Args:
        user_email: User's email
        days:       Number of days to look back (default 7)

    Returns:
        dict with completion stats, streak, and motivational summary
    """
    session = Session()
    try:
        user = session.query(User).filter(User.email == user_email).first()
        if not user:
            return {"success": False, "message": "User not found."}

        since = datetime.utcnow() - timedelta(days=days)
        logs  = session.query(WorkoutLog).filter(
            WorkoutLog.user_id     == user.id,
            WorkoutLog.completed_at >= since,
        ).order_by(WorkoutLog.completed_at).all()

        # Group by date
        by_date: dict = {}
        for log in logs:
            date_key = log.completed_at.date().isoformat()
            if date_key not in by_date:
                by_date[date_key] = []
            ex = session.query(Exercise).filter(Exercise.id == log.exercise_id).first()
            by_date[date_key].append(ex.name if ex else "Unknown exercise")

        # Calculate streak (consecutive days with at least 1 workout)
        streak = 0
        check_date = datetime.utcnow().date()
        while True:
            if check_date.isoformat() in by_date:
                streak += 1
                check_date -= timedelta(days=1)
            else:
                break

        total_sessions   = len(by_date)
        total_exercises  = len(logs)
        completion_rate  = round((total_sessions / days) * 100)

        # Motivational message
        if completion_rate >= 80:
            motivation = "Excellent consistency! You're crushing it 🔥"
        elif completion_rate >= 50:
            motivation = "Good progress — keep showing up! 💪"
        elif completion_rate > 0:
            motivation = "Every session counts — let's pick up the pace! 🌱"
        else:
            motivation = "Time to start! Your first workout is the hardest one. 🚀"

        return {
            "success":          True,
            "period_days":      days,
            "total_sessions":   total_sessions,
            "total_exercises":  total_exercises,
            "completion_rate":  f"{completion_rate}%",
            "current_streak":   streak,
            "daily_breakdown":  by_date,
            "motivation":       motivation,
        }
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
# Email helpers
# ══════════════════════════════════════════════════════════════════════════════

def _send_email(to_email: str, subject: str, html_body: str) -> dict:
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").replace(" ", "").strip()
    from_name = "Adaptive Fitness Planner"

    if not smtp_user or not smtp_pass:
        return {
            "success": False,
            "message": "SMTP not configured — set SMTP_USER and SMTP_PASS in .env (Gmail needs an App Password).",
        }

    msg            = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"{from_name} <{smtp_user}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        return {"success": True, "message": f"Email sent to {to_email}"}
    except Exception as e:
        err = str(e)
        hint = ""
        if "535" in err or "Authentication" in err:
            hint = " Check SMTP_PASS — Gmail requires a 16-character App Password, not your login password."
        return {"success": False, "message": f"Email failed: {err}.{hint}"}


def send_plan_ready_email(user_name: str, to_email: str, goal: str, plan_data: dict) -> dict:
    """Send a summary email when a plan is saved."""
    goal_label = (goal or "fitness").replace("_", " ").title()
    workout_days = sum(1 for d in plan_data.get("week_plan", []) if d.get("type") == "workout")
    diet = plan_data.get("diet_plan", {})
    calories = diet.get("daily_calories_estimate", "")

    day_lines = []
    for day in plan_data.get("week_plan", [])[:7]:
        if day.get("type") == "workout":
            names = ", ".join(e.get("name", "") for e in day.get("exercises", [])[:4])
            day_lines.append(f"<li><strong>{day.get('label', 'Workout')}</strong> — {names}</li>")
    schedule_html = "".join(day_lines) or "<li>7-day personalised schedule</li>"

    html = f"""
<!DOCTYPE html><html><body style="font-family:Arial,sans-serif;max-width:600px;margin:auto;background:#f9f9f9;padding:20px">
  <div style="background:#2c7a4b;color:white;padding:20px;border-radius:8px 8px 0 0;text-align:center">
    <h2 style="margin:0">Adaptive Fitness Planner</h2>
    <p style="margin:4px 0;opacity:0.9">Your plan is ready</p>
  </div>
  <div style="background:white;padding:24px;border-radius:0 0 8px 8px">
    <p>Hi {user_name},</p>
    <p>Your <strong>{goal_label}</strong> plan is saved — <strong>{workout_days} workout days</strong> this week
       {f'at around <strong>{calories} kcal/day</strong>' if calories else ''}.</p>
    <p><strong>This week:</strong></p>
    <ul>{schedule_html}</ul>
    <p style="color:#666;font-size:13px">Open the app and tap <strong>Plan</strong> to see full exercises, diet details, and GIF demos.</p>
  </div>
</body></html>"""

    return _send_email(to_email, f"Your {goal_label} plan is ready", html)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 6 — send_reminder
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def send_reminder(
    to_email: str,
    user_name: str,
    plan_day_label: str,
    exercises: str,
    reminder_type: str = "workout",
) -> dict:
    """
    Send a workout or diet reminder email.
    Uses SMTP (configure SMTP_HOST/SMTP_USER/SMTP_PASS in .env).

    Args:
        to_email:       Recipient email
        user_name:      User's first name for personalisation
        plan_day_label: E.g. "Day 3 — Wednesday: Upper Arms"
        exercises:      Comma-separated exercise names for today
        reminder_type:  "workout" | "diet" | "weekly_summary"

    Returns:
        dict with sent status and message
    """
    subject, body = _build_email_content(
        user_name, plan_day_label, exercises, reminder_type
    )
    return _send_email(to_email, subject, body)


def _build_email_content(user_name, plan_day_label, exercises, reminder_type):
    subject = {
        "workout":        f"💪 Today's workout: {plan_day_label}",
        "diet":           f"🥗 Your nutrition reminder",
        "weekly_summary": f"📊 Your weekly fitness summary",
    }.get(reminder_type, f"Your fitness reminder — {plan_day_label}")

    body = _build_email_body(user_name, plan_day_label, exercises, reminder_type)
    return subject, body


def _build_email_body(user_name, plan_day_label, exercises, reminder_type):
    ex_list = "".join(
        f"<li style='padding:4px 0'>{e.strip()}</li>"
        for e in exercises.split(",") if e.strip()
    )

    return f"""
<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; 
             background: #f9f9f9; padding: 20px;">

  <div style="background: #2c7a4b; color: white; padding: 20px; 
              border-radius: 8px 8px 0 0; text-align: center;">
    <h2 style="margin:0">Adaptive Fitness Planner</h2>
    <p style="margin:4px 0; opacity:0.9">Your personal workout & nutrition coach</p>
  </div>

  <div style="background: white; padding: 24px; border-radius: 0 0 8px 8px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.1);">

    <h3 style="color: #2c7a4b;">Hi {user_name}! 👋</h3>

    <p>Time for <strong>{plan_day_label}</strong>. Here are your exercises for today:</p>

    <div style="background: #f0faf4; border-left: 4px solid #2c7a4b; 
                padding: 16px; border-radius: 4px; margin: 16px 0;">
      <ul style="margin:0; padding-left: 20px; color: #333;">
        {ex_list}
      </ul>
    </div>

    <p style="color: #555;">Remember:</p>
    <ul style="color: #555;">
      <li>Warm up for 5 minutes before starting</li>
      <li>Stay hydrated — drink water between sets</li>
      <li>Listen to your body — rest if something hurts</li>
    </ul>

    <div style="text-align: center; margin: 24px 0;">
      <p style="font-size: 18px; color: #2c7a4b; font-weight: bold;">
        Consistency is the key. You've got this! 💪
      </p>
    </div>

    <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
    <p style="color: #999; font-size: 12px; text-align: center;">
      Adaptive Fitness Planner · Powered by India-first health guidelines
    </p>
  </div>
</body>
</html>"""


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Adaptive Fitness Planner MCP Server...")
    print("Tools available:")
    for tool in ["save_plan", "log_workout", "calculate_calories",
                 "get_user_plan", "get_workout_progress", "send_reminder"]:
        print(f"  ✅ {tool}")
    mcp.run(transport="stdio")
