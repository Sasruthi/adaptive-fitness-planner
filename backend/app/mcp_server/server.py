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

import os, json, smtplib, mimetypes, html as html_lib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

# Cap embedded demo media so Gmail (~25MB) and SMTP stay happy.
_EMAIL_MAX_INLINE_BYTES = int(os.getenv("EMAIL_MAX_INLINE_BYTES", str(14 * 1024 * 1024)))
_EMAIL_MAX_SINGLE_FILE = int(os.getenv("EMAIL_MAX_SINGLE_FILE_BYTES", str(2 * 1024 * 1024)))
_BACKEND_ROOT = Path(__file__).resolve().parents[2]  # backend/


def _send_email(
    to_email: str,
    subject: str,
    html_body: str,
    inline_images: Optional[List[Dict[str, Any]]] = None,
) -> dict:
    """
    Send HTML email. Optional inline_images: [{cid, path, data, mime}] —
    embedded as MIME parts and referenced from HTML as cid:<cid>.
    """
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

    inline_images = inline_images or []

    # multipart/related so clients show cid: images inside the HTML body
    related = MIMEMultipart("related")
    related["Subject"] = subject
    related["From"] = f"{from_name} <{smtp_user}>"
    related["To"] = to_email

    alt = MIMEMultipart("alternative")
    related.attach(alt)
    alt.attach(MIMEText(
        "Your Adaptive Fitness Planner plan is attached as HTML. "
        "Open this message in an HTML mail client to see demos and meals.",
        "plain",
    ))
    alt.attach(MIMEText(html_body, "html"))

    attached = 0
    for item in inline_images:
        cid = (item.get("cid") or "").strip()
        if not cid:
            continue
        data = item.get("data")
        path = item.get("path")
        mime = item.get("mime") or "application/octet-stream"
        filename = item.get("filename") or f"{cid}.bin"
        try:
            if data is None and path:
                data = Path(path).read_bytes()
            if not data:
                continue
            maintype, _, subtype = mime.partition("/")
            if maintype == "image":
                part = MIMEImage(data, _subtype=subtype or "gif")
            else:
                part = MIMEBase(maintype or "application", subtype or "octet-stream")
                part.set_payload(data)
                encoders.encode_base64(part)
            part.add_header("Content-ID", f"<{cid}>")
            part.add_header("Content-Disposition", "inline", filename=filename)
            related.attach(part)
            attached += 1
        except Exception as e:
            print(f"[Email] skip inline {cid}: {e}")

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, related.as_string())
        note = f" ({attached} demo image(s) embedded)" if attached else ""
        return {"success": True, "message": f"Email sent to {to_email}{note}"}
    except Exception as e:
        err = str(e)
        hint = ""
        if "535" in err or "Authentication" in err:
            hint = " Check SMTP_PASS — Gmail requires a 16-character App Password, not your login password."
        return {"success": False, "message": f"Email failed: {err}.{hint}"}


def _esc(value: Any) -> str:
    return html_lib.escape("" if value is None else str(value))


def _resolve_local_media_path(url: Optional[str]) -> Optional[Path]:
    """Map /static/... (or absolute file URL path) to a file under backend/static."""
    if not url:
        return None
    u = str(url).strip()
    if u.startswith("http://") or u.startswith("https://"):
        # Only rewrite our own static paths if someone stored absolute localhost URLs
        for marker in ("/static/",):
            if marker in u:
                u = marker + u.split(marker, 1)[1]
                break
        else:
            return None
    if not u.startswith("/static/"):
        return None
    path = (_BACKEND_ROOT / u.lstrip("/")).resolve()
    try:
        path.relative_to((_BACKEND_ROOT / "static").resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


class _InlineMediaRegistry:
    """Dedupes demo files and assigns Content-IDs for cid: embedding."""

    def __init__(self) -> None:
        self._by_path: Dict[str, str] = {}
        self.items: List[Dict[str, Any]] = []
        self.total_bytes = 0
        self.skipped = 0

    def register(self, url: Optional[str]) -> Optional[str]:
        path = _resolve_local_media_path(url)
        if path is None:
            return None
        key = str(path)
        if key in self._by_path:
            return self._by_path[key]
        try:
            size = path.stat().st_size
        except OSError:
            self.skipped += 1
            return None
        if size <= 0 or size > _EMAIL_MAX_SINGLE_FILE:
            self.skipped += 1
            return None
        if self.total_bytes + size > _EMAIL_MAX_INLINE_BYTES:
            self.skipped += 1
            return None
        try:
            data = path.read_bytes()
        except OSError:
            self.skipped += 1
            return None
        mime, _ = mimetypes.guess_type(str(path))
        if not mime:
            mime = "image/gif" if path.suffix.lower() == ".gif" else "image/jpeg"
        cid = f"demo{len(self.items)}"
        self._by_path[key] = cid
        self.items.append({
            "cid": cid,
            "path": key,
            "data": data,
            "mime": mime,
            "filename": path.name,
        })
        self.total_bytes += size
        return cid


def _meal_label(meal_key: str) -> str:
    return {
        "breakfast": "Breakfast",
        "mid_morning_snack": "Mid-morning snack",
        "lunch": "Lunch",
        "evening_snack": "Evening snack",
        "dinner": "Dinner",
    }.get((meal_key or "").lower(), (meal_key or "Meal").replace("_", " ").title())


def _build_full_plan_email_html(
    user_name: str, goal: str, plan_data: dict,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Full plan email HTML + inline image parts (raw file bytes with Content-IDs).
    GIF/JPEG demos are embedded in the message — not remote URL paths.
    """
    media_reg = _InlineMediaRegistry()
    goal_label = _esc((goal or "fitness").replace("_", " ").title())
    week = plan_data.get("week_plan") or []
    diet = plan_data.get("diet_plan") or {}
    safety = plan_data.get("safety_notes") or []
    citations = plan_data.get("citations") or []
    tips = plan_data.get("weekly_tips") or []
    profile = plan_data.get("profile_summary") or {}
    plan_mode = plan_data.get("plan_mode") or profile.get("plan_mode") or "full"

    workout_days = sum(1 for d in week if d.get("type") == "workout")
    calories = diet.get("daily_calories_estimate") or diet.get("target_calories") or ""
    macros = diet.get("macros") or {}

    # ── Profile chips ────────────────────────────────────────────────────────
    profile_bits = []
    for key, label in (
        ("goal", "Goal"),
        ("fitness_level", "Level"),
        ("age", "Age"),
        ("gender", "Sex"),
        ("time_per_day_minutes", "Min/day"),
    ):
        if profile.get(key) not in (None, "", []):
            val = profile[key]
            if key == "goal":
                val = str(val).replace("_", " ")
            if key == "time_per_day_minutes":
                val = f"{val} min"
            profile_bits.append(
                f'<span style="display:inline-block;background:#e8f5ee;color:#1b5e3b;'
                f'padding:4px 10px;border-radius:12px;font-size:12px;margin:2px">'
                f"<strong>{_esc(label)}:</strong> {_esc(val)}</span>"
            )
    if profile.get("target_body_parts"):
        parts = ", ".join(str(p) for p in profile["target_body_parts"])
        profile_bits.append(
            f'<span style="display:inline-block;background:#e8f5ee;color:#1b5e3b;'
            f'padding:4px 10px;border-radius:12px;font-size:12px;margin:2px">'
            f"<strong>Focus:</strong> {_esc(parts)}</span>"
        )
    profile_html = (" ".join(profile_bits)) if profile_bits else ""

    # ── Workout / yoga week ──────────────────────────────────────────────────
    day_blocks: List[str] = []
    for day in week[:7]:
        label = _esc(day.get("label") or "Day")
        dtype = (day.get("type") or "workout").lower()
        note = _esc(day.get("note") or day.get("day_note") or "")
        if dtype == "rest":
            note_html = (
                f'<p style="color:#666;font-size:13px;margin:0">{note}</p>' if note else ""
            )
            day_blocks.append(
                f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin:12px 0">'
                f'<h3 style="margin:0 0 6px;color:#2c7a4b">{label} — Rest</h3>'
                f"{note_html}"
                f"</div>"
            )
            continue

        ex_rows = []
        for ex in day.get("exercises") or []:
            name = _esc(ex.get("name") or "Exercise")
            body = _esc(ex.get("body_part") or "")
            equip = _esc(ex.get("equipment") or "")
            sets = ex.get("sets")
            reps = ex.get("reps")
            dur = ex.get("duration_sec") or ex.get("duration")
            meta = " · ".join(
                x for x in [
                    body,
                    equip,
                    f"{sets} sets" if sets else "",
                    f"{reps} reps" if reps else "",
                    f"{dur}s" if dur else "",
                ] if x
            )
            tip = _esc(
                (ex.get("modification") or ex.get("form_tip") or ex.get("notes") or "")[:220]
            )
            instr = _esc(
                (ex.get("instructions") or ex.get("description") or "")[:280]
            )
            raw_media = ex.get("gif_url") or ex.get("image_url") or ""
            cid = media_reg.register(raw_media)
            media_html = ""
            if cid:
                media_html = (
                    f'<div style="margin-top:8px">'
                    f'<img src="cid:{_esc(cid)}" alt="{name}" width="180" '
                    f'style="max-width:180px;border-radius:8px;border:1px solid #eee;'
                    f'display:block;background:#f3f4f6"/>'
                    f'<p style="font-size:11px;color:#888;margin:4px 0 0">Demo (embedded)</p>'
                    f"</div>"
                )
            elif raw_media:
                media_html = (
                    f'<p style="font-size:11px;color:#999;margin-top:6px">'
                    f"Demo unavailable to embed (missing/oversized file).</p>"
                )
            meta_html = (
                f'<br><span style="font-size:12px;color:#666">{meta}</span>' if meta else ""
            )
            instr_html = (
                f'<br><span style="font-size:12px;color:#444">{instr}</span>' if instr else ""
            )
            tip_html = (
                f'<br><em style="font-size:12px;color:#2c7a4b">{tip}</em>' if tip else ""
            )
            ex_rows.append(
                f'<tr><td style="padding:12px 0;border-top:1px solid #f0f0f0;vertical-align:top">'
                f'<strong style="color:#111">{name}</strong>'
                f"{meta_html}{instr_html}{tip_html}"
                f"{media_html}"
                f"</td></tr>"
            )

        exercises_table = (
            f'<table width="100%" cellpadding="0" cellspacing="0">{"".join(ex_rows)}</table>'
            if ex_rows else '<p style="color:#888;font-size:13px">No exercises listed.</p>'
        )
        note_html = (
            f'<p style="color:#666;font-size:13px;margin:0 0 8px">{note}</p>' if note else ""
        )
        day_blocks.append(
            f'<div style="border:1px solid #e5e7eb;border-radius:8px;padding:12px;margin:12px 0">'
            f'<h3 style="margin:0 0 6px;color:#2c7a4b">{label}</h3>'
            f"{note_html}"
            f"{exercises_table}"
            f"</div>"
        )

    week_html = "".join(day_blocks) or "<p>No weekly schedule in this plan.</p>"
    if str(plan_mode).lower() == "diet_only":
        week_html = "<p style=\"color:#666;font-size:13px\">Diet-only plan — no workout days.</p>"

    # ── Nutrition ────────────────────────────────────────────────────────────
    macro_line = ""
    if macros:
        macro_line = (
            f"<p style=\"font-size:13px;color:#444\">Macros (approx): "
            f"P {_esc(macros.get('protein_g'))}g · "
            f"C {_esc(macros.get('carbs_g') or macros.get('carb_g'))}g · "
            f"F {_esc(macros.get('fat_g'))}g</p>"
        )
    kcal_bits = []
    if diet.get("bmr"):
        kcal_bits.append(f"BMR {_esc(diet.get('bmr'))}")
    if diet.get("tdee"):
        kcal_bits.append(f"TDEE {_esc(diet.get('tdee'))}")
    if diet.get("target_calories") or diet.get("calorie_target", {}).get("target_calories"):
        tc = diet.get("target_calories") or diet.get("calorie_target", {}).get("target_calories")
        kcal_bits.append(f"Target {_esc(tc)}")
    if calories:
        kcal_bits.append(f"Meals ~{_esc(calories)} kcal")
    meta_kcal = (
        f"<p style=\"font-size:13px;color:#444\">{' · '.join(kcal_bits)}</p>"
        if kcal_bits else ""
    )
    if diet.get("calories_estimated") or diet.get("calorie_estimate_note"):
        meta_kcal += (
            f"<p style=\"font-size:12px;color:#b45309\">"
            f"{_esc(diet.get('calorie_estimate_note') or 'Calories use estimated height/weight.')}"
            f"</p>"
        )

    meal_blocks = []
    for meal in diet.get("meals") or []:
        title = _esc(_meal_label(meal.get("meal", "")))
        timing = _esc(meal.get("timing") or "")
        notes = _esc(meal.get("notes") or "")
        suggestions = meal.get("suggestions") or []
        sug_html = "".join(f"<li>{_esc(s)}</li>" for s in suggestions)
        nut = []
        if meal.get("calories"):
            nut.append(f"~{_esc(meal['calories'])} kcal")
        if meal.get("protein_g"):
            nut.append(f"P {_esc(meal['protein_g'])}g")
        if meal.get("carbs_g") or meal.get("carb_g"):
            nut.append(f"C {_esc(meal.get('carbs_g') or meal.get('carb_g'))}g")
        if meal.get("fat_g"):
            nut.append(f"F {_esc(meal['fat_g'])}g")
        nut_html = (
            f"<p style=\"font-size:12px;color:#666;margin:4px 0\">{' · '.join(nut)}</p>"
            if nut else ""
        )
        matched = ""
        if meal.get("matched_food"):
            matched = (
                f"<p style=\"font-size:11px;color:#999;margin:0\">INDB: "
                f"{_esc(meal['matched_food'])}</p>"
            )
        timing_html = (
            f' <span style="color:#888;font-size:12px">{timing}</span>' if timing else ""
        )
        sug_block = (
            f'<ul style="margin:6px 0;padding-left:18px;font-size:13px;color:#333">{sug_html}</ul>'
            if sug_html else ""
        )
        notes_html = (
            f'<p style="font-size:12px;color:#666;font-style:italic;margin:4px 0 0">{notes}</p>'
            if notes else ""
        )
        meal_blocks.append(
            f'<div style="border-left:3px solid #2c7a4b;padding:8px 12px;margin:10px 0;background:#fafafa">'
            f"<strong>{title}</strong>"
            f"{timing_html}"
            f"{nut_html}{matched}"
            f"{sug_block}"
            f"{notes_html}"
            f"</div>"
        )

    tips_india = "".join(
        f"<li>{_esc(t)}</li>" for t in (diet.get("india_specific_tips") or [])
    )
    avoid = "".join(f"<li>{_esc(f)}</li>" for f in (diet.get("foods_to_avoid") or []))
    hydration = _esc(diet.get("hydration") or "")

    diet_html = f"""
      <h2 style="color:#2c7a4b;border-bottom:2px solid #e8f5ee;padding-bottom:6px">Nutrition</h2>
      {meta_kcal}
      {macro_line}
      {''.join(meal_blocks) or '<p style="color:#888">No meal plan attached.</p>'}
      {f'<p style="font-size:13px"><strong>Hydration:</strong> {hydration}</p>' if hydration else ''}
      {f'<p style="font-size:13px;color:#b45309"><strong>India tips</strong></p><ul style="font-size:13px">{tips_india}</ul>' if tips_india else ''}
      {f'<p style="font-size:13px;color:#b91c1c"><strong>Foods to avoid</strong></p><ul style="font-size:13px;color:#b91c1c">{avoid}</ul>' if avoid else ''}
    """

    # ── Safety ───────────────────────────────────────────────────────────────
    safety_items = []
    for s in safety:
        if isinstance(s, dict):
            flag = _esc(s.get("flag") or "note")
            note = _esc(s.get("note") or "")
            cite = _esc(s.get("citation") or "")
            cite_html = f' <em style="color:#888">({cite})</em>' if cite else ""
            safety_items.append(
                f"<li><strong>{flag}</strong> — {note}{cite_html}</li>"
            )
        else:
            safety_items.append(f"<li>{_esc(s)}</li>")
    safety_html = (
        f'<h2 style="color:#2c7a4b;border-bottom:2px solid #e8f5ee;padding-bottom:6px">Safety</h2>'
        f'<ul style="font-size:13px;color:#333;line-height:1.5">{"".join(safety_items)}</ul>'
        if safety_items else ""
    )

    # ── Sources / citations ──────────────────────────────────────────────────
    cite_items = []
    for c in citations:
        if isinstance(c, dict):
            src = _esc(c.get("source") or c.get("source_name") or "Source")
            page = c.get("page") if c.get("page") is not None else c.get("page_number")
            used = _esc(c.get("used_for") or c.get("text") or "")
            page_bit = f", p.{_esc(page)}" if page is not None else ""
            cite_items.append(
                f"<li><strong>{src}{page_bit}</strong>"
                f"{f' — {used}' if used else ''}</li>"
            )
        else:
            cite_items.append(f"<li>{_esc(c)}</li>")
    cites_html = (
        f'<h2 style="color:#2c7a4b;border-bottom:2px solid #e8f5ee;padding-bottom:6px">'
        f"Sources &amp; citations</h2>"
        f'<ul style="font-size:13px;color:#333">{"".join(cite_items)}</ul>'
        if cite_items else ""
    )

    tips_html = ""
    if tips:
        tips_html = (
            f'<h2 style="color:#2c7a4b;border-bottom:2px solid #e8f5ee;padding-bottom:6px">'
            f"Weekly tips</h2><ul style=\"font-size:13px\">"
            + "".join(f"<li>{_esc(t)}</li>" for t in tips)
            + "</ul>"
        )

    n_img = len(media_reg.items)
    skip = media_reg.skipped
    media_note = (
        f"<p style=\"font-size:11px;color:#999;margin-top:20px\">"
        f"{n_img} demo image(s) embedded in this email"
        f"{f'; {skip} skipped (missing or over size cap)' if skip else ''}."
        f"</p>"
    )

    html = f"""<!DOCTYPE html><html><body style="font-family:Arial,Helvetica,sans-serif;max-width:680px;margin:auto;background:#f3f4f6;padding:16px">
  <div style="background:#2c7a4b;color:white;padding:20px;border-radius:8px 8px 0 0;text-align:center">
    <h1 style="margin:0;font-size:22px">Adaptive Fitness Planner</h1>
    <p style="margin:6px 0 0;opacity:0.9">Your full plan</p>
  </div>
  <div style="background:white;padding:24px;border-radius:0 0 8px 8px">
    <p>Hi {_esc(user_name)},</p>
    <p>Your <strong>{goal_label}</strong> plan is saved —
       <strong>{workout_days} workout day(s)</strong>
       {f' · ~<strong>{_esc(calories)}</strong> kcal/day' if calories else ''}
       · mode <strong>{_esc(plan_mode)}</strong>.</p>
    {f'<div style="margin:12px 0">{profile_html}</div>' if profile_html else ''}

    <h2 style="color:#2c7a4b;border-bottom:2px solid #e8f5ee;padding-bottom:6px">
      {"Yoga / workout week" if str(plan_mode).lower() == "yoga_only" else "Workout week"}
    </h2>
    {week_html}

    {diet_html}
    {safety_html}
    {cites_html}
    {tips_html}

    <p style="color:#666;font-size:13px;margin-top:24px">
      You can also open the app → <strong>Plan</strong> tab for the interactive view.
    </p>
    {media_note}
  </div>
</body></html>"""
    return html, media_reg.items


def send_plan_ready_email(user_name: str, to_email: str, goal: str, plan_data: dict) -> dict:
    """Email the full saved plan with demos embedded as inline MIME (not URL paths)."""
    goal_label = (goal or "fitness").replace("_", " ").title()
    html, inline_images = _build_full_plan_email_html(user_name, goal, plan_data)
    return _send_email(
        to_email,
        f"Your {goal_label} plan (full details)",
        html,
        inline_images=inline_images,
    )


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
