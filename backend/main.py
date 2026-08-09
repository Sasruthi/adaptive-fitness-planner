"""
FILE LOCATION: backend/main.py

Adaptive Fitness Planner — FastAPI Backend
==========================================
Entry point. Wires all routers, CORS, and health check.

Run locally:
  cd backend
  uvicorn main:app --reload --port 8000

API docs (auto-generated):
  http://localhost:8000/docs     ← Swagger UI
  http://localhost:8000/redoc    ← ReDoc
"""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env before any app imports so SMTP/GROQ keys are available everywhere.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers.conversation import router as conversation_router
from app.routers.plan         import router as plan_router
from app.routers.workout      import router as workout_router
from app.routers.exercises    import router as exercises_router

# ── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Adaptive Fitness Planner API",
    description="""
India-first personalised fitness planning API.
Combines conversational AI intake, RAG-grounded guideline retrieval,
structured exercise retrieval, and action tools (save / email / progress).

## Flow
1. `POST /api/conversation/start` → get thread_id + greeting
2. `POST /api/conversation/message` (repeat until slots_complete=true)
3. `POST /api/plan/generate` → full 7-day plan
4. `POST /api/plan/save` → persist to DB
5. `POST /api/workout/log` → track completions
6. `GET  /api/workout/progress/{email}` → streak + stats
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow frontend (dev + prod) ────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
if os.getenv("FRONTEND_URL"):
    ALLOWED_ORIGINS.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(conversation_router)
app.include_router(plan_router)
app.include_router(workout_router)
app.include_router(exercises_router)

# Local hasaneyldrm media (images/ + videos/) linked under backend/static/exercises/
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)
# follow_symlink=True required — images/videos are symlinked to data/hasaneyldrm-exercises-dataset/
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR), follow_symlink=True), name="static")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health():
    """
    Health check — verifies DB and Qdrant are reachable.
    Use this on Render to confirm the backend is up.
    """
    checks = {}

    # Check SQLite
    try:
        from app.services.exercise_retrieval import Session
        from app.models.models import Exercise
        s = Session()
        count = s.query(Exercise).count()
        s.close()
        checks["sqlite"] = f"ok ({count} exercises)"
    except Exception as e:
        checks["sqlite"] = f"error: {str(e)}"

    # Check Qdrant
    try:
        from app.services.rag_retrieval import get_qdrant, COLLECTION_NAME
        client = get_qdrant()
        count  = client.count(COLLECTION_NAME).count
        checks["qdrant"] = f"ok ({count} vectors)"
    except Exception as e:
        checks["qdrant"] = f"error: {str(e)}"

    # Check Azure OpenAI env vars are set
    checks["groq"] = (
        "configured" if os.getenv("GROQ_API_KEY") else "missing — set GROQ_API_KEY in .env"
    )

    overall = "healthy" if all("error" not in v for v in checks.values()) else "degraded"

    return {
        "status":    overall,
        "version":   "1.0.0",
        "modules":   checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "name":    "Adaptive Fitness Planner API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }
