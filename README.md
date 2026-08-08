# Adaptive Fitness Planner

India-based personalised fitness assistant: chat to collect a profile, answer questions with guideline RAG, retrieve real exercises (with media), and generate a weekly workout + diet plan grounded in filters, Mifflin–St Jeor calories, and Indian nutrient data.

---

## What this project does

| Capability | Description |
|------------|-------------|
| **Conversational intake** | LangGraph ReAct agent collects goal, body parts, equipment, health flags, etc. |
| **Q&A with RAG** | Answers nutrition / guideline questions from India-based PDF sources (ICMR-NIN, WHO, FSSAI, Fit India, …) |
| **Exercise demos** | Semantic + SQL retrieval over a merged exercise catalog (GIF / image / video when available) |
| **Weekly plans** | Deterministic plan engine (injury filters, equipment, split) + LLM enrichment + INDB meal grounding |
| **Diet calories** | Mifflin–St Jeor BMR/TDEE; if height/weight missing, uses India age/sex midpoints (marked approximate) |
| **Progress** | Save plan, log workouts, streak / reminder email |

---

## How it works (architecture)

```text
React (Chat / Plan / Progress)
        │  HTTP JSON
        ▼
FastAPI routers          ← thin HTTP layer only
        │
        ▼
LangGraph conversation agent (ReAct + tools)
   • answer_fitness_question  → guideline Qdrant + exercise semantic search
   • update_profile           → validated profile slots
   • generate_plan            → plan pipeline (always filtered / grounded)
   • save / log / progress    → MCP + SQLite
        │
        ├── Qdrant: fitness_guidelines (chunked PDFs)
        ├── Qdrant: exercise_semantic  (one vector per exercise)
        ├── SQLite: exercises, users, plans, workouts, nutrition_items
        └── Optional PLAN_AGENTIC=1: second LangGraph retrieve/validate loop
            (draft discarded; final plan always from plan_generator)
```

### Two LangGraphs

1. **Conversation agent** (`backend/app/conversation/agent.py`) — every chat turn (Q&A, profile, plan, save).
2. **Plan agent** (`backend/app/services/plan_agent.py`) — optional (`PLAN_AGENTIC=1`); never returns raw agent JSON as the user plan.

### Turn intents (chat Q&A)

| Intent | Behaviour |
|--------|-----------|
| `info` | Guidelines only (e.g. water, carbs) — no exercise GIF spam |
| `exercise_qa` | Guidelines + a few targeted demos |
| `plan` | Profile collection → `generate_plan` |

---

## Project layout

```text
adaptive-fitness-planner/
├── backend/                 # FastAPI + agents + RAG + MCP
│   ├── main.py              # App entry (uvicorn)
│   ├── app/conversation/    # Chat agent, profile, filters
│   ├── app/services/        # Plan, RAG, NLU, exercises, anthropometrics
│   ├── app/mcp_server/      # save_plan, calories, log_workout, …
│   ├── app/routers/         # /api/conversation, /plan, /workout, /exercises
│   ├── rag/                 # Chunk/embed scripts + local Qdrant
│   └── scripts/             # Exercise corpus build + load_db
├── frontend/                # Vite + React + Tailwind
├── evals/                   # RAG / exercise / agent slot evals
├── data/                    # Guideline sources, media
├── main_scripts/            # Curated code snapshot + MODULE_DOCUMENTATION.md
└── SETUP_AND_TEST_GUIDE.md  # Detailed data-pipeline runbook
```


## Prerequisites

- Python 3.10+
- Node.js 18+ (frontend)
- Groq API key (default LLM)
- Disk space for Qdrant indexes and exercise media (local)

---

## Quick start

### 1. Backend

```bash
cd backend
python3 -m venv ../env          # or use existing env/
source ../env/bin/activate      # Windows: ..\env\Scripts\activate
pip install -r requirements.txt # or deps listed in SETUP_AND_TEST_GUIDE.md
```

Create a root `.env` (never commit this):

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Optional
# SMTP_HOST=... SMTP_USER=... SMTP_PASS=...
```

### 2. Data pipeline 

Exercises and guidelines are **separate** pipelines:

```bash
# Exercises → SQLite + exercise_semantic
python scripts/fetch_external_sources.py
python scripts/build_exercise_corpus.py
python scripts/load_db.py
python rag/embed_exercises.py

# Guidelines → fitness_guidelines
python rag/chunk_all_sources.py
python rag/embed_and_store.py

# Optional: Indian Nutrient Databank for meal macros
# python -m app.scripts.load_nutrition_db
```


### 3. Run API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

- Health: http://localhost:8000/health  
- Swagger: http://localhost:8000/docs  

### 4. Frontend

```bash
cd frontend
npm install
# optional: echo 'VITE_API_URL=http://localhost:8000' > .env
npm run dev
```

Open the URL Vite prints (usually http://localhost:5173) → **Chat** tab.

---

## Typical user flow

1. `POST /api/conversation/start` → `thread_id` + greeting  
2. Chat: ask facts or request a plan → agent tools update profile / answer with RAG  
3. When slots + health flags are complete → `generate_plan` builds week + diet  
4. Optional save/email → Plan & Progress tabs  


---

## API overview

| Area | Endpoints |
|------|-----------|
| Conversation | `POST /api/conversation/start`, `POST /api/conversation/message` |
| Plan | `POST /api/plan/generate`, `POST /api/plan/save`, `GET /api/plan/{email}`, `POST /api/plan/calories` |
| Workout | `POST /api/workout/log`, `GET /api/workout/progress/{email}`, `POST /api/workout/reminder` |
| Exercises | `POST /api/exercises/search`, `GET /api/exercises/taxonomy`, `GET /api/exercises/{id}` |

---

## Evals

```bash
cd backend && source ../env/bin/activate
python ../evals/run_all_evals.py
# or: run_rag_eval.py | run_exercise_eval.py | run_agent_eval.py
```

Measures Hit@k / MRR, generation groundedness, exercise compliance, and NLU slot accuracy. Suites exit non-zero when scores are below thresholds.

---

## Important environment flags

| Variable | Meaning |
|----------|---------|
| `PLAN_AGENTIC=1` | Run optional plan-agent tool loop before one deterministic `generate_plan` |
| `SEMANTIC_LLM_EXTRACT=1` | LLM JSON extract for demographics/health in NLU (default off) |
| `QDRANT_PATH` | Override local Qdrant directory |
| `FRONTEND_URL` | Extra CORS origin |

---

## Safety notes

- Plans are not invented in chat prose — only via `generate_plan`.  
- Health flags must be confirmed before planning (`none` or real flags).  
- Injuries map to hard body-part exclusions in SQL/semantic selection.  
- Calorie math is code (Mifflin); estimated size is labeled in the Diet UI.

---


