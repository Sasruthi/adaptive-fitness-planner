# Adaptive Fitness Planner

Personalised fitness assistant. Users chat in natural language; the system collects a safe profile, answers guideline-grounded questions (with demo media when useful), and builds a weekly workout + diet plan.

---

## End-to-end process flow

```text
┌─────────────┐     HTTP JSON      ┌──────────────┐
│  React UI   │ ─────────────────► │   FastAPI    │
│ Chat / Plan │ ◄───────────────── │   routers    │
│ / Progress  │                    └──────┬───────┘
└─────────────┘                           │
                                          ▼
                               ┌─────────────────────┐
                               │ LangGraph ReAct     │
                               │ conversation agent  │
                               │ (tool-calling loop) │
                               └──────────┬──────────┘
                                          │
          ┌───────────────┬───────────────┼───────────────┬──────────────┐
          ▼               ▼               ▼               ▼              ▼
   answer_fitness   update_profile   generate_plan   save / email   log / progress
   _question              │               │               │              │
          │               │               │               ▼              ▼
          │               ▼               │         SQLite + SMTP   SQLite
          │         ProfileStore          │
          │    (validated slots)          │
          ▼                               ▼
   ┌─────────────┐              ┌──────────────────┐
   │ Text RAG     │              │  plan_generator   │
   │ MiniLM +    │              │  1. schedule      │
   │ CLIP images │              │  2. LLM enrich    │
   └─────────────┘              │  3. INDB ground   │
          │                     └──────────────────┘
          ▼                               │
   Qdrant collections                     ▼
   • fitness_guidelines            Plan JSON → UI tabs
   • guideline_images              Workout / Nutrition / Safety
   • exercise_semantic
```

---

## 1. User chat flow

| Step | What happens |
|------|----------------|
| **Start** | `POST /api/conversation/start` → `thread_id` + greeting |
| **Each message** | `POST /api/conversation/message` → agent chooses tools |
| **Q&A** | `answer_fitness_question` → guideline text (+ yoga photos or exercise GIFs when appropriate) |
| **Profile** | `update_profile` → validated slots (goal, body parts, equipment, health flags, …) |
| **Plan** | When profile is safe/complete → `generate_plan` → full plan attached to the response |
| **Save** | Optional name/email → persist + email; Plan / Progress tabs update |

The agent decides the path. Plans are **never invented in chat prose** — only via `generate_plan`.

### Chat intents (Q&A routing)

| Intent | Behaviour |
|--------|-----------|
| `info` | Guidelines only (diet, water, protocol tables) — no exercise GIF spam |
| `exercise_qa` | Guidelines + a few targeted exercise demos |
| `plan` | Collect profile → `generate_plan` (not one-off tips) |

Media for yoga demos: text-page anchors first, then CLIP hybrid over booklet photos (`SRC009` / `SRC003`). Gym asks use the exercise catalog, not random PDF art.

---

## 2. Plan generation flow

```text
Profile + SQL/RAG filters
        │
        ▼
┌───────────────────────────┐
│ Retrieve guideline chunks │  ← Qdrant fitness_guidelines
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Schedule week (code)      │  ← equipment, injuries, split
│ or skip if diet_only      │     yoga_only → bodyweight yoga week
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ LLM enrichment            │  ← form tips, meal *names*, safety notes
│ (no invented macros)      │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Ground diet in INDB       │  ← real kcal/macros from nutrition_items
│ Mifflin BMR/TDEE (code)   │     height/weight optional (India midpoints if missing)
└─────────────┬─────────────┘
              ▼
     week_plan + diet_plan + safety_notes + citations
```

| `plan_mode` | Result |
|-------------|--------|
| `full` | 7-day workout + diet |
| `diet_only` | Meals / nutrition only (no workout days) |
| `yoga_only` | Yoga/asana week + light diet guidance |

Optional: `PLAN_AGENTIC=1` runs a short retrieve/validate LangGraph loop first; the **draft is discarded** and the final plan always comes from `plan_generator`.

---

## 3. Data pipelines (offline, before serving)

Exercises and guidelines are built separately. **Stop the API** before writing local Qdrant (single-writer lock).

```text
Exercise catalog                         Guidelines + images
─────────────────                        ───────────────────
fetch_external_sources.py                chunk_all_sources.py
        │                                        │
        ▼                                        ▼
build_exercise_corpus.py                 extract_pdf_images.py  → static/guideline_images/
        │                                        │
        ▼                                        ▼
load_db.py  → SQLite exercises           embed_and_store.py     → fitness_guidelines
        │                                embed_images_clip.py   → guideline_images
        ▼
embed_exercises.py → exercise_semantic

Optional: python -m app.scripts.load_nutrition_db  → nutrition_items (INDB)
```

---

## 4. Runtime architecture

| Layer | Role |
|-------|------|
| **Frontend** (`frontend/`) | Chat, Plan (Workout / Nutrition / Safety), Progress |
| **Routers** | Thin HTTP: conversation, plan, workout, exercises |
| **Conversation agent** | ReAct tools: Q&A, profile, plan, save, reminders |
| **Services** | RAG, NLU, exercise selection, plan generator, anthropometrics |
| **Action tools** | `save_plan`, calories, log workout, progress, email |
| **Stores** | Qdrant (vectors), SQLite (exercises, plans, nutrition, logs) |

LLM: Groq primary, Azure OpenAI failover when configured.

---

## Project layout

```text
adaptive-fitness-planner/
├── backend/
│   ├── main.py                 # uvicorn entry
│   ├── app/conversation/       # agent + profile store
│   ├── app/services/           # plan, RAG, NLU, exercises, nutrition
│   ├── app/tools/              # save / email / progress action tools
│   ├── app/routers/            # HTTP API
│   ├── rag/                    # chunk / embed / CLIP scripts + qdrant_local
│   └── scripts/                # exercise corpus → SQLite
├── frontend/                   # Vite + React + Tailwind
├── evals/                      # RAG / exercise / agent evals
├── data/                       # sources & media (large files gitignored)
├── .env.example                # copy to .env — never commit secrets
└── requirements.txt
```

---

## Quick start

### Prerequisites

- Python 3.10+, Node.js 18+
- Groq API key (Azure optional failover)
- Disk for Qdrant + exercise media

### Backend

```bash
cd backend
python3 -m venv ../env
source ../env/bin/activate          # Windows: ..\env\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` → `.env` (repo root and/or `backend/.env`):

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
# Optional: AZURE_OPENAI_*, SMTP_*
```

```bash
uvicorn main:app --reload --port 8000
```

- Health: http://localhost:8000/health  
- Swagger: http://localhost:8000/docs  

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL (usually http://localhost:5173) → **Chat**.

---

## API overview

| Area | Endpoints |
|------|-----------|
| Conversation | `POST /api/conversation/start`, `POST /api/conversation/message` |
| Plan | `POST /api/plan/generate`, `POST /api/plan/save`, `GET /api/plan/{email}`, `POST /api/plan/calories` |
| Workout | `POST /api/workout/log`, `GET /api/workout/progress/{email}`, `POST /api/workout/reminder` |
| Exercises | `POST /api/exercises/search`, `GET /api/exercises/taxonomy`, `GET /api/exercises/{id}` |

Chat already returns `plan` when generation succeeds — the Plan tab does not need a separate generate call for the normal UI path.

---

## Safety rules (enforced in code)

- Health flags must be confirmed before `generate_plan` (`none` or real flags).
- Injuries → hard body-part exclusions in exercise selection.
- Calorie math is Mifflin–St Jeor in code; meal macros come from INDB lookup, not LLM guesses.
- Estimated height/weight (India age/sex midpoints) is labeled in the Nutrition UI.

---

## Evals

```bash
cd backend && source ../env/bin/activate
python ../evals/run_all_evals.py
# or: run_rag_eval.py | run_multimodal_rag_eval.py | run_exercise_eval.py | run_agent_eval.py
```

Stop the API first if evals open the same local Qdrant path.

---

## Useful flags

| Variable | Meaning |
|----------|---------|
| `PLAN_AGENTIC=1` | Optional plan-agent loop before one deterministic `generate_plan` |
| `SEMANTIC_LLM_EXTRACT=1` | Extra LLM JSON extract in NLU (default off) |
| `QDRANT_PATH` | Override local Qdrant directory |
| `FRONTEND_URL` | Extra CORS origin |
